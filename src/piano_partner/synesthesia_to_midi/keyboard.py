from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

LOWEST_MIDI = 21  # A0
HIGHEST_MIDI = 108  # C8
N_WHITE_KEYS = 52
BLACK_PITCH_CLASSES = {1, 3, 6, 8, 10}

# x-center of each pitch class within an octave, expressed in white-key-widths
# from the C-left edge of that octave. Black keys use the "balanced groups"
# model: the 2-group (C#, D#) is evenly distributed across the C–D–E section
# and the 3-group (F#, G#, A#) across the F–G–A–B section. This matches real
# piano CAD geometry better than placing each black key on the boundary
# between its neighboring whites.
_PITCH_CLASS_X = {
    0: 0.5,            # C
    1: 0.75,           # C# (2-group, leftmost)
    2: 1.5,            # D
    3: 2.25,           # D# (2-group, rightmost)
    4: 2.5,            # E
    5: 3.5,            # F
    6: 3.0 + 4.0 / 6,  # F# (3-group, leftmost)  ≈ 3.667
    7: 4.5,            # G
    8: 3.0 + 4.0 / 2,  # G# (3-group, middle)    = 5.0
    9: 5.5,            # A
    10: 3.0 + 20.0 / 6,  # A# (3-group, rightmost) ≈ 6.333
    11: 6.5,           # B
}

_A0_OCTAVE = LOWEST_MIDI // 12          # 1
_A0_PITCH_X = _PITCH_CLASS_X[LOWEST_MIDI % 12]  # 5.5 (A inside C-octave)

# Sample-box dimensions as fractions of the white-key width.
_WHITE_BOX_W_FRAC = 0.35
_BLACK_KEY_W_FRAC = 0.6      # black-key visual width relative to white-key width
_BLACK_BOX_W_FRAC = _BLACK_KEY_W_FRAC * _WHITE_BOX_W_FRAC


def is_black_key(midi_note: int) -> bool:
    return (midi_note % 12) in BLACK_PITCH_CLASSES


@dataclass(frozen=True)
class KeyboardLayout:
    x0: int                      # left edge of leftmost white key (A0)
    x1: int                      # right edge of rightmost white key (C8)
    y_trigger: int               # sample row for white keys (just above the keyboard)
    y_black: int | None = None   # sample row for black keys; falls back to y_trigger when None

    @property
    def white_key_width(self) -> float:
        return (self.x1 - self.x0) / N_WHITE_KEYS

    @property
    def effective_y_black(self) -> int:
        return self.y_black if self.y_black is not None else self.y_trigger


@dataclass(frozen=True)
class KeySample:
    midi_note: int
    x: int            # center x of the sample box
    y: int            # center y of the sample box
    is_black: bool
    half_w: int       # half-width of the sample box (box spans [x-half_w, x+half_w])
    half_h: int       # half-height of the sample box


def all_key_samples(layout: KeyboardLayout) -> list[KeySample]:
    """Return one sample box per piano key, A0 through C8.

    White keys sit at the center of each white-key column. Black keys use the
    balanced-groups model so they match real piano geometry: C# / D# are
    evenly distributed across C–D–E, F# / G# / A# across F–G–A–B. Each sample
    has a box (``half_w``, ``half_h``) sized as a fraction of the white-key
    width, so the detector can sample a region instead of a single pixel.
    """
    w = layout.white_key_width
    yw = layout.y_trigger
    yb = layout.effective_y_black

    half_w_white = max(2, int(round(w * _WHITE_BOX_W_FRAC)))
    half_w_black = max(1, int(round(w * _BLACK_BOX_W_FRAC)))
    half_h = 2

    samples: list[KeySample] = []
    for midi in range(LOWEST_MIDI, HIGHEST_MIDI + 1):
        oct_idx = midi // 12
        pc = midi % 12
        pitch_x = _PITCH_CLASS_X[pc]
        # White-key-widths from A0's left edge to this key's center.
        offset_w = (oct_idx - _A0_OCTAVE) * 7.0 + (pitch_x - _A0_PITCH_X + 0.5)
        x = layout.x0 + offset_w * w
        is_b = is_black_key(midi)
        samples.append(
            KeySample(
                midi_note=midi,
                x=int(round(x)),
                y=yb if is_b else yw,
                is_black=is_b,
                half_w=half_w_black if is_b else half_w_white,
                half_h=half_h,
            )
        )
    return samples


def auto_detect_layout(video_path: Path) -> KeyboardLayout:
    """Detect the keyboard bounding box from a frame near the middle of the video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n_frames // 2))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read sample frame for layout detection")
    finally:
        cap.release()
    return layout_from_frame(frame)


def layout_from_frame(
    frame: np.ndarray,
    edge_threshold: int = 50,
    min_transitions: int = 50,
    min_row_brightness: int = 80,
    sample_offset: int = 20,
    min_keyboard_span_frac: float = 0.5,
) -> KeyboardLayout:
    """Locate the keyboard by finding the topmost row that has many strong
    horizontal brightness transitions (alternating white/black keys).

    A keyboard top edge crosses ~88 key boundaries, so transitions per row are
    high; bright UI bars above the keyboard look uniform and have few
    transitions. The trigger row is then placed *below* the keyboard top, on
    the key body where pressed keys get tinted by the falling block — that
    avoids vertical UI lines and measure markers that live in the falling-block
    area above the keyboard.

    Assumes a full 88-key piano: ``min_keyboard_span_frac`` rejects detections
    where the bright row covers less than that fraction of the frame width
    (a partial keyboard or a narrow UI element rather than a full piano).
    """
    h, frame_w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    bottom_start = int(h * 0.4)
    region = gray[bottom_start:].astype(np.int16)
    diffs = np.abs(np.diff(region, axis=1))
    transitions = (diffs > edge_threshold).sum(axis=1)
    row_mean = region.mean(axis=1)

    valid = (transitions >= min_transitions) & (row_mean >= min_row_brightness)
    if not valid.any():
        raise RuntimeError(
            "Auto-detect failed: no row in the lower frame shows the alternating "
            "bright/dark pattern of a piano keyboard."
        )
    # Topmost valid row is the keyboard's top edge.
    y_top = int(np.argmax(valid)) + bottom_start
    y_trigger = min(h - 1, y_top + sample_offset)

    row = gray[y_top]
    bright_idx = np.where(row > 150)[0]
    if len(bright_idx) == 0:
        raise RuntimeError("Auto-detect failed: no bright pixels at detected keyboard top edge")
    x0 = int(bright_idx[0])
    x1 = int(bright_idx[-1]) + 1

    if (x1 - x0) < frame_w * min_keyboard_span_frac:
        raise RuntimeError(
            f"Auto-detect failed: detected keyboard spans {x1 - x0}px "
            f"({(x1 - x0) / frame_w:.0%} of frame width); a full 88-key piano should "
            f"cover at least {min_keyboard_span_frac:.0%}. Use --interactive to set "
            "x0/x1 manually."
        )

    white_w = (x1 - x0) / N_WHITE_KEYS
    if not (8.0 <= white_w <= 50.0):
        raise RuntimeError(
            f"Auto-detect failed: white-key width {white_w:.1f}px is out of range. "
            "Pass --x0/--x1/--y-trigger manually."
        )
    return KeyboardLayout(x0=x0, x1=x1, y_trigger=y_trigger)


def apply_overrides(layout: KeyboardLayout, **overrides: int | None) -> KeyboardLayout:
    """Return a new layout with the given fields overridden (None values are ignored)."""
    valid = {k: v for k, v in overrides.items() if v is not None}
    return replace(layout, **valid)
