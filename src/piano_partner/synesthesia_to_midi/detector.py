from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from piano_partner.synesthesia_to_midi.keyboard import KeyboardLayout, all_key_samples


@dataclass(frozen=True)
class NoteEvent:
    midi_note: int
    start_s: float
    end_s: float
    color: tuple[int, int, int] = (0, 0, 0)  # BGR captured at note-on
    track: int = 0


def detect_events_from_frames(
    frames: Iterable[np.ndarray],
    fps: float,
    layout: KeyboardLayout,
    value_threshold: int = 100,
    saturation_threshold: int = 50,
    min_duration_s: float = 0.06,
    min_lit_fraction: float = 0.5,
) -> list[NoteEvent]:
    """Detect note events from a stream of BGR frames.

    For each key, a small box around its sample center is examined every frame.
    A pixel is "lit" if its HSV value AND saturation both exceed their
    thresholds — that catches Synthesia falling blocks of any color while
    rejecting the dark background and grayscale UI elements (hands, white
    keys). The key itself is considered lit when more than
    ``min_lit_fraction`` of pixels in its box are lit, which is far less
    sensitive to single-pixel noise than sampling one point.
    """
    samples = all_key_samples(layout)
    n = len(samples)
    midi_notes = np.array([s.midi_note for s in samples], dtype=np.int32)

    boxes: list[tuple[int, int, int, int]] = []
    for s in samples:
        boxes.append(
            (
                max(0, s.y - s.half_h),
                s.y + s.half_h + 1,
                max(0, s.x - s.half_w),
                s.x + s.half_w + 1,
            )
        )

    prev_lit = np.zeros(n, dtype=bool)
    start_frame = np.full(n, -1, dtype=np.int64)
    start_color = np.zeros((n, 3), dtype=np.uint8)
    events: list[NoteEvent] = []
    frame_idx = 0

    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lit_mask = (hsv[..., 2] > value_threshold) & (hsv[..., 1] > saturation_threshold)

        lit = np.zeros(n, dtype=bool)
        for i, (y_lo, y_hi, x_lo, x_hi) in enumerate(boxes):
            box = lit_mask[y_lo:y_hi, x_lo:x_hi]
            if box.size == 0:
                continue
            lit[i] = box.mean() > min_lit_fraction

        new_on = lit & ~prev_lit
        new_off = prev_lit & ~lit

        for i in np.flatnonzero(new_on):
            y_lo, y_hi, x_lo, x_hi = boxes[i]
            box_bgr = frame[y_lo:y_hi, x_lo:x_hi]
            if box_bgr.size == 0:
                continue
            start_frame[i] = frame_idx
            start_color[i] = np.median(box_bgr.reshape(-1, 3), axis=0)

        for i in np.flatnonzero(new_off):
            if start_frame[i] >= 0:
                events.append(
                    NoteEvent(
                        midi_note=int(midi_notes[i]),
                        start_s=float(start_frame[i] / fps),
                        end_s=float(frame_idx / fps),
                        color=tuple(int(c) for c in start_color[i]),
                    )
                )
                start_frame[i] = -1

        prev_lit = lit
        frame_idx += 1

    end_s = frame_idx / fps
    for i in np.flatnonzero(prev_lit):
        if start_frame[i] >= 0:
            events.append(
                NoteEvent(
                    midi_note=int(midi_notes[i]),
                    start_s=float(start_frame[i] / fps),
                    end_s=float(end_s),
                    color=tuple(int(c) for c in start_color[i]),
                )
            )

    events = [e for e in events if (e.end_s - e.start_s) >= min_duration_s]
    events.sort(key=lambda e: (e.start_s, e.midi_note))
    return events


def _video_frames(cap: cv2.VideoCapture) -> Iterator[np.ndarray]:
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            return
        yield frame


def detect_note_events(
    video_path: Path,
    layout: KeyboardLayout,
    value_threshold: int = 100,
    saturation_threshold: int = 50,
    min_duration_s: float = 0.06,
) -> list[NoteEvent]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        return detect_events_from_frames(
            _video_frames(cap),
            fps,
            layout,
            value_threshold=value_threshold,
            saturation_threshold=saturation_threshold,
            min_duration_s=min_duration_s,
        )
    finally:
        cap.release()


def assign_tracks_by_color(
    events: list[NoteEvent],
    n_tracks: int,
    single_hue_threshold: float = 0.95,
) -> list[NoteEvent]:
    """Cluster events into ``n_tracks`` tracks by hue similarity.

    Hue is projected onto the unit circle (cos, sin) before k-means so that
    near-wraparound colors don't get split incorrectly. If event hues all
    cluster tightly (mean resultant length R > ``single_hue_threshold``), the
    video uses a single block color and clustering would produce a meaningless
    split — fall back to a single track in that case.
    """
    if n_tracks <= 1 or len(events) < n_tracks:
        return [replace(e, track=0) for e in events]

    bgr = np.array([e.color for e in events], dtype=np.uint8).reshape(-1, 1, 3)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    angles = hsv[:, 0].astype(np.float32) * (2.0 * np.pi / 180.0)
    cs = np.cos(angles)
    sn = np.sin(angles)
    R = float(np.sqrt(cs.mean() ** 2 + sn.mean() ** 2))
    if R > single_hue_threshold:
        return [replace(e, track=0) for e in events]

    samples = np.column_stack([cs, sn]).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, _ = cv2.kmeans(
        samples, n_tracks, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.flatten()
    return [replace(e, track=int(labels[i])) for i, e in enumerate(events)]


def filter_background_events(
    events: list[NoteEvent],
    total_span_s: float,
    threshold: float = 0.5,
) -> list[NoteEvent]:
    """Drop events for any pitch that's "lit" more than ``threshold`` of the video.

    Persistent UI columns (measure lines, watermarks, frame borders) sample as
    permanently saturated/bright at certain x positions, which the per-frame
    detector misreads as one extremely long note per affected key. A real song
    rarely sustains a single pitch for more than half the piece, so any pitch
    with total lit time above the threshold is treated as background and
    removed entirely.
    """
    if total_span_s <= 0 or not events:
        return events
    by_note: dict[int, float] = {}
    for ev in events:
        by_note[ev.midi_note] = by_note.get(ev.midi_note, 0.0) + (ev.end_s - ev.start_s)
    background = {n for n, total in by_note.items() if total > threshold * total_span_s}
    return [e for e in events if e.midi_note not in background]
