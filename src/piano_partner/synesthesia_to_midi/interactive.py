from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from piano_partner.synesthesia_to_midi.keyboard import (
    KeyboardLayout,
    all_key_samples,
    layout_from_frame,
)

WINDOW_NAME = "Synesthesia Tuner - Enter=accept, Esc=cancel"

# BGR colors for the overlay
_CYAN = (255, 255, 0)
_GREEN = (0, 255, 0)
_MAGENTA = (255, 0, 255)
_RED = (0, 0, 255)
_WHITE = (255, 255, 255)


@dataclass(frozen=True)
class _VideoMeta:
    cap: cv2.VideoCapture
    frame_count: int
    width: int
    height: int


def _open_video(video_path: Path) -> _VideoMeta:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_count <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Video has invalid metadata: {video_path}")
    return _VideoMeta(cap=cap, frame_count=frame_count, width=width, height=height)


def _read_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
    ok, frame = cap.read()
    return frame if ok else None


def _seed_defaults(meta: _VideoMeta) -> dict[str, int]:
    """Seed slider defaults from the existing auto-detect heuristic, falling
    back to a centred guess if auto-detect raises on this video."""
    midpoint = max(0, meta.frame_count // 2)
    frame = _read_frame(meta.cap, midpoint)
    if frame is not None:
        try:
            layout = layout_from_frame(frame)
            return {
                "x0": layout.x0,
                "x1": layout.x1,
                "y_trigger": layout.y_trigger,
                "y_black": 0,  # 0 means "unset → falls back to y_trigger"
            }
        except RuntimeError:
            pass
    return {
        "x0": int(meta.width * 0.05),
        "x1": int(meta.width * 0.95),
        "y_trigger": int(meta.height * 0.85),
        "y_black": 0,
    }


def _render_overlay(frame: np.ndarray, layout: KeyboardLayout) -> np.ndarray:
    """Draw the keyboard layout overlay on a copy of ``frame``.

    Pure: no cv2 GUI calls, safe to run headlessly in tests.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    if layout.x1 <= layout.x0 or layout.white_key_width < 1.0:
        cv2.putText(
            annotated,
            "invalid x0/x1",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            _RED,
            2,
        )
        return annotated

    cv2.line(annotated, (layout.x0, 0), (layout.x0, h - 1), _CYAN, 1)
    x1_draw = min(layout.x1, w - 1)
    cv2.line(annotated, (x1_draw, 0), (x1_draw, h - 1), _CYAN, 1)

    if 0 <= layout.y_trigger < h:
        cv2.line(annotated, (0, layout.y_trigger), (w - 1, layout.y_trigger), _GREEN, 1)
    if (
        layout.y_black is not None
        and layout.y_black != layout.y_trigger
        and 0 <= layout.y_black < h
    ):
        cv2.line(annotated, (0, layout.y_black), (w - 1, layout.y_black), _MAGENTA, 1)

    for sample in all_key_samples(layout):
        if not (0 <= sample.x < w and 0 <= sample.y < h):
            continue
        color = _MAGENTA if sample.is_black else _GREEN
        x_lo = max(0, sample.x - sample.half_w)
        x_hi = min(w - 1, sample.x + sample.half_w)
        y_lo = max(0, sample.y - sample.half_h)
        y_hi = min(h - 1, sample.y + sample.half_h)
        cv2.rectangle(annotated, (x_lo, y_lo), (x_hi, y_hi), color, 1)

    return annotated


def _layout_from_sliders(x0: int, x1: int, y_trigger: int, y_black_raw: int) -> KeyboardLayout:
    return KeyboardLayout(
        x0=x0,
        x1=x1,
        y_trigger=y_trigger,
        y_black=y_black_raw if y_black_raw > 0 else None,
    )


def _run_loop(meta: _VideoMeta, defaults: dict[str, int]) -> dict[str, int | None]:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    def _noop(_: int) -> None:
        return None

    cv2.createTrackbar("frame", WINDOW_NAME, 0, max(1, meta.frame_count - 1), _noop)
    cv2.createTrackbar("x0", WINDOW_NAME, defaults["x0"], max(1, meta.width - 1), _noop)
    cv2.createTrackbar("x1", WINDOW_NAME, defaults["x1"], max(1, meta.width - 1), _noop)
    cv2.createTrackbar(
        "y_trigger", WINDOW_NAME, defaults["y_trigger"], max(1, meta.height - 1), _noop
    )
    cv2.createTrackbar(
        "y_black (0=off)", WINDOW_NAME, defaults["y_black"], max(1, meta.height - 1), _noop
    )

    cached_idx: int | None = None
    cached_frame: np.ndarray | None = None

    while True:
        frame_idx = cv2.getTrackbarPos("frame", WINDOW_NAME)
        if frame_idx != cached_idx:
            new_frame = _read_frame(meta.cap, frame_idx)
            if new_frame is not None:
                cached_frame = new_frame
                cached_idx = frame_idx

        if cached_frame is None:
            raise RuntimeError("Could not read any frame from the video for tuning")

        x0 = cv2.getTrackbarPos("x0", WINDOW_NAME)
        x1 = cv2.getTrackbarPos("x1", WINDOW_NAME)
        y_trigger = cv2.getTrackbarPos("y_trigger", WINDOW_NAME)
        y_black_raw = cv2.getTrackbarPos("y_black (0=off)", WINDOW_NAME)
        layout = _layout_from_sliders(x0, x1, y_trigger, y_black_raw)

        annotated = _render_overlay(cached_frame, layout)
        cv2.putText(
            annotated,
            f"frame {frame_idx}/{meta.frame_count - 1}  "
            f"x0={x0} x1={x1} y={y_trigger} yb={y_black_raw or 'off'}",
            (10, annotated.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _WHITE,
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW_NAME, annotated)

        key = cv2.waitKey(30) & 0xFF
        if key == 13:  # Enter
            return {
                "x0": x0,
                "x1": x1,
                "y_trigger": y_trigger,
                "y_black": y_black_raw if y_black_raw > 0 else None,
            }
        if key == 27:  # Esc
            raise KeyboardInterrupt("Layout tuning cancelled by user")


def tune_layout_interactive(video_path: Path) -> dict[str, int | None]:
    """Open an OpenCV window letting the user pick the keyboard geometry on
    a frame of ``video_path``. Returns a dict with ``x0``, ``x1``,
    ``y_trigger``, and ``y_black`` (which may be ``None``) — ready to splat
    into ``convert_video``'s ``layout_overrides``.
    """
    meta = _open_video(video_path)
    try:
        defaults = _seed_defaults(meta)
        return _run_loop(meta, defaults)
    finally:
        meta.cap.release()
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except cv2.error:
            pass
