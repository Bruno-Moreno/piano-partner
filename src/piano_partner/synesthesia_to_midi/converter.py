from __future__ import annotations

from pathlib import Path

import cv2

from piano_partner.common.paths import DEFAULT_MIDI_DIR
from piano_partner.synesthesia_to_midi.detector import (
    assign_tracks_by_color,
    detect_note_events,
    filter_background_events,
)
from piano_partner.synesthesia_to_midi.keyboard import (
    KeyboardLayout,
    apply_overrides,
    auto_detect_layout,
)
from piano_partner.synesthesia_to_midi.midi_writer import write_midi


def _video_duration_s(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return n / fps if fps > 0 else 0.0
    finally:
        cap.release()


def convert_video(
    video_path: Path,
    output_path: Path | None = None,
    layout_overrides: dict[str, int] | None = None,
    tracks: int = 1,
) -> tuple[Path, KeyboardLayout, int]:
    """Detect notes in a Synthesia-style video and write a MIDI file.

    With ``tracks > 1``, events are clustered by hue and emitted as that many
    MIDI tracks — typical Synthesia coloring puts left/right hands on different
    hues, so ``tracks=2`` separates them. Single-color videos are auto-detected
    and stay on a single track.

    Returns the output path, the layout used, and the number of notes written.
    """
    overrides = layout_overrides or {}
    if {"x0", "x1", "y_trigger"}.issubset(overrides):
        layout = KeyboardLayout(
            x0=overrides["x0"],
            x1=overrides["x1"],
            y_trigger=overrides["y_trigger"],
            y_black=overrides.get("y_black"),
        )
    else:
        layout = auto_detect_layout(video_path)
        if overrides:
            layout = apply_overrides(layout, **overrides)

    events = detect_note_events(video_path, layout)
    events = filter_background_events(events, _video_duration_s(video_path))
    if tracks > 1:
        events = assign_tracks_by_color(events, tracks)

    if output_path is None:
        output_path = DEFAULT_MIDI_DIR / video_path.with_suffix(".mid").name
    write_midi(events, output_path)
    return output_path, layout, len(events)
