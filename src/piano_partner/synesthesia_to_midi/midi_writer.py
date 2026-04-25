from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import mido

from piano_partner.synesthesia_to_midi.detector import NoteEvent

TICKS_PER_BEAT = 480
TEMPO = 500_000  # 120 BPM in microseconds per beat
DEFAULT_VELOCITY = 80


def write_midi(events: list[NoteEvent], output_path: Path, velocity: int = DEFAULT_VELOCITY) -> Path:
    """Write events to a MIDI file with one track per distinct ``event.track``.

    The first track always carries the tempo meta-message. With single-track
    input, output is a single-track file (back-compat with prior behavior).
    """
    by_track: dict[int, list[NoteEvent]] = defaultdict(list)
    for ev in events:
        by_track[ev.track].append(ev)
    track_ids = sorted(by_track) or [0]

    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)

    for n, track_id in enumerate(track_ids):
        track = mido.MidiTrack()
        mid.tracks.append(track)
        if n == 0:
            track.append(mido.MetaMessage("set_tempo", tempo=TEMPO, time=0))

        timeline: list[tuple[float, int, int]] = []
        for ev in by_track[track_id]:
            timeline.append((ev.start_s, 1, ev.midi_note))
            timeline.append((ev.end_s, 0, ev.midi_note))
        timeline.sort()

        last_tick = 0
        for time_s, type_order, note in timeline:
            abs_tick = int(round(mido.second2tick(time_s, TICKS_PER_BEAT, TEMPO)))
            delta = max(0, abs_tick - last_tick)
            msg_type = "note_on" if type_order == 1 else "note_off"
            vel = velocity if type_order == 1 else 0
            track.append(mido.Message(msg_type, note=note, velocity=vel, time=delta))
            last_tick = abs_tick

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(output_path))
    return output_path
