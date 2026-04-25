import mido

from piano_partner.synesthesia_to_midi.detector import NoteEvent
from piano_partner.synesthesia_to_midi.midi_writer import write_midi


def test_round_trip(tmp_path):
    events = [
        NoteEvent(midi_note=60, start_s=0.0, end_s=0.5),
        NoteEvent(midi_note=64, start_s=0.25, end_s=0.75),
        NoteEvent(midi_note=67, start_s=0.5, end_s=1.0),
    ]
    out = tmp_path / "test.mid"
    write_midi(events, out)

    mid = mido.MidiFile(str(out))
    notes_on = [m.note for m in mid.tracks[0] if m.type == "note_on"]
    notes_off = [m.note for m in mid.tracks[0] if m.type == "note_off"]
    assert sorted(notes_on) == [60, 64, 67]
    assert sorted(notes_off) == [60, 64, 67]


def test_durations_preserved(tmp_path):
    events = [NoteEvent(midi_note=60, start_s=1.0, end_s=2.0)]
    out = tmp_path / "test.mid"
    write_midi(events, out)

    mid = mido.MidiFile(str(out))
    # Sum total time across messages — should be ~2 seconds.
    total = sum(m.time for m in mid.play())
    assert 1.9 <= total <= 2.1


def test_multi_track_split(tmp_path):
    events = [
        NoteEvent(midi_note=60, start_s=0.0, end_s=0.5, track=0),
        NoteEvent(midi_note=72, start_s=0.0, end_s=0.5, track=1),
    ]
    out = tmp_path / "split.mid"
    write_midi(events, out)

    mid = mido.MidiFile(str(out))
    assert len(mid.tracks) == 2
    notes_t0 = [m.note for m in mid.tracks[0] if m.type == "note_on"]
    notes_t1 = [m.note for m in mid.tracks[1] if m.type == "note_on"]
    assert notes_t0 == [60]
    assert notes_t1 == [72]
