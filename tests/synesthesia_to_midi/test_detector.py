import numpy as np

from piano_partner.synesthesia_to_midi.detector import (
    NoteEvent,
    assign_tracks_by_color,
    detect_events_from_frames,
    filter_background_events,
)
from piano_partner.synesthesia_to_midi.keyboard import (
    KeyboardLayout,
    all_key_samples,
)


def _blank(h=40, w=520):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _light_key(frame, x, y, color_bgr):
    frame[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3] = color_bgr


def test_detects_simple_press_and_release():
    layout = KeyboardLayout(x0=0, x1=520, y_trigger=10)
    samples = all_key_samples(layout)
    middle_c = next(s for s in samples if s.midi_note == 60)

    blue = (255, 100, 0)  # BGR
    frames = [
        _blank(),
        _press := _blank(),
        _press2 := _blank(),
        _blank(),
    ]
    _light_key(_press, middle_c.x, middle_c.y, blue)
    _light_key(_press2, middle_c.x, middle_c.y, blue)

    events = detect_events_from_frames(frames, fps=30.0, layout=layout, min_duration_s=0.0)
    assert len(events) == 1
    ev = events[0]
    assert ev.midi_note == 60
    assert ev.color == blue
    assert ev.start_s == 1 / 30.0
    assert ev.end_s == 3 / 30.0


def test_filters_short_notes_below_min_duration():
    layout = KeyboardLayout(x0=0, x1=520, y_trigger=10)
    middle_c = next(s for s in all_key_samples(layout) if s.midi_note == 60)
    blink = _blank()
    _light_key(blink, middle_c.x, middle_c.y, (0, 200, 255))
    events = detect_events_from_frames(
        [_blank(), blink, _blank()], fps=30.0, layout=layout, min_duration_s=0.1
    )
    assert events == []


def test_unsaturated_pixel_is_not_lit():
    """White / gray pixels (S~0) must not register as a press, only saturated colors do."""
    layout = KeyboardLayout(x0=0, x1=520, y_trigger=10)
    middle_c = next(s for s in all_key_samples(layout) if s.midi_note == 60)
    bright_gray = _blank()
    _light_key(bright_gray, middle_c.x, middle_c.y, (240, 240, 240))
    events = detect_events_from_frames(
        [_blank(), bright_gray, _blank()], fps=30.0, layout=layout, min_duration_s=0.0
    )
    assert events == []


def test_black_keys_use_y_black_row():
    """A black key is detected only when the lit pixel is on its y_black row, not y_trigger."""
    layout = KeyboardLayout(x0=0, x1=520, y_trigger=20, y_black=10)
    samples = all_key_samples(layout)
    cs = next(s for s in samples if s.midi_note == 61)  # C#4, a black key
    assert cs.is_black and cs.y == 10

    on_black_row = _blank()
    _light_key(on_black_row, cs.x, 10, (255, 0, 100))
    events = detect_events_from_frames(
        [_blank(), on_black_row, _blank()], fps=30.0, layout=layout, min_duration_s=0.0
    )
    assert [e.midi_note for e in events] == [61]

    # Same color at the white-key row should NOT trigger the black key.
    on_white_row_only = _blank()
    on_white_row_only[18:23, cs.x - 2 : cs.x + 3] = (255, 0, 100)
    on_white_row_only[8:13, :] = 0  # ensure black row is clean
    events = detect_events_from_frames(
        [_blank(), on_white_row_only, _blank()], fps=30.0, layout=layout, min_duration_s=0.0
    )
    assert all(e.midi_note != 61 for e in events)


def test_assign_tracks_splits_by_hue():
    """Two distinct hues should get distinct track ids."""
    events = [
        NoteEvent(midi_note=60, start_s=0.0, end_s=0.5, color=(255, 0, 0)),    # blue
        NoteEvent(midi_note=64, start_s=0.1, end_s=0.5, color=(255, 50, 30)),  # blue-ish
        NoteEvent(midi_note=67, start_s=0.2, end_s=0.5, color=(0, 200, 0)),    # green
        NoteEvent(midi_note=72, start_s=0.3, end_s=0.5, color=(30, 220, 40)),  # green-ish
    ]
    out = assign_tracks_by_color(events, n_tracks=2)
    tracks = [e.track for e in out]
    # Pairs (0,1) and (2,3) should land in the same cluster respectively.
    assert tracks[0] == tracks[1]
    assert tracks[2] == tracks[3]
    assert tracks[0] != tracks[2]


def test_assign_tracks_noop_for_one_track():
    events = [NoteEvent(60, 0.0, 0.5, color=(10, 20, 30))]
    assert all(e.track == 0 for e in assign_tracks_by_color(events, n_tracks=1))


def test_assign_tracks_too_few_events_keeps_single_track():
    events = [NoteEvent(60, 0.0, 0.5, color=(10, 20, 30))]
    assert all(e.track == 0 for e in assign_tracks_by_color(events, n_tracks=2))


def test_assign_tracks_single_hue_video_stays_one_track():
    """A monochrome Synthesia video should NOT be split into 2 tracks even with
    --tracks=2: clustering a single hue produces a meaningless random split."""
    same_blue = (255, 100, 30)
    events = [NoteEvent(60 + i, 0.1 * i, 0.1 * i + 0.4, color=same_blue) for i in range(10)]
    out = assign_tracks_by_color(events, n_tracks=2)
    assert {e.track for e in out} == {0}


def test_filter_background_drops_persistent_pitches():
    """A pitch lit for >50% of the video span is treated as a background UI
    artifact and removed; normal short notes are kept."""
    span = 100.0
    events = [
        NoteEvent(60, 0.0, 0.5),  # short, kept
        NoteEvent(60, 1.0, 1.5),  # short, kept
        NoteEvent(98, 0.0, 80.0),  # one stuck "note", 80% of span — background
        NoteEvent(101, 0.0, 60.0),  # also background
    ]
    out = filter_background_events(events, total_span_s=span, threshold=0.5)
    assert {e.midi_note for e in out} == {60}


def test_box_sampling_ignores_single_pixel_noise():
    """A single noisy pixel inside an otherwise-dark sample box must not
    register as a press — the detector requires a majority of the box to be
    lit, which is the whole point of box sampling over single-pixel sampling."""
    layout = KeyboardLayout(x0=0, x1=520, y_trigger=10)
    middle_c = next(s for s in all_key_samples(layout) if s.midi_note == 60)
    speckle = _blank()
    # One bright saturated pixel right at the sample center.
    speckle[middle_c.y, middle_c.x] = (255, 0, 100)
    events = detect_events_from_frames(
        [_blank(), speckle, _blank()], fps=30.0, layout=layout, min_duration_s=0.0
    )
    assert events == []


def test_filter_background_keeps_legitimate_repeated_notes():
    """A pitch struck many times but never sustained beyond half the span is kept."""
    span = 100.0
    events = [NoteEvent(60, 2.0 * i, 2.0 * i + 0.5) for i in range(20)]  # 10s total, 10% of span
    out = filter_background_events(events, total_span_s=span, threshold=0.5)
    assert len(out) == 20
