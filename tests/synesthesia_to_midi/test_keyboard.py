import numpy as np
import pytest

from piano_partner.synesthesia_to_midi.keyboard import (
    HIGHEST_MIDI,
    LOWEST_MIDI,
    N_WHITE_KEYS,
    KeyboardLayout,
    all_key_samples,
    apply_overrides,
    is_black_key,
    layout_from_frame,
)


def test_88_keys():
    layout = KeyboardLayout(x0=0, x1=520, y_trigger=10)
    samples = all_key_samples(layout)
    assert len(samples) == 88
    assert samples[0].midi_note == LOWEST_MIDI
    assert samples[-1].midi_note == HIGHEST_MIDI


def test_white_and_black_counts():
    layout = KeyboardLayout(x0=0, x1=520, y_trigger=10)
    samples = all_key_samples(layout)
    whites = [s for s in samples if not s.is_black]
    blacks = [s for s in samples if s.is_black]
    assert len(whites) == N_WHITE_KEYS
    assert len(blacks) == 36


def test_black_key_pitch_classes():
    blacks = {21 + i for i in range(88) if is_black_key(21 + i)}
    assert all((m % 12) in {1, 3, 6, 8, 10} for m in blacks)


def test_white_keys_are_evenly_spaced():
    layout = KeyboardLayout(x0=0, x1=520, y_trigger=10)
    samples = all_key_samples(layout)
    whites = [s.x for s in samples if not s.is_black]
    diffs = [whites[i + 1] - whites[i] for i in range(len(whites) - 1)]
    assert max(diffs) - min(diffs) <= 1


def test_all_samples_share_y_when_y_black_unset():
    layout = KeyboardLayout(x0=0, x1=520, y_trigger=99)
    samples = all_key_samples(layout)
    assert {s.y for s in samples} == {99}


def test_y_black_overrides_only_black_keys():
    layout = KeyboardLayout(x0=0, x1=520, y_trigger=20, y_black=10)
    samples = all_key_samples(layout)
    assert {s.y for s in samples if s.is_black} == {10}
    assert {s.y for s in samples if not s.is_black} == {20}


def test_apply_overrides_partial():
    layout = KeyboardLayout(x0=10, x1=520, y_trigger=99)
    new = apply_overrides(layout, x0=42, y_trigger=None)
    assert new == KeyboardLayout(x0=42, x1=520, y_trigger=99)


def test_apply_overrides_y_black():
    layout = KeyboardLayout(x0=10, x1=520, y_trigger=99)
    new = apply_overrides(layout, y_black=80)
    assert new.effective_y_black == 80


def test_black_keys_use_balanced_groups_not_boundaries():
    """C# / D# / F# / A# should NOT sit on the boundary between adjacent
    whites; they're shifted to match real piano CAD geometry. G# is the only
    black key that sits exactly between its neighboring whites."""
    layout = KeyboardLayout(x0=0, x1=5200, y_trigger=10)
    w = layout.white_key_width  # 100.0
    by_midi = {s.midi_note: s for s in all_key_samples(layout)}
    # C#4 (61): white left edge at C4 (white index 23 from A0): x = 23*100 = 2300.
    # Boundary model would put C#4 at 2400; balanced model at 2300 + 0.75*100 = 2375.
    assert by_midi[61].x == 2375
    # D#4 (63): balanced at 2300 + 2.25*100 = 2525 (boundary would be 2500).
    assert by_midi[63].x == 2525
    # G#4 (68): balanced at 2300 + 5.0*100 = 2800 (same as boundary — G# is centered).
    assert by_midi[68].x == 2800


def test_sample_boxes_are_sized_by_key_width():
    """Larger keyboards should get proportionally larger sample boxes."""
    small = all_key_samples(KeyboardLayout(x0=0, x1=520, y_trigger=10))
    large = all_key_samples(KeyboardLayout(x0=0, x1=2600, y_trigger=10))
    s_white = next(s for s in small if not s.is_black)
    l_white = next(s for s in large if not s.is_black)
    assert l_white.half_w > s_white.half_w


def test_black_key_box_smaller_than_white_key_box():
    samples = all_key_samples(KeyboardLayout(x0=0, x1=2600, y_trigger=10))
    white = next(s for s in samples if not s.is_black)
    black = next(s for s in samples if s.is_black)
    assert black.half_w < white.half_w


def _make_synthetic_keyboard_frame(
    h: int = 200,
    w: int = 1040,
    keyboard_top: int = 120,
    has_ui_bar_above: bool = False,
) -> np.ndarray:
    """Synthesize a frame: dark background, then at ``keyboard_top`` an
    alternating bright/dark column pattern (52 white keys + black-key gaps),
    optionally with a uniform bright UI bar above the keyboard. The bar is
    brighter on average than the keyboard but has no transitions, so the
    edge-count auto-detect should ignore it.
    """
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    if has_ui_bar_above:
        frame[keyboard_top - 30 : keyboard_top - 10, :, :] = 240  # uniform bar
    white_w = w // N_WHITE_KEYS
    for i in range(N_WHITE_KEYS):
        x_start = i * white_w
        # white key column at the top of the keyboard region
        frame[keyboard_top : keyboard_top + 60, x_start + 1 : x_start + white_w - 1] = 250
        # black-key gap (every other boundary, simplified)
        if i % 2 == 0:
            frame[keyboard_top : keyboard_top + 30, x_start + white_w - 2 : x_start + white_w + 2] = 0
    return frame


def test_layout_from_frame_finds_keyboard_top():
    frame = _make_synthetic_keyboard_frame(keyboard_top=120)
    layout = layout_from_frame(frame)
    # Trigger row should sit on the key body (below the keyboard top edge).
    assert layout.y_trigger > 120
    assert layout.y_trigger <= 120 + 30
    # Layout should span most of the frame width.
    assert layout.white_key_width > 15.0


def test_layout_from_frame_ignores_uniform_ui_bar():
    """A uniform bright bar above the keyboard must NOT be picked over the
    actual keyboard, even though its row mean brightness is higher."""
    frame = _make_synthetic_keyboard_frame(keyboard_top=140, has_ui_bar_above=True)
    layout = layout_from_frame(frame)
    # The UI bar lives at y=110..130; the keyboard top is at 140. Trigger should
    # be below the keyboard top, never inside or above the UI bar.
    assert layout.y_trigger > 140
    assert layout.y_trigger < 140 + 30


def test_layout_from_frame_raises_on_non_keyboard_frame():
    frame = np.zeros((200, 1040, 3), dtype=np.uint8)
    frame[150:170, :, :] = 200  # uniform bright band, no transitions
    with pytest.raises(RuntimeError, match="alternating bright/dark pattern"):
        layout_from_frame(frame)


def test_layout_from_frame_rejects_partial_keyboard():
    """A detected keyboard must span ~the full frame width — synesthesia
    videos always show the entire 88-key piano, so anything smaller is
    almost certainly a misdetection (UI element, partial crop, etc.)."""
    h, w = 200, 3500
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    keyboard_top = 120
    # 52 white keys spanning ~1560px (~45% of the 3500px-wide frame). Below
    # 50% so the keyboard-span check should reject this layout.
    white_w = 30
    for i in range(N_WHITE_KEYS):
        x_start = i * white_w
        frame[keyboard_top : keyboard_top + 60, x_start + 1 : x_start + white_w - 1] = 250
        if i % 2 == 0:
            frame[keyboard_top : keyboard_top + 30, x_start + white_w - 2 : x_start + white_w + 2] = 0
    with pytest.raises(RuntimeError, match="spans"):
        layout_from_frame(frame)
