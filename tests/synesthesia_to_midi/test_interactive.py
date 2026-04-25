import numpy as np
import pytest

from piano_partner.synesthesia_to_midi.interactive import (
    _layout_from_sliders,
    _render_overlay,
)
from piano_partner.synesthesia_to_midi.keyboard import (
    KeyboardLayout,
    all_key_samples,
)


def _blank_frame(h: int = 200, w: int = 1040) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_render_overlay_draws_x_lines():
    frame = _blank_frame()
    layout = KeyboardLayout(x0=20, x1=1000, y_trigger=150)
    annotated = _render_overlay(frame, layout)
    # Cyan in BGR is (255, 255, 0). Inspect the x0 column above the trigger
    # row so it isn't overwritten by the green horizontal line.
    column = annotated[: layout.y_trigger, layout.x0]
    assert np.all(column[:, 0] == 255)
    assert np.all(column[:, 1] == 255)
    assert np.all(column[:, 2] == 0)


def test_render_overlay_draws_trigger_row():
    frame = _blank_frame()
    layout = KeyboardLayout(x0=20, x1=1000, y_trigger=150)
    annotated = _render_overlay(frame, layout)
    # Pick a column outside the keyboard span (left of x0) so no sample-box
    # rectangles overwrite the trigger pixel.
    pixel = annotated[layout.y_trigger, 5]
    assert tuple(pixel) == (0, 255, 0)


def test_render_overlay_marks_sample_points():
    frame = _blank_frame()
    layout = KeyboardLayout(x0=20, x1=1000, y_trigger=150)
    annotated = _render_overlay(frame, layout)
    # Each key's sample box is drawn as an outline rectangle. Black-key boxes
    # are magenta — at least one such pixel must exist.
    is_magenta = (
        (annotated[..., 0] == 255)
        & (annotated[..., 1] == 0)
        & (annotated[..., 2] == 255)
    )
    assert is_magenta.any()


def test_render_overlay_draws_box_outlines_not_filled():
    """Rectangle outlines must not fill the box interior. Sample one row
    inside the box but off the trigger line (which would otherwise overwrite
    the center) — that pixel must stay at the original background color."""
    frame = _blank_frame()
    layout = KeyboardLayout(x0=20, x1=1000, y_trigger=150)
    annotated = _render_overlay(frame, layout)
    samples = all_key_samples(layout)
    white = next(s for s in samples if not s.is_black)
    # white.y == y_trigger, half_h == 2 → box spans rows y±2. y-1 is interior.
    assert tuple(annotated[white.y - 1, white.x]) == (0, 0, 0)


def test_render_overlay_handles_invalid_x():
    frame = _blank_frame()
    layout = KeyboardLayout(x0=500, x1=400, y_trigger=150)
    annotated = _render_overlay(frame, layout)
    # Should annotate without crashing — and since x1<x0 we skip drawing the
    # sample circles, so most of the frame is still black.
    assert annotated.shape == frame.shape
    # Some red pixels must exist from the error caption.
    has_red = ((annotated[..., 2] > 100) & (annotated[..., 0] < 50)).any()
    assert has_red


def test_render_overlay_does_not_mutate_input():
    frame = _blank_frame()
    layout = KeyboardLayout(x0=20, x1=1000, y_trigger=150)
    _render_overlay(frame, layout)
    assert not frame.any()


def test_layout_from_sliders_zero_y_black_means_unset():
    layout = _layout_from_sliders(x0=10, x1=100, y_trigger=50, y_black_raw=0)
    assert layout.y_black is None
    assert layout.effective_y_black == 50


def test_layout_from_sliders_positive_y_black_passes_through():
    layout = _layout_from_sliders(x0=10, x1=100, y_trigger=50, y_black_raw=42)
    assert layout.y_black == 42


def test_render_overlay_clips_offscreen_samples():
    """If sliders push x1 wildly out of bounds, samples beyond the frame must
    not raise — they should silently be skipped."""
    frame = _blank_frame(h=200, w=1040)
    layout = KeyboardLayout(x0=20, x1=10000, y_trigger=150)
    annotated = _render_overlay(frame, layout)
    assert annotated.shape == frame.shape


def test_render_overlay_y_black_line():
    frame = _blank_frame()
    layout = KeyboardLayout(x0=20, x1=1000, y_trigger=150, y_black=120)
    annotated = _render_overlay(frame, layout)
    # Sample a column not crossed by the cyan x-lines or sample circles.
    # Pick x=100 (past x0 but between sample circles).
    samples = all_key_samples(layout)
    sample_xs = {s.x for s in samples}
    test_x = next(x for x in range(50, 200) if x not in sample_xs)
    assert tuple(annotated[layout.y_black, test_x]) == (255, 0, 255)


@pytest.mark.parametrize("y_trigger", [0, 199])
def test_render_overlay_handles_edge_y(y_trigger):
    frame = _blank_frame(h=200, w=1040)
    layout = KeyboardLayout(x0=20, x1=1000, y_trigger=y_trigger)
    annotated = _render_overlay(frame, layout)
    assert annotated.shape == frame.shape
