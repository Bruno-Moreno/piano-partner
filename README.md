# piano-partner

Tools to assist piano lessons:

1. **`youtube`** — download YouTube videos as MP4.
2. **`synesthesia`** — convert piano synesthesia videos into MIDI.
3. **`teach`** — connect a MIDI keyboard and follow along with a MIDI file _(work in progress)_.

## Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/)
- [`ffmpeg`](https://ffmpeg.org/) on `PATH` (used to merge best video + audio streams)

## Install

```bash
poetry install
```

## Usage

All tools are exposed under a single CLI:

```bash
poetry run piano-partner --help
```

### Download a YouTube video

```bash
poetry run piano-partner youtube download "https://www.youtube.com/watch?v=..."
```

Files land in `data/videos/<title>.mp4` by default. Override with `-o` and cap resolution with `-q`:

```bash
poetry run piano-partner youtube download <URL> -o ./elsewhere -q 720
```

### Convert a synesthesia video to MIDI

```bash
poetry run piano-partner synesthesia convert "data/videos/Tchaikovsky – Swan Lake.mp4"
```

The keyboard bounding box is auto-detected from a frame near the middle of the
video. If detection picks the wrong region, override any of the four geometry
fields:

```bash
poetry run piano-partner synesthesia convert <video> \
  --x0 80 --x1 1840 --y-trigger 980 --y-black 920
```

For videos where eyeballing pixel coordinates is hard, use `--interactive` (or
`-i`) to open a window with sliders for `x0`, `x1`, `y_trigger`, `y_black`, and
a frame seek. The 88 sample points are drawn live on the frame so you can see
the detected layout before converting. Press Enter to accept and continue,
Esc to cancel.

```bash
poetry run piano-partner synesthesia convert <video> --interactive
```

Output lands at `data/midi/<title>.mid` by default; override with `-o`. Sustain
and velocity dynamics are not captured — every note is written at a fixed
velocity.

### Future tools

```bash
poetry run piano-partner teach         # WIP
```

## Repo layout

```
src/piano_partner/
├── cli.py                  # unified CLI entry point
├── common/                 # shared utilities (paths, etc.)
├── youtube_downloader/     # tool 1
├── synesthesia_to_midi/    # tool 2 (stub)
└── teaching_app/           # tool 3 (stub)
```

Artifacts (downloaded videos, MIDI, model caches) are written to a gitignored `data/` directory at the repo root.
