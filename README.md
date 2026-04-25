# piano-partner

Tools to assist piano lessons:

1. **`youtube`** — download YouTube videos as MP4.
2. **`synesthesia`** — convert piano synesthesia videos into MIDI _(work in progress)_.
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

### Future tools

```bash
poetry run piano-partner synesthesia   # WIP
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
