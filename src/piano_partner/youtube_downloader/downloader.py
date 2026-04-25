from pathlib import Path

import yt_dlp


def _format_selector(max_height: int | None) -> str:
    h = f"[height<={max_height}]" if max_height else ""
    return (
        f"bestvideo{h}[ext=mp4]+bestaudio[ext=m4a]/"
        f"best{h}[ext=mp4]/best{h}"
    )


def download_video(
    url: str,
    output_dir: Path,
    max_height: int | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": _format_selector(max_height),
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": False,
        "noprogress": False,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return Path(ydl.prepare_filename(info)).with_suffix(".mp4")
