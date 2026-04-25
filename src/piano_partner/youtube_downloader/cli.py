from pathlib import Path

import typer

from piano_partner.common.paths import DEFAULT_VIDEOS_DIR
from piano_partner.youtube_downloader.downloader import download_video

app = typer.Typer(help="Download YouTube videos as MP4.", no_args_is_help=True)


@app.command()
def download(
    url: str = typer.Argument(..., help="YouTube video URL."),
    output_dir: Path = typer.Option(
        DEFAULT_VIDEOS_DIR, "--output-dir", "-o", help="Where to save the MP4."
    ),
    max_height: int | None = typer.Option(
        None, "--max-height", "-q", help="Cap resolution, e.g. 1080 or 720."
    ),
) -> None:
    """Download a single YouTube video as MP4 (video + audio merged)."""
    path = download_video(url, output_dir=output_dir, max_height=max_height)
    typer.echo(f"Saved: {path}")
