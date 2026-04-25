from pathlib import Path

import typer

from piano_partner.synesthesia_to_midi.converter import convert_video
from piano_partner.synesthesia_to_midi.interactive import tune_layout_interactive

app = typer.Typer(help="Convert piano synesthesia videos to MIDI.", no_args_is_help=True)


@app.command()
def convert(
    video: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output .mid path."),
    x0: int | None = typer.Option(None, "--x0", help="Override: left edge of leftmost white key (px)."),
    x1: int | None = typer.Option(None, "--x1", help="Override: right edge of rightmost white key (px)."),
    y_trigger: int | None = typer.Option(None, "--y-trigger", help="Override: sample row above keyboard (px)."),
    y_black: int | None = typer.Option(None, "--y-black", help="Override: sample row for black keys (px); defaults to y_trigger."),
    tracks: int = typer.Option(1, "--tracks", min=1, max=8, help="Split detected notes into N tracks by hue (e.g. 2 = left/right hand)."),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Open a window to set x0/x1/y_trigger/y_black on a video frame before converting.",
    ),
) -> None:
    """Convert a piano synesthesia video into a MIDI file."""
    if interactive and any(v is not None for v in (x0, x1, y_trigger, y_black)):
        raise typer.BadParameter(
            "--interactive cannot be combined with --x0/--x1/--y-trigger/--y-black; "
            "the interactive tuner sets those itself."
        )

    overrides: dict[str, int] = {
        k: v
        for k, v in dict(x0=x0, x1=x1, y_trigger=y_trigger, y_black=y_black).items()
        if v is not None
    }

    if interactive:
        tuned = tune_layout_interactive(video)
        for key, value in tuned.items():
            if value is not None:
                overrides[key] = value

    path, layout, n_events = convert_video(
        video, output_path=output, layout_overrides=overrides, tracks=tracks
    )
    typer.echo(
        f"Layout: x0={layout.x0} x1={layout.x1} y_trigger={layout.y_trigger} "
        f"y_black={layout.effective_y_black} (white-key width {layout.white_key_width:.1f}px)"
    )
    typer.echo(f"Notes: {n_events} (tracks: {tracks})")
    typer.echo(f"Saved: {path}")
