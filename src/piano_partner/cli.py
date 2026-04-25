import typer

from piano_partner.synesthesia_to_midi.cli import app as synesthesia_app
from piano_partner.teaching_app.cli import app as teach_app
from piano_partner.youtube_downloader.cli import app as youtube_app

app = typer.Typer(help="Piano Partner — tools to support piano lessons.", no_args_is_help=True)
app.add_typer(youtube_app, name="youtube", help="YouTube video downloader.")
app.add_typer(synesthesia_app, name="synesthesia", help="Synesthesia video → MIDI (WIP).")
app.add_typer(teach_app, name="teach", help="Follow-along teaching app (WIP).")


if __name__ == "__main__":
    app()
