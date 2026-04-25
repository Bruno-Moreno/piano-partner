import typer

app = typer.Typer(help="Convert piano synesthesia videos to MIDI (work in progress).")


@app.callback(invoke_without_command=True)
def main() -> None:
    typer.echo("synesthesia → MIDI: not yet implemented.")
