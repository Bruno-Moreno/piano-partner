import typer

app = typer.Typer(help="Follow-along piano teaching app (work in progress).")


@app.callback(invoke_without_command=True)
def main() -> None:
    typer.echo("Teaching app: not yet implemented.")
