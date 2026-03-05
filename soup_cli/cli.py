"""Main CLI entry point — all commands registered here."""

import typer
from rich.console import Console

from soup_cli import __version__
from soup_cli.commands import (
    chat,
    data,
    diff,
    eval,
    export,
    generate,
    init,
    merge,
    push,
    runs,
    serve,
    sweep,
    train,
)

console = Console()

app = typer.Typer(
    name="soup",
    help="Fine-tune LLMs in one command. No SSH, no config hell.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Register sub-commands
app.command()(init.init)
app.command()(train.train)
app.command()(chat.chat)
app.command()(push.push)
app.command(name="export")(export.export)
app.command()(merge.merge)
app.add_typer(
    data.app, name="data",
    help="Dataset tools: inspect, convert, merge, dedup, validate, stats.",
)
app.add_typer(runs.app, name="runs", help="Experiment tracking: list, show, compare runs.")
app.command(name="eval")(eval.eval_model)
app.command()(serve.serve)
app.command()(sweep.sweep)
app.command(name="diff")(diff.diff)

# Register data generate as a subcommand of data
data.app.command(name="generate")(generate.generate)


@app.command()
def version():
    """Show Soup CLI version."""
    console.print(f"[bold green]soup[/] v{__version__}")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Soup — fine-tune LLMs in one command."""
    pass
