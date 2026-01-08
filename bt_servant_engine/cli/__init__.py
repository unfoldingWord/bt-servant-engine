"""CLI commands for bt-servant-engine."""

import typer

from bt_servant_engine.cli.keys import app as keys_app

main_app = typer.Typer(
    name="bt-servant",
    help="BT Servant Engine CLI",
    no_args_is_help=True,
)
main_app.add_typer(keys_app, name="keys")


def main() -> None:
    """Entry point for the CLI."""
    main_app()


__all__ = ["main", "main_app"]
