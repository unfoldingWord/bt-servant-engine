"""CLI commands for API key management."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import typer
from rich.console import Console
from rich.table import Table

from bt_servant_engine.adapters.api_keys import APIKeyAdapter
from bt_servant_engine.core.api_key_models import KEY_PREFIX_LENGTH
from bt_servant_engine.services.api_keys import APIKeyService

app = typer.Typer(name="keys", help="Manage API keys")
console = Console()


def _get_service() -> APIKeyService:
    """Get the API key service with default adapter."""
    adapter = APIKeyAdapter()
    return APIKeyService(adapter)


@app.command("create")
def create_key(
    name: str = typer.Option(..., "--name", "-n", help="Human-readable name for the key"),
    env: str = typer.Option("prod", "--env", "-e", help="Environment: prod, staging, or dev"),
    rate_limit: int = typer.Option(60, "--rate-limit", "-r", help="Requests per minute"),
    expires_days: int | None = typer.Option(None, "--expires", help="Days until expiration"),
) -> None:
    """Create a new API key."""
    service = _get_service()

    expires_at = None
    if expires_days:
        expires_at = datetime.now(timezone.utc) + timedelta(days=expires_days)

    try:
        result = service.create_key(
            name=name,
            environment=env,
            rate_limit_per_minute=rate_limit,
            expires_at=expires_at,
        )
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

    console.print("\n[green]API key created successfully![/green]\n")
    console.print(f"[bold]Key ID:[/bold] {result.key.id}")
    console.print(f"[bold]Name:[/bold] {result.key.name}")
    console.print(f"[bold]Environment:[/bold] {result.key.environment.value}")
    console.print(f"[bold]Rate Limit:[/bold] {result.key.rate_limit_per_minute} req/min")
    if expires_at:
        console.print(f"[bold]Expires:[/bold] {expires_at.isoformat()}")

    console.print("\n[yellow]IMPORTANT: Save this key now - it will not be shown again![/yellow]")
    console.print(f"\n[bold cyan]{result.raw_key}[/bold cyan]\n")


@app.command("list")
def list_keys(
    include_revoked: bool = typer.Option(False, "--all", "-a", help="Include revoked keys"),
    env: str | None = typer.Option(None, "--env", "-e", help="Filter by environment"),
) -> None:
    """List all API keys."""
    service = _get_service()
    keys = service.list_keys(include_revoked=include_revoked, environment=env)

    if not keys:
        console.print("[dim]No API keys found.[/dim]")
        return

    table = Table(title="API Keys")
    table.add_column("Prefix", style="cyan")
    table.add_column("Name")
    table.add_column("Env")
    table.add_column("Status")
    table.add_column("Created")
    table.add_column("Last Used")
    table.add_column("Rate Limit")

    for key in keys:
        status_str = "[green]active[/green]" if key.is_active else "[red]revoked[/red]"
        if key.expires_at and key.expires_at < datetime.now(timezone.utc):
            status_str = "[yellow]expired[/yellow]"

        table.add_row(
            key.key_prefix,
            key.name,
            key.environment.value,
            status_str,
            key.created_at.strftime("%Y-%m-%d"),
            key.last_used_at.strftime("%Y-%m-%d %H:%M") if key.last_used_at else "-",
            str(key.rate_limit_per_minute),
        )

    console.print(table)


@app.command("revoke")
def revoke_key(
    key_identifier: str = typer.Argument(..., help="Key prefix or full key to revoke"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Revoke an API key."""
    service = _get_service()

    # Try to find key by prefix
    prefix = (
        key_identifier[:KEY_PREFIX_LENGTH]
        if len(key_identifier) > KEY_PREFIX_LENGTH
        else key_identifier
    )

    if not force:
        confirm = typer.confirm(f"Are you sure you want to revoke key '{prefix}'?")
        if not confirm:
            console.print("[dim]Aborted.[/dim]")
            raise typer.Exit(0)

    success = service.revoke_key_by_prefix(prefix)

    if success:
        console.print(f"[green]Key '{prefix}' has been revoked.[/green]")
    else:
        console.print(f"[red]Key '{prefix}' not found or already revoked.[/red]")
        raise typer.Exit(1)


__all__ = ["app"]
