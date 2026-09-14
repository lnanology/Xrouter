"""xrouter CLI. Run as `python3 -m app.cli <command>` (or via the `xrouter`
wrapper installed by scripts/install-cli.sh, if present)."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import click
import httpx

ROOT = Path(__file__).resolve().parents[1]
PID_FILE = ROOT / "data" / "xrouter.pid"
LOG_FILE = ROOT / "data" / "xrouter.log"


def _port() -> int:
    return int(os.environ.get("XROUTER_PORT", "20128"))


def _base_url() -> str:
    return f"http://localhost:{_port()}"


@click.group()
def cli():
    """XRouter — OpenAI-compatible AI intelligence gateway."""


@cli.command()
def start():
    """Start the XRouter server in the background."""
    subprocess.run(["bash", str(ROOT / "scripts" / "start.sh")], check=True)


@cli.command()
def stop():
    """Stop the background XRouter server."""
    if not PID_FILE.exists():
        click.echo("No pid file found — is XRouter running?")
        return
    pid = int(PID_FILE.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        click.echo(f"Sent SIGTERM to pid {pid}")
    except ProcessLookupError:
        click.echo("Process not found (already stopped?)")
    PID_FILE.unlink(missing_ok=True)


@cli.command()
def status():
    """Check /health on the running server."""
    try:
        resp = httpx.get(f"{_base_url()}/health", timeout=5.0)
        click.echo(json.dumps(resp.json(), indent=2))
    except httpx.ConnectError:
        click.echo("XRouter is not reachable — is it running? (`xrouter start`)")
        sys.exit(1)


@cli.command()
def models():
    """List available models (GET /v1/models)."""
    resp = httpx.get(f"{_base_url()}/v1/models", timeout=10.0)
    click.echo(json.dumps(resp.json(), indent=2))


@cli.command()
def providers():
    """List providers and their health (GET /health/providers)."""
    resp = httpx.get(f"{_base_url()}/health/providers", timeout=10.0)
    click.echo(json.dumps(resp.json(), indent=2))


@cli.command()
def health():
    """Alias for `status`."""
    status.callback()


@cli.command()
def benchmark():
    """Run scripts/benchmark.py directly against configured providers."""
    subprocess.run([sys.executable, str(ROOT / "scripts" / "benchmark.py")], check=False)


@cli.command()
def config():
    """Print the resolved (non-secret) configuration."""
    sys.path.insert(0, str(ROOT))
    from app.core.config import get_settings

    settings = get_settings()
    click.echo(
        json.dumps(
            {
                "server": {"host": settings.server.host, "port": settings.server.port},
                "routing": {"default_policy": settings.routing.default_policy},
                "cache": {"enabled": settings.cache.enabled},
                "providers": {pid: {"type": p.type, "enabled": p.enabled} for pid, p in settings.providers.items()},
            },
            indent=2,
        )
    )


@cli.command()
@click.option("--lines", "-n", default=50)
def logs(lines: int):
    """Tail the background server's log file."""
    if not LOG_FILE.exists():
        click.echo("No log file yet.")
        return
    subprocess.run(["tail", "-n", str(lines), str(LOG_FILE)])


if __name__ == "__main__":
    cli()
