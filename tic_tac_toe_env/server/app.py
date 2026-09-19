# SPDX-License-Identifier: BSD-3-Clause

"""
FastAPI application for the Tic Tac Toe Environment.

This module creates an HTTP server that exposes the TicTacToeEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import TicTacToeAction, TicTacToeObservation
    from .gradio_board import mount_tic_tac_toe_board
    from .tic_tac_toe_env_environment import TicTacToeEnvironment
except ImportError:
    from models import TicTacToeAction, TicTacToeObservation
    from server.gradio_board import mount_tic_tac_toe_board
    from server.tic_tac_toe_env_environment import TicTacToeEnvironment


# Create the app for the REST/WebSocket API (/reset, /step, /ws, /docs, /health).
app = create_app(
    TicTacToeEnvironment,
    TicTacToeAction,
    TicTacToeObservation,
    env_name="tic_tac_toe_env",
    max_concurrent_envs=4,  # allow several concurrent WebSocket game sessions
)

# create_app may or may not have already mounted its own web interface at
# /web depending on the ENABLE_WEB_INTERFACE env var (and, we found, that
# gate's behavior isn't consistent across deployment environments — see
# gradio_board.py's module docstring). Strip any /web routes it added so
# our own board (mounted next) is unambiguously the only thing served
# there, regardless of what create_app decided to do internally.
app.router.routes = [route for route in app.router.routes if not getattr(route, "path", "").startswith("/web")]

# Mount the clickable board UI at /web ourselves. This makes /web always
# available, with no env var required.
mount_tic_tac_toe_board(app, TicTacToeEnvironment, TicTacToeAction, TicTacToeObservation)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m tic_tac_toe_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn tic_tac_toe_env.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
