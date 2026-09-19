# SPDX-License-Identifier: BSD-3-Clause

"""
Custom `/web` board UI for the Tic Tac Toe environment.

OpenEnv's `create_app` auto-generates a generic form for any environment
(a raw "cell" number field plus a JSON dump of the response), which works
but forces you to read the board out of JSON on every move. This module
builds an actual clickable 3x3 grid instead.

`create_app` also accepts a `gradio_builder` hook that's *supposed* to let
an environment plug in exactly this kind of custom tab, but in practice
its availability/wiring varies across installed `openenv` versions: it
worked when run locally against the pinned `openenv==0.4.2` in this
project's `.venv`, but the same code deployed to Hugging Face Spaces
(via `openenv push`, same `uv.lock`) silently fell back to the generic
default tab with no error and no custom "Board" tab at all. Rather than
depend on that hook, `mount_tic_tac_toe_board` builds its own
`WebInterfaceManager` and mounts this board directly at `/web` with
`gradio.mount_gradio_app`, independent of `create_app`'s internal
web-interface wiring — see `server/app.py`.
"""

from typing import Any, Dict, List, Type

import gradio as gr
from fastapi import FastAPI
from openenv.core.env_server.types import Action, Observation
from openenv.core.env_server.web_interface import WebInterfaceManager

_CELL_SYMBOL = {0: "", 1: "X", -1: "O"}
_CELL_CLASS = {0: ["ttt-cell"], 1: ["ttt-cell", "ttt-cell-x"], -1: ["ttt-cell", "ttt-cell-o"]}

_BOARD_CSS = """
#ttt-board { max-width: 340px; margin: 8px auto 4px; }
#ttt-board .ttt-cell {
    height: 96px !important;
    min-width: 96px !important;
    font-size: 3rem !important;
    font-weight: 800 !important;
    border-radius: 16px !important;
    border: 2px solid #475569 !important;
    background: #1e293b !important;
    color: #e2e8f0 !important;
    transition: transform 0.12s ease, box-shadow 0.12s ease, background 0.12s ease;
}
#ttt-board .ttt-cell:hover:not(:disabled) {
    transform: scale(1.05);
    border-color: #7dd3fc !important;
    box-shadow: 0 0 16px rgba(125, 211, 252, 0.5);
}
#ttt-board .ttt-cell-x {
    background: #0c4a6e !important;
    border-color: #38bdf8 !important;
    color: #7dd3fc !important;
}
#ttt-board .ttt-cell-o {
    background: #7c2d12 !important;
    border-color: #fb923c !important;
    color: #fdba74 !important;
}
#ttt-board .ttt-cell:disabled {
    opacity: 1 !important;
    cursor: default !important;
}
#ttt-new-game { max-width: 340px; margin: 4px auto 8px; }
#ttt-new-game button {
    height: 56px !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    border-radius: 14px !important;
    border: none !important;
    background: linear-gradient(135deg, #22c55e, #16a34a) !important;
    color: white !important;
}
#ttt-new-game button:hover {
    filter: brightness(1.08);
}
"""


def _board_updates(board: List[int], done: bool) -> List[Dict[str, Any]]:
    """One gr.update() per cell: show its mark, color it, and disable once filled or done."""
    return [
        gr.update(value=_CELL_SYMBOL.get(v, ""), interactive=(v == 0 and not done), elem_classes=_CELL_CLASS.get(v, ["ttt-cell"]))
        for v in board
    ]


def _status_message(obs: Dict[str, Any], done: bool) -> str:
    message = obs.get("message", "")
    if not done:
        return message or "Your turn — click an empty cell."
    winner = obs.get("winner")
    if winner == 1:
        return "You win! Click New Game to play again."
    if winner == -1:
        return f"{message or 'Opponent wins.'} Click New Game to play again."
    if winner == 0:
        return "Draw. Click New Game to play again."
    return message


def build_tic_tac_toe_board(
    web_manager: Any,
    action_fields,
    metadata,
    is_chat_env: bool,
    title: str,
    quick_start_md: str,
) -> gr.Blocks:
    """Build the clickable-board tab mounted at `/web` (see the `gradio_builder`
    parameter of `openenv.core.env_server.http_server.create_app`)."""

    with gr.Blocks(title=title) as demo:
        gr.HTML(f"<style>{_BOARD_CSS}</style>")
        gr.Markdown(
            "## Tic Tac Toe\n"
            "You are **X** and move first; the opponent (**O**) replies "
            "automatically after every move. Click **New Game** to start."
        )
        status = gr.Textbox(label="Status", value="Click New Game to start.", interactive=False)

        buttons: List[gr.Button] = []
        with gr.Column(elem_id="ttt-board"):
            for _row in range(3):
                with gr.Row():
                    for _col in range(3):
                        buttons.append(gr.Button(value="", interactive=False, elem_classes=["ttt-cell"]))

        with gr.Column(elem_id="ttt-new-game"):
            new_game_btn = gr.Button("New Game", variant="primary")

        async def _reset():
            data = await web_manager.reset_environment()
            obs = data.get("observation", {})
            updates = _board_updates(obs.get("board", [0] * 9), done=False)
            return (*updates, _status_message(obs, done=False))

        def _make_play_fn(cell: int):
            async def _play():
                data = await web_manager.step_environment({"cell": cell})
                obs = data.get("observation", {})
                done = bool(data.get("done", False))
                updates = _board_updates(obs.get("board", [0] * 9), done=done)
                return (*updates, _status_message(obs, done=done))

            return _play

        new_game_btn.click(fn=_reset, outputs=[*buttons, status])
        for i, btn in enumerate(buttons):
            btn.click(fn=_make_play_fn(i), outputs=[*buttons, status])

    return demo


def mount_tic_tac_toe_board(
    app: FastAPI,
    env_cls: Any,
    action_cls: Type[Action],
    observation_cls: Type[Observation],
    path: str = "/web",
) -> None:
    """Mount the clickable board at `path` on `app`, independent of
    `create_app`'s own web-interface flag/hook (see module docstring for why)."""

    manager = WebInterfaceManager(env_cls, action_cls, observation_cls)
    blocks = build_tic_tac_toe_board(
        manager,
        action_fields=[],
        metadata=None,
        is_chat_env=False,
        title="Tic Tac Toe",
        quick_start_md="",
    )
    gr.mount_gradio_app(app, blocks, path=path)
