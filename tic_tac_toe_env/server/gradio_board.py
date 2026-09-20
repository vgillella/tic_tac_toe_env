# SPDX-License-Identifier: BSD-3-Clause

"""
Custom `/web` board UI for the Tic Tac Toe environment.

OpenEnv's `create_app` auto-generates a generic form for any environment
(a raw "cell" number field plus a JSON dump of the response), which works
but forces you to read the board out of JSON on every move. This module
builds an actual clickable 3x3 grid instead, plus a scoreboard/streak
tracker and win/draw/loss particle-burst effects for a bit of game feel.

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

from typing import Any, Dict, List, Optional, Tuple, Type

import gradio as gr
from fastapi import FastAPI
from openenv.core.env_server.types import Action, Observation
from openenv.core.env_server.web_interface import WebInterfaceManager

_CELL_SYMBOL = {0: "", 1: "X", -1: "O"}
_CELL_CLASS = {0: ["ttt-cell"], 1: ["ttt-cell", "ttt-cell-x"], -1: ["ttt-cell", "ttt-cell-o"]}

_WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
    (0, 4, 8), (2, 4, 6),             # diagonals
]

_DEFAULT_SCORE = {"wins": 0, "draws": 0, "losses": 0, "streak": 0, "best_streak": 0, "last_result": None}

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
    box-shadow: 0 0 16px rgba(125, 211, 248, 0.5);
}
#ttt-board .ttt-cell-x {
    background: #0c4a6e !important;
    border-color: #38bdf8 !important;
    color: #7dd3fc !important;
    animation: ttt-pop 0.28s cubic-bezier(.34,1.56,.64,1);
}
#ttt-board .ttt-cell-o {
    background: #7c2d12 !important;
    border-color: #fb923c !important;
    color: #fdba74 !important;
    animation: ttt-pop 0.28s cubic-bezier(.34,1.56,.64,1);
}
#ttt-board .ttt-cell:disabled { opacity: 1 !important; cursor: default !important; }
#ttt-board .ttt-cell-winline {
    border-color: #facc15 !important;
    box-shadow: 0 0 22px rgba(250, 204, 21, 0.75) !important;
    animation: ttt-pop 0.28s cubic-bezier(.34,1.56,.64,1), ttt-glow 0.9s ease-in-out infinite;
}
@keyframes ttt-pop {
    0% { transform: scale(0.4); }
    70% { transform: scale(1.12); }
    100% { transform: scale(1); }
}
@keyframes ttt-glow {
    0%, 100% { box-shadow: 0 0 12px rgba(250, 204, 21, 0.55); }
    50% { box-shadow: 0 0 26px rgba(250, 204, 21, 0.95); }
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
#ttt-new-game button:hover { filter: brightness(1.08); }

#ttt-scoreboard { max-width: 340px; margin: 0 auto 10px; text-align: center; }
.ttt-score-row { display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; }
.ttt-score-pill {
    padding: 6px 14px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.95rem;
    border: 1.5px solid transparent;
}
.ttt-score-pill.win { background: rgba(34,197,94,0.15); border-color: #22c55e; color: #4ade80; }
.ttt-score-pill.draw { background: rgba(56,189,248,0.15); border-color: #38bdf8; color: #7dd3fc; }
.ttt-score-pill.loss { background: rgba(239,68,68,0.15); border-color: #ef4444; color: #f87171; }
.ttt-badge {
    margin-top: 8px;
    padding: 8px 14px;
    border-radius: 12px;
    background: linear-gradient(135deg, rgba(250,204,21,0.18), rgba(251,146,60,0.18));
    border: 1.5px solid #facc15;
    color: #fde68a;
    font-weight: 700;
    display: inline-block;
    animation: ttt-badge-in 0.4s cubic-bezier(.34,1.56,.64,1);
}
@keyframes ttt-badge-in {
    0% { transform: scale(0.7) translateY(6px); opacity: 0; }
    100% { transform: scale(1) translateY(0); opacity: 1; }
}
@keyframes ttt-shake {
    0%, 100% { transform: translateX(0); }
    20% { transform: translateX(-8px); }
    40% { transform: translateX(8px); }
    60% { transform: translateX(-5px); }
    80% { transform: translateX(5px); }
}
#ttt-board.ttt-shake { animation: ttt-shake 0.4s ease; }
"""

# Client-side particle-burst engine, injected once. `window.tttBlast(kind)`
# spawns a full-screen canvas burst: a colorful multi-blob "graffiti
# splatter" for a win, a calmer blue/grey confetti burst for a draw, and a
# duller red splatter (plus a board shake) for a loss.
_FX_JS = """
() => {
    if (window.__tttFxInit) return;
    window.__tttFxInit = true;

    const canvas = document.createElement('canvas');
    canvas.id = 'ttt-fx-canvas';
    canvas.style.cssText = 'position:fixed;inset:0;width:100vw;height:100vh;pointer-events:none;z-index:99999;';
    document.body.appendChild(canvas);
    const ctx = canvas.getContext('2d');

    function resize() { canvas.width = window.innerWidth; canvas.height = window.innerHeight; }
    window.addEventListener('resize', resize);
    resize();

    let particles = [];
    let running = false;
    const rand = (a, b) => a + Math.random() * (b - a);

    const PALETTES = {
        win: ['#ef4444', '#f59e0b', '#eab308', '#22c55e', '#38bdf8', '#a855f7', '#ec4899'],
        draw: ['#94a3b8', '#60a5fa', '#38bdf8', '#cbd5e1', '#818cf8'],
        loss: ['#7f1d1d', '#b91c1c', '#ef4444', '#57534e'],
    };
    const COUNT = { win: 150, draw: 70, loss: 50 };
    const SPEED = { win: [5, 15], draw: [2, 7], loss: [2, 8] };
    const SIZE = { win: [5, 18], draw: [3, 10], loss: [4, 12] };
    const BLOBS = { win: [2, 5], draw: [1, 1], loss: [1, 3] };

    function spawn(kind) {
        const palette = PALETTES[kind] || PALETTES.draw;
        const count = COUNT[kind] || 60;
        const [minSpeed, maxSpeed] = SPEED[kind] || [2, 8];
        const [minSize, maxSize] = SIZE[kind] || [4, 10];
        const [minBlobs, maxBlobs] = BLOBS[kind] || [1, 1];
        const cx = canvas.width / 2;
        const cy = canvas.height * (kind === 'win' ? 0.35 : 0.4);
        for (let i = 0; i < count; i++) {
            const angle = rand(0, Math.PI * 2);
            const speed = rand(minSpeed, maxSpeed);
            particles.push({
                x: cx, y: cy,
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed - rand(0, 4),
                size: rand(minSize, maxSize),
                color: palette[Math.floor(Math.random() * palette.length)],
                rot: rand(0, Math.PI * 2),
                vr: rand(-0.25, 0.25),
                life: 1,
                decay: rand(0.008, 0.02),
                gravity: 0.28,
                blobs: Math.floor(rand(minBlobs, maxBlobs + 1)),
            });
        }
        if (!running) { running = true; requestAnimationFrame(tick); }
    }

    function tick() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        let alive = false;
        for (let i = particles.length - 1; i >= 0; i--) {
            const p = particles[i];
            p.x += p.vx; p.y += p.vy; p.vy += p.gravity; p.rot += p.vr; p.life -= p.decay;
            if (p.life <= 0) { particles.splice(i, 1); continue; }
            alive = true;
            ctx.save();
            ctx.globalAlpha = Math.max(p.life, 0);
            ctx.translate(p.x, p.y);
            ctx.rotate(p.rot);
            ctx.fillStyle = p.color;
            for (let b = 0; b < p.blobs; b++) {
                const ox = (b - (p.blobs - 1) / 2) * p.size * 0.55;
                ctx.beginPath();
                ctx.arc(ox, 0, p.size * (0.5 + 0.5 * Math.random()), 0, Math.PI * 2);
                ctx.fill();
            }
            ctx.restore();
        }
        if (alive) { requestAnimationFrame(tick); } else { running = false; }
    }

    window.tttBlast = spawn;
}
"""

_TRIGGER_JS = """
(score) => {
    if (!score) return;
    const result = score.last_result;
    if (!window.tttBlast || !result) return;
    window.tttBlast(result);
    if (result === 'loss') {
        const board = document.getElementById('ttt-board');
        if (board) {
            board.classList.remove('ttt-shake');
            void board.offsetWidth;
            board.classList.add('ttt-shake');
        }
    }
}
"""


def _winning_line(board: List[int]) -> Optional[Tuple[int, int, int]]:
    for a, b, c in _WIN_LINES:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return (a, b, c)
    return None


def _board_updates(board: List[int], done: bool, win_line: Optional[Tuple[int, int, int]] = None) -> List[Dict[str, Any]]:
    """One gr.update() per cell: show its mark, color it, highlight a winning
    line, and disable once filled or done."""
    win_line_set = set(win_line) if win_line else set()
    updates = []
    for i, v in enumerate(board):
        classes = list(_CELL_CLASS.get(v, ["ttt-cell"]))
        if i in win_line_set:
            classes.append("ttt-cell-winline")
        updates.append(gr.update(value=_CELL_SYMBOL.get(v, ""), interactive=(v == 0 and not done), elem_classes=classes))
    return updates


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


def _score_html(score: Dict[str, Any]) -> str:
    wins, draws, losses = score.get("wins", 0), score.get("draws", 0), score.get("losses", 0)
    streak, best_streak = score.get("streak", 0), score.get("best_streak", 0)

    badge = ""
    if score.get("last_result") == "win":
        badge = "<div class='ttt-badge'>🎉 IMPOSSIBLE — you beat an unbeatable opponent!</div>"
    elif streak >= 10:
        badge = f"<div class='ttt-badge'>🏆 {streak}-draw streak — flawless play!</div>"
    elif streak >= 5:
        badge = f"<div class='ttt-badge'>🔥 {streak} draws in a row — you're on fire!</div>"
    elif streak >= 3:
        badge = f"<div class='ttt-badge'>✨ {streak}-draw streak — nicely played!</div>"
    elif best_streak >= 3 and streak == 0 and losses > 0:
        badge = "<div class='ttt-badge'>💡 Tip: minimax never loses — try to force a draw every time.</div>"

    return (
        "<div id='ttt-scoreboard'><div class='ttt-score-row'>"
        f"<span class='ttt-score-pill win'>🏆 Wins {wins}</span>"
        f"<span class='ttt-score-pill draw'>🤝 Draws {draws}</span>"
        f"<span class='ttt-score-pill loss'>💥 Losses {losses}</span>"
        "</div>" + badge + "</div>"
    )


def _apply_result(score: Dict[str, Any], winner: Optional[int]) -> Dict[str, Any]:
    score = dict(score)
    if winner == 1:
        score["wins"] += 1
        score["streak"] = 0
        score["last_result"] = "win"
    elif winner == -1:
        score["losses"] += 1
        score["streak"] = 0
        score["last_result"] = "loss"
    elif winner == 0:
        score["draws"] += 1
        score["streak"] += 1
        score["best_streak"] = max(score["best_streak"], score["streak"])
        score["last_result"] = "draw"
    else:
        score["last_result"] = None
    return score


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
            "automatically after every move — and plays perfectly, so the "
            "best you can do is draw. Click **New Game** to start."
        )

        score_state = gr.State(value=dict(_DEFAULT_SCORE))
        scoreboard = gr.HTML(_score_html(_DEFAULT_SCORE))
        status = gr.Textbox(label="Status", value="Click New Game to start.", interactive=False)

        buttons: List[gr.Button] = []
        with gr.Column(elem_id="ttt-board"):
            for _row in range(3):
                with gr.Row():
                    for _col in range(3):
                        buttons.append(gr.Button(value="", interactive=False, elem_classes=["ttt-cell"]))

        with gr.Column(elem_id="ttt-new-game"):
            new_game_btn = gr.Button("New Game", variant="primary")

        async def _reset(score: Dict[str, Any]):
            data = await web_manager.reset_environment()
            obs = data.get("observation", {})
            updates = _board_updates(obs.get("board", [0] * 9), done=False)
            score = dict(score)
            score["last_result"] = None
            return (*updates, _status_message(obs, done=False), _score_html(score), score)

        def _make_play_fn(cell: int):
            async def _play(score: Dict[str, Any]):
                data = await web_manager.step_environment({"cell": cell})
                obs = data.get("observation", {})
                done = bool(data.get("done", False))
                board = obs.get("board", [0] * 9)
                win_line = _winning_line(board) if done else None
                updates = _board_updates(board, done=done, win_line=win_line)
                new_score = _apply_result(score, obs.get("winner")) if done else dict(score, last_result=None)
                return (*updates, _status_message(obs, done=done), _score_html(new_score), new_score)

            return _play

        demo.load(fn=None, js=_FX_JS)

        outputs = [*buttons, status, scoreboard, score_state]
        new_game_btn.click(fn=_reset, inputs=[score_state], outputs=outputs)
        for i, btn in enumerate(buttons):
            btn.click(fn=_make_play_fn(i), inputs=[score_state], outputs=outputs).then(
                fn=None, inputs=[score_state], js=_TRIGGER_JS
            )

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
