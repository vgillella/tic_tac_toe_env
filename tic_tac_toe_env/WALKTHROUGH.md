# Tic Tac Toe Env — Walkthrough

> Note: this project has no `sprints/vN/`, `PRD.md`, `TASKS.md`, or git
> history to diff against, so this walkthrough documents the **current
> state of the code** rather than a specific sprint's diff.

## Summary
Built an [OpenEnv](https://github.com/huggingface/OpenEnv)-compatible
reinforcement-learning environment for Tic Tac Toe: a FastAPI server that
exposes `reset()`/`step()` over HTTP and WebSocket, a Python client with
matching semantics, Pydantic action/observation models, and a standalone
tabular Q-learning training script that beats the built-in opponent without
needing the server at all. The package is also wired up for Docker and
Hugging Face Spaces deployment via `openenv.yaml`.

## Architecture Overview

```
┌───────────────────────────────────────────────────────────────────┐
│                         Training / Client Code                    │
│                                                                     │
│   train_q_learning.py                 client.py                    │
│   (imports TicTacToeEnvironment        (TicTacToeEnv, an           │
│    directly — no HTTP, fastest         EnvClient subclass —        │
│    for thousands of episodes)          talks to the server over    │
│         │                              HTTP/WebSocket)             │
│         │                                     │                    │
│         ▼                                     ▼                    │
│  ┌─────────────────────────┐        ┌───────────────────────────┐ │
│  │ TicTacToeEnvironment     │        │ FastAPI app (server/app.py)│ │
│  │ (server/tic_tac_toe_env_ │◀───────│  create_app(...)            │ │
│  │  environment.py)         │  used  │  - POST /reset              │ │
│  │  - board state (list[9]) │  by    │  - POST /step                │ │
│  │  - win detection          │        │  - GET  /state               │ │
│  │  - opponent policy         │        │  - GET  /schema               │ │
│  │    (heuristic | random)    │        │  - WS   /ws (per-session)      │ │
│  └─────────────────────────┘        └───────────────────────────┘ │
│                                                    │                │
│                                                    ▼                │
│                                        ┌───────────────────────┐   │
│                                        │ models.py               │   │
│                                        │  TicTacToeAction         │   │
│                                        │  TicTacToeObservation    │   │
│                                        └───────────────────────┘   │
└───────────────────────────────────────────────────────────────────┘

Deployment path:
  server/Dockerfile ──▶ tic_tac_toe_env-env:latest image ──▶ openenv push ──▶ HF Spaces
```

## Files Created/Modified

### `models.py`
**Purpose**: Defines the RL action/observation contract shared by the
server and the client.
**Key Components**:
- `TicTacToeAction` — one field, `cell: int` (0-8), validated with
  Pydantic's `ge=0, le=8`.
- `TicTacToeObservation` — `board`, `valid_actions`, `winner`,
  `current_player`, `message`, plus the base `Observation` class's
  `done`/`reward`/`metadata`.

**How it works**:
Both classes subclass OpenEnv's `Action`/`Observation` base types, which is
what makes them interoperable with `EnvClient`/`Environment`/`create_app`
without extra glue code. `winner` is `Optional[int]` specifically so the
client can distinguish "still playing" (`None`) from "drew" (`0`) — a
common bug source if collapsed into a single falsy value.

### `client.py`
**Purpose**: HTTP/WebSocket client for talking to a running Tic Tac Toe
server, matching Gym-style `reset()`/`step()` semantics.
**Key Components**:
- `TicTacToeEnv(EnvClient[...])` — implements the three hooks `EnvClient`
  needs: `_step_payload`, `_parse_result`, `_parse_state`.

**How it works**:
`_step_payload` just serializes `{"cell": action.cell}` for the wire.
`_parse_result` reconstructs a `TicTacToeObservation` from the raw JSON
response, being careful to read `reward`/`metadata` from the top level of
the payload (not just the nested `observation` dict) since the server
returns both. `_parse_state` extracts `episode_id`/`step_count` for
resuming or inspecting a session. The class supports both a context-manager
style (`with TicTacToeEnv(base_url=...) as env`) and a Docker-backed mode
(`TicTacToeEnv.from_docker_image(...).sync()`).

### `server/tic_tac_toe_env_environment.py`
**Purpose**: The actual game engine — board state, win detection, and the
scripted opponent. This is the only file with real game logic; everything
else is plumbing around it.
**Key Components**:
- `_winner(board)` — checks all 8 win lines by summing three cells (works
  because X=+1/O=-1, so three-in-a-row sums to ±3).
- `_empty_cells(board)` — list of legal move indices.
- `TicTacToeEnvironment(Environment)` — holds one board's worth of
  per-instance state; `SUPPORTS_CONCURRENT_SESSIONS = True` so the WebSocket
  server can give each connected client its own isolated game.
- `reset()` / `step()` — Gym-style lifecycle methods.
- `_opponent_move(board)` — dispatches to one of three policies based on
  `opponent_kind`: `"random"`, `"heuristic"` (win-if-possible, else block,
  else random), or `"minimax"`.
- `_minimax_score(board, player)` / `_minimax_move(board)` — exhaustively
  solves the game tree (memoized with `functools.lru_cache`, keyed on the
  board as a tuple) to find optimal play for "O". Small enough state space
  (3x3) that no alpha-beta pruning or depth limit is needed.

**How it works**:
`step()` does two moves per call: the agent's, then (if the game isn't
over) the opponent's immediate reply — so a training loop never has to
special-case "whose turn is it." An out-of-range or already-occupied `cell`
ends the episode immediately with `reward=-1.0` and
`metadata={"illegal_move": True}`, rather than silently ignoring the
action — this is what lets `train_q_learning.py`'s evaluation report an
`illegal` count and teaches the Q-learning agent (via the large negative
reward) to only ever pick from `valid_actions`.

`_opponent_move` implements the classic "win/block/random" heuristic by
brute-force trial: for each empty cell, simulate placing the opponent's
mark and check `_winner`; if that wins, take it. If no winning move exists,
repeat the same check for the *agent's* mark to find and block it. This is
O(empty_cells) per turn — trivial at 3x3 — rather than a real minimax, so
the opponent is beatable/drawable but not unbeatable, which is the explicit
design intent stated in the file's docstring.

### `server/app.py`
**Purpose**: Wires `TicTacToeEnvironment` into an OpenEnv-standard FastAPI
app.
**Key Components**:
- `app = create_app(TicTacToeEnvironment, TicTacToeAction,
  TicTacToeObservation, env_name=..., max_concurrent_envs=4)` — this one
  call from `openenv.core.env_server.http_server` generates all the HTTP
  routes, the WebSocket route, the web UI, and `/docs`.
- `main(host, port)` — entry point for `uv run --project . server`.

**How it works**:
There's very little custom code here by design — nearly all server
behavior comes from OpenEnv's `create_app` factory, which is why the
environment-specific logic all lives in
`tic_tac_toe_env_environment.py` instead. The try/except import block
(`from ..models import ...` then falling back to `from models import
...`) exists to support both "imported as a package"
(`tic_tac_toe_env.server.app`) and "run as a standalone script from inside
the directory" usage — the same reason `tic_tac_toe_env_environment.py`
has an identical fallback import for `models`.

### `train_q_learning.py`
**Purpose**: A working example agent — tabular Q-learning trained directly
against `TicTacToeEnvironment`, bypassing HTTP entirely for speed.
**Key Functions**:
- `epsilon_greedy(q_table, state, valid_actions, epsilon)` — explore
  randomly with probability `epsilon`, else exploit the best known
  Q-value among currently legal moves.
- `train(episodes, opponent, alpha, gamma, epsilon_start, epsilon_end)` —
  runs the standard Q-learning update
  `Q[s,a] += alpha * (reward + gamma * max(Q[s']) - Q[s,a])` per step,
  linearly decaying epsilon from `epsilon_start` to `epsilon_end` over the
  run.
- `evaluate(q_table, opponent, games)` — plays `games` greedy (epsilon=0)
  games and tallies win/draw/loss/illegal counts.
- `save_q_table(q_table, path)` / `load_q_table(path)` — persist/restore a
  trained policy as JSON. Board-tuple keys aren't valid JSON object keys,
  so they're serialized as comma-joined strings (e.g. `"1,0,-1,..."`) and
  parsed back into `tuple[int, ...]` on load; `--load-path` skips training
  entirely and jumps straight to evaluation.

**How it works**:
The state key is `tuple(obs.board)` — a hashable 9-tuple of `{-1,0,1}` —
used directly as a dict key into a `defaultdict(lambda: [0.0]*9)` Q-table,
so no feature engineering or neural net is needed for a state space this
small (at most 3^9 = 19,683 reachable-ish states, and only ~1,000+ are
actually visited under legal play, per the README's example run:
"Learned 1188 distinct board states"). Because `step()` already advances
through the opponent's reply, each Q-update only ever sees the *agent's*
transitions — the opponent is treated as part of the environment dynamics,
which is exactly the setup single-agent Q-learning expects.

`main()` runs training, then evaluates the learned policy against both
`heuristic` and `random` opponents and prints win/draw/loss/illegal rates —
useful as a quick regression check that a change to the environment or
reward shaping hasn't broken learning.

### `openenv.yaml`
**Purpose**: OpenEnv manifest describing how to deploy this environment as
a "Space" (`spec_version`, `runtime: fastapi`, `app: server.app:app`,
`port: 8000`).

### `server/Dockerfile`
**Purpose**: Multi-stage build producing a runnable container image.
**How it works**:
Stage 1 (`builder`) installs `uv` if missing and runs `uv sync` against
`uv.lock` (or resolves fresh if no lockfile) to build a `.venv`. Stage 2
copies just the `.venv` and the app code into a clean image based on the
same `openenv-base` image, avoiding build tooling (git, uv installer) in
the final image. `PYTHONPATH=/app/env` lets the `try/except` relative
imports in `app.py`/`tic_tac_toe_env_environment.py` resolve. A
`HEALTHCHECK` hits `/health` every 30s, which is what Hugging Face Spaces
and `docker run` rely on to report container health.

### `README.md`, `pyproject.toml`, `server/requirements.txt`, `__init__.py`, `server/__init__.py`
Supporting docs/config: `README.md` documents the reward table, opponent
policies, and both local/Docker/Spaces workflows (see Quick Start above);
`pyproject.toml` declares the `openenv` dependency and packages the repo as
`openenv-tic_tac_toe_env`; `requirements.txt` pins the same for the Docker
build path; the `__init__.py` files re-export `TicTacToeEnv`,
`TicTacToeAction`, `TicTacToeObservation`, and `TicTacToeEnvironment` as the
package's public API.

## Data Flow

**Training path (no server):**
1. `train_q_learning.py` constructs `TicTacToeEnvironment(opponent=...)` directly.
2. `env.reset()` → empty board, agent (X) to move.
3. Loop: `epsilon_greedy` picks a cell → `env.step(TicTacToeAction(cell=...))`
   → environment places X, checks for a win/draw, then (if still playing)
   places O via `_opponent_move`, checks again → returns
   `TicTacToeObservation` with `reward`/`done`.
4. Q-table updated in place; repeat until `done`.
5. After training, `evaluate()` runs greedy games and prints win/draw/loss
   rates vs. both opponent types.

**Server path (HTTP/WebSocket):**
1. Client opens `TicTacToeEnv(base_url=...)`, which establishes a
   WebSocket session against `/ws` — this gives it its own
   `TicTacToeEnvironment` instance (`SUPPORTS_CONCURRENT_SESSIONS`), isolated
   from other concurrent clients.
2. `env.reset()` → server calls `TicTacToeEnvironment.reset()` → JSON
   observation → `client._parse_result` → `TicTacToeObservation`.
3. `env.step(TicTacToeAction(cell=4))` → `_step_payload` serializes the
   action → server calls `TicTacToeEnvironment.step()` (agent move +
   opponent reply) → response parsed back into an observation with
   `reward`/`done`/`metadata`.
4. Plain REST `/reset` and `/step` exist too, but operate on one shared
   environment instance — fine for `curl` smoke tests, not for concurrent
   games (the README calls this out explicitly).

## Test Coverage
- Unit: 18 tests in `tests/test_environment.py` — win-line detection for
  all 8 lines, `reset()` initial state, illegal-move penalty handling,
  no-op behavior after an episode ends, the heuristic opponent's
  block/win priority, the minimax opponent's block/win correctness, full
  randomized games against all three opponents terminating with a valid
  winner/reward, and a 30-game check that the minimax opponent never
  loses. Run with `pytest tests/` from inside `tic_tac_toe_env/`.
- Integration: 0 (the HTTP/WebSocket server layer in `server/app.py` is
  still untested — would need a `TestClient`/`httpx` fixture).
- E2E: 0.

## Security Measures
None implemented — this is a local/self-hosted RL training environment,
not a service handling untrusted input or secrets. No auth on the HTTP/WS
endpoints; `max_concurrent_envs=4` is a resource cap, not a security
control.

## Known Limitations
- No integration/E2E tests for the HTTP/WebSocket server layer
  (`server/app.py`) — only the environment engine itself is unit-tested.
- Plain REST `/reset`/`/step` share one global environment instance, which
  is a foot-gun for concurrent callers who don't realize they need the
  WebSocket client instead (documented, but not enforced in code).
- The minimax opponent recomputes its search on every move with no
  cross-process cache (only in-process `lru_cache`), so a fresh server
  process pays the full solve cost again on first use. Negligible in
  practice for 3x3 Tic Tac Toe, but wouldn't scale to a larger board.
- No self-play training loop — the agent only ever trains against the
  three fixed scripted opponents.

## What's Next
Completed this round:
1. ✅ Added a `pytest` suite (`tests/test_environment.py`, 18 tests)
   covering `_winner`, illegal-move handling, opponent block/win
   correctness, and full scripted games against each opponent type.
2. ✅ `train_q_learning.py` now supports `--save-path`/`--load-path` to
   persist/reload a trained Q-table as JSON instead of retraining every run.
3. ✅ Added a `"minimax"` opponent (`_minimax_move`/`_minimax_score` in
   `server/tic_tac_toe_env_environment.py`) — exhaustively solved, never
   loses, used as an evaluation upper bound.
4. ✅ `git init`, initial commit, and pushed to
   [github.com/vgillella/tic_tac_toe_env](https://github.com/vgillella/tic_tac_toe_env).

Suggested next iteration:
1. Add integration tests for `server/app.py` using FastAPI's `TestClient`
   (REST `/reset`/`/step`) and a WebSocket client fixture (`/ws`).
2. Add a self-play training mode so the agent isn't capped by the fixed
   opponents' skill ceiling.
3. Add `PRD.md`/`TASKS.md` under a `sprints/v1/` directory if continuing
   with the sprint-based `/prd` + `/dev` + `/walkthrough` workflow going
   forward.
