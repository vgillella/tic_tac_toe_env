---
title: Tic Tac Toe Environment Server
emoji: ❌
colorFrom: indigo
colorTo: pink
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Tic Tac Toe Environment

An [OpenEnv](https://github.com/huggingface/OpenEnv) environment for training an
agent to play Tic Tac Toe. The agent always plays **X** and moves first; the
server plays **O** automatically after every agent move, so one `step()` call
advances a full round (your move + the opponent's reply) unless your move
already ends the game.

Board indices:

```
0 | 1 | 2
3 | 4 | 5
6 | 7 | 8
```

## Quick Start

```python
from tic_tac_toe_env import TicTacToeAction, TicTacToeEnv

with TicTacToeEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(result.observation.board)          # [0, 0, 0, 0, 0, 0, 0, 0, 0]

    result = env.step(TicTacToeAction(cell=4))  # play center
    print(result.observation.board)           # e.g. [0, 0, 0, 0, 1, 0, 0, 0, -1]
    print(result.reward, result.done)
```

Or with Docker:

```python
from tic_tac_toe_env import TicTacToeAction, TicTacToeEnv

client = TicTacToeEnv.from_docker_image("tic_tac_toe_env-env:latest").sync()
try:
    result = client.reset()
    result = client.step(TicTacToeAction(cell=4))
finally:
    client.close()
```

## Training an Agent

For fast training you don't need HTTP/Docker at all — import the environment
class directly and run thousands of episodes in-process:

```bash
python3 train_q_learning.py --episodes 20000 --opponent heuristic
```

This trains a tabular Q-learning agent and evaluates it against both the
`heuristic` and `random` built-in opponents:

```
Training tabular Q-learning agent for 20000 episodes vs 'heuristic' opponent...
Learned 1188 distinct board states.
vs heuristic: win=92.4% draw=7.6% loss=0.0% illegal=0/1000
vs    random: win=85.4% draw=6.5% loss=8.1% illegal=0/1000
```

Once you have a policy, swap in the HTTP `TicTacToeEnv` client to evaluate it
against a server running in Docker or deployed to Hugging Face Spaces — same
`reset()`/`step()` calls, same `TicTacToeAction`.

## Environment Details

### Action

**TicTacToeAction**
- `cell` (int, 0-8) — board position to place the agent's mark ("X")

### Observation

**TicTacToeObservation**
- `board` (list[int], length 9) — flattened board; `1` = agent ("X"), `-1` = opponent ("O"), `0` = empty
- `valid_actions` (list[int]) — empty cell indices currently available
- `winner` (int | None) — `1` agent win, `-1` opponent win, `0` draw, `None` game in progress
- `current_player` (int) — always `1` (the agent plays "X")
- `message` (str) — human-readable status
- `done` (bool) — episode finished
- `reward` (float) — see below
- `metadata` (dict) — e.g. `step`, `opponent_cell`, or `illegal_move: True`

### Reward

| Outcome                          | Reward |
|-----------------------------------|--------|
| Agent wins                        | `+1.0` |
| Agent loses                       | `-1.0` |
| Draw                               | `0.0`  |
| Illegal move (occupied/out-of-range cell) | `-1.0` (episode ends immediately) |
| Game still in progress             | `0.0`  |

### Opponent

The built-in opponent (`server/tic_tac_toe_env_environment.py`) supports two
policies, chosen when constructing `TicTacToeEnvironment(opponent=...)`:

- `"heuristic"` (default) — takes a winning move if available, otherwise
  blocks the agent's winning move, otherwise plays randomly. A reasonable
  training adversary that a good agent can consistently beat or draw.
- `"random"` — always plays a uniformly random empty cell. Easier baseline.

Swap in a minimax or self-play opponent here if you want a harder target.

## Building the Docker Image

```bash
# From this directory
docker build -t tic_tac_toe_env-env:latest -f server/Dockerfile .
docker run -p 8000:8000 tic_tac_toe_env-env:latest
```

## Deploying to Hugging Face Spaces

```bash
# From this directory (where openenv.yaml lives)
openenv push

# Or with options
openenv push --repo-id my-org/tic-tac-toe-env --private
```

After deployment your Space will be available at
`https://huggingface.co/spaces/<repo-id>`, exposing:
- **Web Interface** at `/web`
- **API Docs** at `/docs`
- **Health Check** at `/health`
- **WebSocket** at `/ws` (used by `TicTacToeEnv` for stateful play)

## Development & Testing

Run the environment logic directly (no server needed):

```bash
python3 -c "
from server.tic_tac_toe_env_environment import TicTacToeEnvironment
from models import TicTacToeAction

env = TicTacToeEnvironment()
obs = env.reset()
print(obs.board)
obs = env.step(TicTacToeAction(cell=4))
print(obs.board, obs.reward, obs.done)
"
```

Run the server locally:

```bash
uv sync
uvicorn server.app:app --reload
```

> **Note on the plain REST `/reset` and `/step` endpoints:** they operate on
> a single shared environment instance and are best for quick manual checks
> (e.g. `curl`). For real multi-step games with isolated per-client state,
> use the `TicTacToeEnv` Python client (or connect to `/ws` directly), which
> opens a persistent WebSocket session per client — this is what
> `train_q_learning.py` and the Quick Start example above rely on.

## Project Structure

```
tic_tac_toe_env/
├── README.md                  # This file
├── openenv.yaml                # OpenEnv manifest
├── pyproject.toml              # Project metadata and dependencies
├── uv.lock                     # Locked dependencies (generated)
├── client.py                   # TicTacToeEnv WebSocket client
├── models.py                   # TicTacToeAction / TicTacToeObservation
├── train_q_learning.py         # Example tabular Q-learning training script
└── server/
    ├── tic_tac_toe_env_environment.py  # Game logic + opponent policy
    ├── app.py                          # FastAPI application (HTTP + WebSocket)
    └── Dockerfile                      # Container image definition
```
