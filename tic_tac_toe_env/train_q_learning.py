# SPDX-License-Identifier: BSD-3-Clause

"""
Train a tabular Q-learning agent to play Tic Tac Toe against this
environment's built-in opponent.

This talks to `TicTacToeEnvironment` directly (no HTTP/Docker needed), which
is the fastest way to run thousands of training episodes. Once you have a
policy you like, swap in `TicTacToeEnv` (the HTTP client) to evaluate it
against a server running in Docker or on Hugging Face Spaces — same
`reset()`/`step()` calls, same `TicTacToeAction`.

Usage:
    python3 train_q_learning.py
    python3 train_q_learning.py --episodes 50000 --opponent heuristic
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

from models import TicTacToeAction
from server.tic_tac_toe_env_environment import TicTacToeEnvironment


def _default_q_values() -> list[float]:
    return [0.0] * 9


def epsilon_greedy(q_table, state, valid_actions, epsilon):
    if random.random() < epsilon:
        return random.choice(valid_actions)
    q_values = q_table[state]
    return max(valid_actions, key=lambda a: q_values[a])


def save_q_table(q_table: dict, path: str) -> None:
    """Persist a Q-table to JSON (board tuples become comma-joined string keys)."""
    serializable = {",".join(map(str, state)): values for state, values in q_table.items()}
    with open(path, "w") as f:
        json.dump(serializable, f)


def load_q_table(path: str) -> dict:
    """Load a Q-table previously written by `save_q_table`."""
    with open(path) as f:
        raw = json.load(f)
    q_table: dict[tuple, list[float]] = defaultdict(_default_q_values)
    for key, values in raw.items():
        state = tuple(int(x) for x in key.split(","))
        q_table[state] = values
    return q_table


def train(episodes: int, opponent: str, alpha: float, gamma: float, epsilon_start: float, epsilon_end: float) -> dict:
    env = TicTacToeEnvironment(opponent=opponent)
    q_table: dict[tuple, list[float]] = defaultdict(_default_q_values)

    for episode in range(episodes):
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * (episode / max(episodes - 1, 1))

        obs = env.reset()
        state = tuple(obs.board)

        while not obs.done:
            action_cell = epsilon_greedy(q_table, state, obs.valid_actions, epsilon)
            obs = env.step(TicTacToeAction(cell=action_cell))
            next_state = tuple(obs.board)

            best_next = 0.0 if obs.done else max(q_table[next_state][a] for a in obs.valid_actions)
            td_target = obs.reward + gamma * best_next
            q_table[state][action_cell] += alpha * (td_target - q_table[state][action_cell])

            state = next_state

    return q_table


def evaluate(q_table: dict, opponent: str, games: int) -> dict:
    env = TicTacToeEnvironment(opponent=opponent)
    wins = draws = losses = illegal = 0

    for _ in range(games):
        obs = env.reset()
        state = tuple(obs.board)
        while not obs.done:
            action_cell = epsilon_greedy(q_table, state, obs.valid_actions, epsilon=0.0)
            obs = env.step(TicTacToeAction(cell=action_cell))
            state = tuple(obs.board)

        if obs.metadata.get("illegal_move"):
            illegal += 1
        elif obs.winner == 1:
            wins += 1
        elif obs.winner == -1:
            losses += 1
        else:
            draws += 1

    return {"wins": wins, "draws": draws, "losses": losses, "illegal": illegal, "games": games}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--opponent", choices=["heuristic", "random", "hard", "minimax"], default="heuristic")
    parser.add_argument("--alpha", type=float, default=0.5, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--eval-games", type=int, default=1000)
    parser.add_argument("--save-path", type=str, default=None, help="write the trained Q-table to this JSON file")
    parser.add_argument(
        "--load-path",
        type=str,
        default=None,
        help="load a previously saved Q-table from this JSON file instead of training",
    )
    args = parser.parse_args()

    if args.load_path:
        print(f"Loading Q-table from {args.load_path}...")
        q_table = load_q_table(args.load_path)
        print(f"Loaded {len(q_table)} distinct board states.")
    else:
        print(f"Training tabular Q-learning agent for {args.episodes} episodes vs '{args.opponent}' opponent...")
        q_table = train(
            episodes=args.episodes,
            opponent=args.opponent,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
        )
        print(f"Learned {len(q_table)} distinct board states.")

    if args.save_path:
        save_q_table(q_table, args.save_path)
        print(f"Saved Q-table to {args.save_path}")

    for opp in ("heuristic", "random", "hard", "minimax"):
        results = evaluate(q_table, opponent=opp, games=args.eval_games)
        win_rate = results["wins"] / results["games"]
        draw_rate = results["draws"] / results["games"]
        loss_rate = results["losses"] / results["games"]
        print(
            f"vs {opp:>9}: win={win_rate:.1%} draw={draw_rate:.1%} loss={loss_rate:.1%} "
            f"illegal={results['illegal']}/{results['games']}"
        )


if __name__ == "__main__":
    main()
