# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Tic Tac Toe environment engine (server/tic_tac_toe_env_environment.py)."""

import random

import pytest

from models import TicTacToeAction
from server.tic_tac_toe_env_environment import (
    TicTacToeEnvironment,
    _minimax_move,
    _winner,
)


@pytest.mark.parametrize(
    "board,expected",
    [
        ([0] * 9, None),
        ([1, 1, 1, 0, 0, 0, 0, 0, 0], 1),  # top row
        ([0, 0, 0, -1, -1, -1, 0, 0, 0], -1),  # middle row
        ([1, 0, 0, 1, 0, 0, 1, 0, 0], 1),  # left column
        ([1, 0, 0, 0, 1, 0, 0, 0, 1], 1),  # main diagonal
        ([0, 0, 1, 0, 1, 0, 1, 0, 0], 1),  # anti-diagonal
        ([1, -1, 1, -1, 1, -1, -1, 1, -1], None),  # full board, no line
    ],
)
def test_winner_detects_all_lines(board, expected):
    assert _winner(board) == expected


def test_reset_returns_empty_board_with_agent_to_move():
    env = TicTacToeEnvironment()
    obs = env.reset()
    assert obs.board == [0] * 9
    assert obs.valid_actions == list(range(9))
    assert obs.done is False
    assert obs.winner is None
    assert obs.current_player == 1


def test_illegal_move_ends_episode_with_penalty():
    env = TicTacToeEnvironment(opponent="random", seed=0)
    env.reset()
    env.step(TicTacToeAction(cell=0))  # legal: X takes cell 0, O replies
    obs = env.step(TicTacToeAction(cell=0))  # illegal: cell 0 already occupied

    assert obs.done is True
    assert obs.reward == -1.0
    assert obs.metadata.get("illegal_move") is True


def test_step_after_done_is_a_noop():
    env = TicTacToeEnvironment(opponent="random", seed=0)
    obs = env.reset()
    while not obs.done:
        obs = env.step(TicTacToeAction(cell=obs.valid_actions[0]))

    obs2 = env.step(TicTacToeAction(cell=0))
    assert obs2.done is True
    assert obs2.reward == 0.0
    assert obs2.metadata != {"illegal_move": True}


def test_heuristic_opponent_blocks_agents_winning_move():
    env = TicTacToeEnvironment(opponent="heuristic", seed=0)
    board = [1, 1, 0, 0, 0, 0, 0, 0, 0]  # X threatens to win at cell 2
    assert env._opponent_move(board) == 2


def test_heuristic_opponent_takes_its_own_winning_move():
    env = TicTacToeEnvironment(opponent="heuristic", seed=0)
    board = [-1, -1, 0, 1, 1, 0, 0, 0, 0]  # O can win at cell 2
    assert env._opponent_move(board) == 2


def test_minimax_move_blocks_a_forced_loss():
    board = [1, 1, 0, 0, 0, 0, 0, 0, 0]
    assert _minimax_move(board) == 2


def test_minimax_move_takes_an_available_win():
    board = [-1, -1, 0, 1, 1, 0, 0, 0, 0]
    assert _minimax_move(board) == 2


@pytest.mark.parametrize("opponent", ["random", "heuristic", "minimax"])
def test_full_games_always_terminate_validly(opponent):
    env = TicTacToeEnvironment(opponent=opponent, seed=42)
    rng = random.Random(1)
    for _ in range(50):
        obs = env.reset()
        while not obs.done:
            cell = rng.choice(obs.valid_actions)
            obs = env.step(TicTacToeAction(cell=cell))
        assert obs.winner in (1, -1, 0)
        assert obs.reward in (1.0, -1.0, 0.0)


def test_minimax_opponent_never_loses():
    """A perfectly-played O should never lose, regardless of X's (random) play."""
    env = TicTacToeEnvironment(opponent="minimax", seed=7)
    rng = random.Random(3)
    for _ in range(30):
        obs = env.reset()
        while not obs.done:
            cell = rng.choice(obs.valid_actions)
            obs = env.step(TicTacToeAction(cell=cell))
        assert obs.winner != 1
