# SPDX-License-Identifier: BSD-3-Clause

"""
Tic Tac Toe Environment Implementation.

The agent plays "X" and always moves first. A built-in opponent plays "O"
and moves automatically at the end of every `step()` call, unless the
agent's move already ended the game. Three opponent policies are
available (see `_opponent_move`): "random" (uniform random), "heuristic"
(win if possible, else block, else random — beatable, a reasonable
training adversary), and "minimax" (exhaustive optimal play — never
loses, useful as an upper-bound evaluation baseline).
"""

import random
from functools import lru_cache
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import TicTacToeAction, TicTacToeObservation
except ImportError:
    from models import TicTacToeAction, TicTacToeObservation

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
    (0, 4, 8), (2, 4, 6),             # diagonals
]


def _winner(board: list[int]) -> int | None:
    """Return 1/-1 if that player has three in a row, else None."""
    for a, b, c in WIN_LINES:
        total = board[a] + board[b] + board[c]
        if total == 3:
            return 1
        if total == -3:
            return -1
    return None


def _empty_cells(board: list[int]) -> list[int]:
    return [i for i, v in enumerate(board) if v == 0]


@lru_cache(maxsize=None)
def _minimax_score(board: tuple[int, ...], player: int) -> int:
    """Fully-solved score of `board` with `player` to move next.

    Returns 1 if X can force a win, -1 if O can force a win, 0 for a
    forced draw, assuming both sides play optimally from here on.
    Memoized because the same sub-boards recur across many top-level
    calls; the whole game tree is small enough (<= 9! nodes, far fewer
    in practice) to solve exhaustively rather than needing alpha-beta
    pruning or a fixed search depth.
    """
    winner = _winner(list(board))
    if winner is not None:
        return winner
    empties = _empty_cells(list(board))
    if not empties:
        return 0

    scores = []
    for cell in empties:
        trial = list(board)
        trial[cell] = player
        scores.append(_minimax_score(tuple(trial), -player))
    return max(scores) if player == 1 else min(scores)


def _minimax_move(board: list[int]) -> int | None:
    """Optimal move for "O" (-1): the cell minimizing X's best achievable score.

    Returns None if `board` has no empty cells (never happens in
    practice: `_opponent_move` only calls this after the agent's move,
    which is only reached when at least one cell is still empty).
    """
    best_cell = None
    best_score = None
    for cell in _empty_cells(board):
        trial = list(board)
        trial[cell] = -1
        score = _minimax_score(tuple(trial), 1)
        if best_score is None or score < best_score:
            best_score = score
            best_cell = cell
    return best_cell


class TicTacToeEnvironment(Environment):
    """
    Single-agent Tic Tac Toe environment with a scripted opponent.

    The agent always plays "X" (+1) and moves first; the built-in opponent
    plays "O" (-1) and responds automatically after each agent move.

    Example:
        >>> env = TicTacToeEnvironment()
        >>> obs = env.reset()
        >>> obs.board  # [0, 0, 0, 0, 0, 0, 0, 0, 0]
        >>>
        >>> obs = env.step(TicTacToeAction(cell=4))  # agent plays center
        >>> obs.done  # False (opponent has replied, game continues)
    """

    # This environment holds per-instance board state, so each WebSocket
    # session (factory mode) gets its own isolated game.
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, opponent: str = "heuristic", seed: int | None = None):
        """
        Args:
            opponent: "heuristic" (win/block/random), "random", or
                "minimax" (optimal play — never loses; a good agent can at
                best force a draw against it).
            seed: Optional RNG seed for reproducible opponent play.
        """
        self._opponent_kind = opponent
        self._rng = random.Random(seed)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._board: list[int] = [0] * 9
        self._done = False

    def reset(self) -> TicTacToeObservation:
        """Start a new game with an empty board. The agent (X) moves first."""
        self._board = [0] * 9
        self._done = False
        self._state = State(episode_id=str(uuid4()), step_count=0)

        return TicTacToeObservation(
            board=list(self._board),
            valid_actions=_empty_cells(self._board),
            winner=None,
            current_player=1,
            message="New game. You are X and move first.",
            done=False,
            reward=0.0,
        )

    def step(self, action: TicTacToeAction) -> TicTacToeObservation:  # type: ignore[override]
        """
        Apply the agent's move, then (if the game continues) the opponent's
        reply.

        Reward convention:
            +1.0  agent wins
            -1.0  agent loses, or plays an illegal move (episode ends)
             0.0  draw, or the game is still in progress
        """
        self._state.step_count += 1

        if self._done:
            return TicTacToeObservation(
                board=list(self._board),
                valid_actions=_empty_cells(self._board),
                winner=_winner(self._board),
                current_player=1,
                message="Episode already finished; call reset().",
                done=True,
                reward=0.0,
            )

        cell = action.cell
        if cell < 0 or cell > 8 or self._board[cell] != 0:
            # Illegal move: end the episode with a penalty rather than
            # silently ignoring it, so the agent learns to only pick from
            # `valid_actions`.
            self._done = True
            return TicTacToeObservation(
                board=list(self._board),
                valid_actions=_empty_cells(self._board),
                winner=-1,
                current_player=1,
                message=f"Illegal move: cell {cell} is not empty.",
                done=True,
                reward=-1.0,
                metadata={"illegal_move": True},
            )

        # Agent's move ("X" = 1)
        self._board[cell] = 1
        win = _winner(self._board)
        if win == 1:
            self._done = True
            return self._terminal_observation(win, "You win!")
        if not _empty_cells(self._board):
            self._done = True
            return self._terminal_observation(0, "Draw.")

        # Opponent's move ("O" = -1)
        opp_cell = self._opponent_move(self._board)
        self._board[opp_cell] = -1
        win = _winner(self._board)
        if win == -1:
            self._done = True
            return self._terminal_observation(win, "Opponent wins.")
        if not _empty_cells(self._board):
            self._done = True
            return self._terminal_observation(0, "Draw.")

        return TicTacToeObservation(
            board=list(self._board),
            valid_actions=_empty_cells(self._board),
            winner=None,
            current_player=1,
            message="Game in progress.",
            done=False,
            reward=0.0,
            metadata={"step": self._state.step_count, "opponent_cell": opp_cell},
        )

    def _terminal_observation(self, winner: int, message: str) -> TicTacToeObservation:
        reward = {1: 1.0, -1: -1.0, 0: 0.0}[winner]
        return TicTacToeObservation(
            board=list(self._board),
            valid_actions=[],
            winner=winner,
            current_player=1,
            message=message,
            done=True,
            reward=reward,
            metadata={"step": self._state.step_count},
        )

    def _opponent_move(self, board: list[int]) -> int:
        """Pick the opponent's cell: win if possible, else block, else random."""
        empties = _empty_cells(board)

        if self._opponent_kind == "random":
            return self._rng.choice(empties)

        if self._opponent_kind == "minimax":
            move = _minimax_move(board)
            assert move is not None  # `empties` above is non-empty
            return move

        # Heuristic: try to win, then block the agent, then play randomly.
        for player in (-1, 1):
            for cell in empties:
                trial = list(board)
                trial[cell] = player
                if _winner(trial) == player:
                    return cell
        return self._rng.choice(empties)

    @property
    def state(self) -> State:
        """Current State (episode_id and step_count)."""
        return self._state
