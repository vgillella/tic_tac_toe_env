# SPDX-License-Identifier: BSD-3-Clause

"""
Data models for the Tic Tac Toe environment.

The agent always plays "X" and moves first. After each agent move, the
environment automatically plays the opponent's ("O") move, so one call to
`step()` advances the game by one full round (unless the agent's move ends
the game).
"""

from typing import List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class TicTacToeAction(Action):
    """Action for the Tic Tac Toe environment.

    Attributes:
        cell: Board position to place the agent's mark, 0-8, indexed left to
            right, top to bottom:
                0 | 1 | 2
                3 | 4 | 5
                6 | 7 | 8
    """

    cell: int = Field(..., ge=0, le=8, description="Board cell index (0-8) to play")


class TicTacToeObservation(Observation):
    """Observation from the Tic Tac Toe environment.

    Attributes:
        board: 9-element board, one entry per cell. 1 = agent ("X"),
            -1 = opponent ("O"), 0 = empty.
        valid_actions: Cell indices that are currently empty and legal to play.
        winner: 1 if the agent won, -1 if the opponent won, 0 for a draw,
            None while the game is still in progress.
        current_player: Always 1 (the agent plays "X"); included for clarity
            and to keep the observation self-describing.
        message: Human-readable status message.
    """

    board: List[int] = Field(default_factory=lambda: [0] * 9, description="3x3 board flattened row-major")
    valid_actions: List[int] = Field(default_factory=list, description="Empty cell indices available to play")
    winner: Optional[int] = Field(default=None, description="1=agent win, -1=opponent win, 0=draw, None=ongoing")
    current_player: int = Field(default=1, description="Player to move next (always 1, the agent)")
    message: str = Field(default="", description="Human-readable status message")
