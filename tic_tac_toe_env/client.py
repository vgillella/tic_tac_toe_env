# SPDX-License-Identifier: BSD-3-Clause

"""Tic Tac Toe Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import TicTacToeAction, TicTacToeObservation


class TicTacToeEnv(EnvClient[TicTacToeAction, TicTacToeObservation, State]):
    """
    Client for the Tic Tac Toe Environment.

    The agent plays "X" and moves first; the server's built-in opponent
    plays "O" and replies automatically after every `step()` call.

    Example:
        >>> with TicTacToeEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset()
        ...     print(result.observation.board)
        ...
        ...     result = env.step(TicTacToeAction(cell=4))
        ...     print(result.observation.board, result.reward, result.done)

    Example with Docker:
        >>> client = TicTacToeEnv.from_docker_image("tic_tac_toe_env-env:latest").sync()
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(TicTacToeAction(cell=0))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: TicTacToeAction) -> Dict:
        """Convert TicTacToeAction to JSON payload for the step message."""
        return {
            "cell": action.cell,
        }

    def _parse_result(self, payload: Dict) -> StepResult[TicTacToeObservation]:
        """Parse server response into StepResult[TicTacToeObservation]."""
        obs_data = payload.get("observation", {})
        observation = TicTacToeObservation(
            board=obs_data.get("board", [0] * 9),
            valid_actions=obs_data.get("valid_actions", []),
            winner=obs_data.get("winner"),
            current_player=obs_data.get("current_player", 1),
            message=obs_data.get("message", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=payload.get("metadata", obs_data.get("metadata", {})),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
