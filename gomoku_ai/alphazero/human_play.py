"""Human-vs-AlphaZero game runner and recording.

Human play differs from self-play in three important ways:
- `GameRecord.source` is always `"human_play"`, not `"self_play"`
- the AI uses evaluation-style search by default (`root_noise=False`,
  `temperature=0.0`)
- this stage records only AI turns as training samples
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

import numpy as np

from gomoku_ai.alphazero.human_play_utils import (
    REQUIRED_HUMAN_PLAY_METADATA_KEYS,
    build_human_play_game_id,
    build_human_play_mcts_config,
    finalize_human_ai_turn_records,
    resolve_human_and_ai_colors,
)
from gomoku_ai.alphazero.mcts import MCTS, MCTSConfig
from gomoku_ai.alphazero.model import PolicyValueNet
from gomoku_ai.alphazero.specs import ACTION_SIZE, GameRecord, POLICY_SHAPE, STATE_SHAPE
from gomoku_ai.common.agents import BaseAgent
from gomoku_ai.env import BLACK, DRAW, GomokuEnv, WHITE


@dataclass(frozen=True)
class HumanPlayConfig:
    """Configuration for human-vs-AlphaZero games.

    Fixed default behavior for this stage:
    - `human_color = "select_each_game"`
    - `ai_temperature = 0.0`
    - `use_root_noise = False`
    - `ai_num_simulations = 50`
    - `record_ai_turn_only = True`
    - `game_id_prefix = "humanplay"`
    """

    human_color: str | int = "select_each_game"
    ai_temperature: float = 0.0
    use_root_noise: bool = False
    ai_num_simulations: int = 50
    record_ai_turn_only: bool = True
    game_id_prefix: str = "humanplay"

    def __post_init__(self) -> None:
        """Validate human-play defaults and stage-8 limitations."""

        resolve_human_and_ai_colors(
            human_color=self.human_color if self.human_color != "select_each_game" else BLACK,
            default_human_color=BLACK,
        )
        if self.ai_temperature < 0.0:
            raise ValueError("ai_temperature must be non-negative.")
        if not isinstance(self.use_root_noise, bool):
            raise ValueError("use_root_noise must be a bool.")
        if self.ai_num_simulations <= 0:
            raise ValueError("ai_num_simulations must be positive.")
        if not isinstance(self.record_ai_turn_only, bool):
            raise ValueError("record_ai_turn_only must be a bool.")
        if not self.record_ai_turn_only:
            raise ValueError("Stage 8 baseline currently supports only AI-turn-only samples.")
        if not self.game_id_prefix or not self.game_id_prefix.strip():
            raise ValueError("game_id_prefix must be a non-empty string.")


@dataclass(frozen=True)
class HumanAITurnRecord:
    """Per-turn AI record captured before the AI action is applied.

    Contract:
    - `state` is the encoded pre-action state
    - `policy_target` is the AI MCTS root visit-count distribution after
      applying the configured AI temperature
    - `player_to_move` is always the AI color for that recorded turn
    """

    state: np.ndarray
    policy_target: np.ndarray
    player_to_move: int
    move_index: int
    action_taken: int

    def __post_init__(self) -> None:
        """Validate the AI-turn record against AlphaZero tensor contracts."""

        state = np.asarray(self.state, dtype=np.float32)
        policy_target = np.asarray(self.policy_target, dtype=np.float32)

        if state.shape != STATE_SHAPE:
            raise ValueError(f"state must have shape {STATE_SHAPE}, got {state.shape}.")
        if not np.isfinite(state).all():
            raise ValueError("state must contain only finite values.")

        if policy_target.shape != POLICY_SHAPE:
            raise ValueError(f"policy_target must have shape {POLICY_SHAPE}, got {policy_target.shape}.")
        if not np.isfinite(policy_target).all():
            raise ValueError("policy_target must contain only finite values.")
        if np.any(policy_target < 0.0):
            raise ValueError("policy_target must not contain negative probabilities.")
        if not np.isclose(float(policy_target.sum()), 1.0, atol=1e-5):
            raise ValueError(f"policy_target must sum to 1.0, got {float(policy_target.sum()):.6f}.")

        if self.player_to_move not in {BLACK, WHITE}:
            raise ValueError(f"player_to_move must be BLACK({BLACK}) or WHITE({WHITE}).")
        if self.move_index < 0:
            raise ValueError("move_index must be non-negative.")
        if not 0 <= self.action_taken < ACTION_SIZE:
            raise ValueError(f"action_taken must be in [0, {ACTION_SIZE - 1}].")

        object.__setattr__(self, "state", state.astype(np.float32, copy=False))
        object.__setattr__(self, "policy_target", policy_target.astype(np.float32, copy=False))


class HumanVsAlphaZeroGameRunner:
    """Run one human-vs-AlphaZero game and return a `human_play` record.

    This baseline stores:
    - the full move list in `GameRecord.moves`
    - only AI turns in `GameRecord.samples`

    The AI acts in evaluation-style mode:
    - `root_noise=False` by default
    - `temperature=0.0` by default
    - `ai_num_simulations` controls MCTS search effort
    """

    def __init__(
        self,
        config: HumanPlayConfig | None = None,
        env_factory: Callable[[], GomokuEnv] | None = None,
        human_input_provider: Callable[[GomokuEnv], int | str] | None = None,
        color_selection_provider: Callable[[], str | int] | None = None,
    ) -> None:
        """Initialize the runner with optional input callbacks."""

        self.config = config or HumanPlayConfig()
        self.env_factory = env_factory or GomokuEnv
        self.human_input_provider = human_input_provider
        self.color_selection_provider = color_selection_provider

    def play_game(
        self,
        model: PolicyValueNet,
        human_color: str | int | None = None,
        human_moves: list[int] | None = None,
    ) -> GameRecord:
        """Play one human-vs-AI game and return a `GameRecord`.

        Human moves are always applied directly to the environment.
        AI turns are recorded before the action:
        - `state` is captured pre-action
        - `policy_target` comes from root visit counts with AI temperature
        - samples are stored only for AI turns
        """

        if not isinstance(model, PolicyValueNet):
            raise TypeError("model must be a PolicyValueNet instance.")

        human_player, ai_player = resolve_human_and_ai_colors(
            human_color=human_color,
            default_human_color=self.config.human_color,
            selection_callback=self._select_human_color,
        )

        env = self.env_factory()
        if not isinstance(env, GomokuEnv):
            raise TypeError("env_factory must create GomokuEnv instances.")
        env.reset()

        game_id = build_human_play_game_id(self.config.game_id_prefix)
        move_history: list[int] = []
        ai_turn_records: list[HumanAITurnRecord] = []
        scripted_human_moves = iter(human_moves) if human_moves is not None else None
        ai_mcts_config = self._build_ai_mcts_config()

        was_training = model.training
        model.eval()
        try:
            while not env.done:
                move_index = len(move_history)
                if env.current_player == human_player:
                    action = self._get_human_action(env, scripted_human_moves)
                    move_history.append(action)
                    env.apply_move(action)
                    continue

                state = np.asarray(env.encode_state(), dtype=np.float32).copy()
                policy_target, action = self._select_ai_action(env, model, ai_mcts_config)
                ai_turn_records.append(
                    HumanAITurnRecord(
                        state=state,
                        policy_target=policy_target,
                        player_to_move=env.current_player,
                        move_index=move_index,
                        action_taken=action,
                    )
                )
                move_history.append(action)
                env.apply_move(action)
        finally:
            if was_training:
                model.train()
            else:
                model.eval()

        if env.winner is None:
            raise RuntimeError("Human-vs-AI game ended without a resolved winner value.")

        samples = finalize_human_ai_turn_records(ai_turn_records, env.winner, game_id)
        metadata = {
            "human_color": human_player,
            "ai_color": ai_player,
            "num_moves": len(move_history),
            "ai_temperature": self.config.ai_temperature,
            "use_root_noise": self.config.use_root_noise,
            "ai_num_simulations": self.config.ai_num_simulations,
            "record_ai_turn_only": self.config.record_ai_turn_only,
        }
        missing_required_keys = [key for key in REQUIRED_HUMAN_PLAY_METADATA_KEYS if key not in metadata]
        if missing_required_keys:
            raise RuntimeError(f"Missing required human-play metadata keys: {missing_required_keys}")

        return GameRecord(
            game_id=game_id,
            moves=move_history,
            winner=env.winner,
            source="human_play",
            samples=samples,
            metadata=metadata,
        )

    def play_game_with_agent(
        self,
        ai_agent: BaseAgent,
        human_color: str | int | None = None,
        human_moves: list[int] | None = None,
    ) -> GameRecord:
        """Play one human-vs-agent game through the shared `BaseAgent` interface.

        TODO: Replace one-hot policy targets with Athenan/agent search
        distributions once a search API is finalized.
        """

        if not isinstance(ai_agent, BaseAgent):
            raise TypeError("ai_agent must implement BaseAgent.")

        human_player, ai_player = resolve_human_and_ai_colors(
            human_color=human_color,
            default_human_color=self.config.human_color,
            selection_callback=self._select_human_color,
        )

        env = self.env_factory()
        if not isinstance(env, GomokuEnv):
            raise TypeError("env_factory must create GomokuEnv instances.")
        env.reset()

        game_id = build_human_play_game_id(self.config.game_id_prefix)
        move_history: list[int] = []
        ai_turn_records: list[HumanAITurnRecord] = []
        scripted_human_moves = iter(human_moves) if human_moves is not None else None

        while not env.done:
            move_index = len(move_history)
            if env.current_player == human_player:
                action = self._get_human_action(env, scripted_human_moves)
                move_history.append(action)
                env.apply_move(action)
                continue

            state = np.asarray(env.encode_state(), dtype=np.float32).copy()
            policy_target, action = self._select_ai_action_from_agent(env, ai_agent)
            ai_turn_records.append(
                HumanAITurnRecord(
                    state=state,
                    policy_target=policy_target,
                    player_to_move=env.current_player,
                    move_index=move_index,
                    action_taken=action,
                )
            )
            move_history.append(action)
            env.apply_move(action)

        if env.winner is None:
            raise RuntimeError("Human-vs-AI game ended without a resolved winner value.")

        samples = finalize_human_ai_turn_records(ai_turn_records, env.winner, game_id)
        metadata = {
            "human_color": human_player,
            "ai_color": ai_player,
            "num_moves": len(move_history),
            "ai_temperature": self.config.ai_temperature,
            "use_root_noise": self.config.use_root_noise,
            "ai_num_simulations": self.config.ai_num_simulations,
            "record_ai_turn_only": self.config.record_ai_turn_only,
            "ai_driver": "external_agent",
        }
        missing_required_keys = [key for key in REQUIRED_HUMAN_PLAY_METADATA_KEYS if key not in metadata]
        if missing_required_keys:
            raise RuntimeError(f"Missing required human-play metadata keys: {missing_required_keys}")

        return GameRecord(
            game_id=game_id,
            moves=move_history,
            winner=env.winner,
            source="human_play",
            samples=samples,
            metadata=metadata,
        )

    def _build_ai_mcts_config(self) -> MCTSConfig:
        """Build the evaluation-style MCTS config used for AI turns."""

        return build_human_play_mcts_config(self.config)

    def _select_ai_action(
        self,
        env: GomokuEnv,
        model: PolicyValueNet,
        mcts_config: MCTSConfig,
    ) -> tuple[np.ndarray, int]:
        """Run one AI search step and return `(policy_target, action)`."""

        mcts = MCTS(mcts_config)
        root = mcts.run(env, model)
        policy_target = mcts.get_action_probs(root, temperature=mcts_config.temperature)
        action = mcts.select_action(root, temperature=mcts_config.temperature)
        return policy_target, action

    def _select_ai_action_from_agent(
        self,
        env: GomokuEnv,
        ai_agent: BaseAgent,
    ) -> tuple[np.ndarray, int]:
        """Call a shared-interface agent and normalize to `(policy_target, action)`."""

        action = int(ai_agent.select_action(env))
        env.action_to_coord(action)
        if not env.get_valid_moves()[action]:
            raise ValueError(f"External AI selected an illegal action: {action}.")

        policy_target = np.zeros(POLICY_SHAPE, dtype=np.float32)
        policy_target[action] = 1.0
        return policy_target, action

    def _get_human_action(
        self,
        env: GomokuEnv,
        scripted_human_moves: Iterator[int] | None,
    ) -> int:
        """Get the next legal human move from scripted or interactive input."""

        if scripted_human_moves is not None:
            try:
                raw_move = next(scripted_human_moves)
            except StopIteration as exc:
                raise ValueError("Not enough scripted human moves were provided.") from exc
            return self._normalize_human_action(env, raw_move)

        while True:
            raw_move = self._read_human_move(env)
            try:
                return self._normalize_human_action(env, raw_move)
            except ValueError as exc:
                print(f"Invalid move: {exc}")

    def _read_human_move(self, env: GomokuEnv) -> int | str:
        """Read a raw human move from the configured provider or standard input."""

        if self.human_input_provider is not None:
            return self.human_input_provider(env)
        return input("Enter your move as action index or 'row,col': ")

    def _select_human_color(self) -> str | int:
        """Resolve a per-game human color choice interactively or via callback."""

        if self.color_selection_provider is not None:
            return self.color_selection_provider()

        while True:
            raw_choice = input("Choose your color ('black' or 'white'): ")
            normalized = raw_choice.strip().lower()
            if normalized in {"black", "white", "b", "w"}:
                return raw_choice
            print("Invalid color selection. Enter 'black' or 'white'.")

    def _normalize_human_action(self, env: GomokuEnv, raw_move: int | str) -> int:
        """Normalize one human move into a validated flat action index."""

        if isinstance(raw_move, int):
            action = raw_move
        elif isinstance(raw_move, str):
            normalized = raw_move.strip()
            if "," in normalized:
                row_text, col_text = normalized.split(",", maxsplit=1)
                action = env.coord_to_action(int(row_text.strip()), int(col_text.strip()))
            else:
                action = int(normalized)
        else:
            raise ValueError("Human move must be an int action or a 'row,col' string.")

        env.action_to_coord(action)
        if not env.get_valid_moves()[action]:
            raise ValueError(f"Action {action} is not legal in the current position.")
        return action
