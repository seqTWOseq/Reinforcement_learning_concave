from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gomoku_project.core.constants import BLACK, DRAW, WHITE
from gomoku_project.core.utils import copy_info_dict
from gomoku_project.envs.gomoku_env import GomokuEnv
from gomoku_project.players.base import BasePlayer
from gomoku_project.players.heuristic_player import HeuristicPlayer
from gomoku_project.players.human_player import HumanPlayer
from gomoku_project.players.ppo_player import PPOPlayer, PPOPlayerAction
from gomoku_project.players.rl_player import RLPlayer
from gomoku_project.rl.ppo_trainer import PPORolloutStep, PPOTrainer
from gomoku_project.rl.replay_buffer import AlphaZeroExample
from gomoku_project.rl.trainer import AlphaZeroTrainer


@dataclass(frozen=True)
class MatchParticipant:
    player: BasePlayer
    trainer: PPOTrainer | AlphaZeroTrainer | None = None
    algorithm: str | None = None


@dataclass(frozen=True)
class ParticipantTrainingSummary:
    color: int
    color_label: str
    player_name: str
    algorithm: str
    stored_items: int
    buffer_size: int
    update_performed: bool
    update_metrics: dict[str, float] | None


@dataclass(frozen=True)
class TrainingMatchResult:
    matchup: str
    black_name: str
    white_name: str
    winner: int | None
    winner_name: str
    reason: str
    move_count: int
    actions: list[int]
    ppo_steps_collected: int
    alphazero_examples_collected: int
    participant_summaries: list[ParticipantTrainingSummary]
    final_info: dict[str, Any]


@dataclass(frozen=True)
class _PendingAlphaZeroStep:
    observation: np.ndarray
    policy_target: np.ndarray
    acting_player: int
    action: int


@dataclass
class _ParticipantCollector:
    participant: MatchParticipant
    color: int
    ppo_steps: list[PPORolloutStep] = field(default_factory=list)
    alphazero_steps: list[_PendingAlphaZeroStep] = field(default_factory=list)


def run_training_match(
    *,
    black_participant: MatchParticipant,
    white_participant: MatchParticipant,
    matchup: str,
    env: GomokuEnv | None = None,
    render: bool = False,
    renderer: Any | None = None,
    move_delay: float = 0.0,
    post_game_delay: float = 0.0,
    close_renderer: bool = True,
    train_after_game: bool = True,
    alphazero_train_steps: int = 1,
) -> TrainingMatchResult:
    env = env or GomokuEnv()
    black_participant = _validated_participant(black_participant)
    white_participant = _validated_participant(white_participant)
    participants = {
        BLACK: black_participant,
        WHITE: white_participant,
    }
    collectors = {
        BLACK: _ParticipantCollector(participant=black_participant, color=BLACK),
        WHITE: _ParticipantCollector(participant=white_participant, color=WHITE),
    }

    if render and renderer is None:
        from gomoku_project.ui.tkinter_renderer import TkinterRenderer

        renderer = TkinterRenderer(board_size=env.board_size)

    black_participant.player.set_player_id(BLACK)
    white_participant.player.set_player_id(WHITE)
    black_participant.player.set_renderer(renderer if render else None)
    white_participant.player.set_renderer(renderer if render else None)
    black_participant.player.reset()
    white_participant.player.reset()

    observation, info = env.reset()
    terminated = False
    truncated = False
    actions: list[int] = []

    try:
        if render and renderer is not None:
            renderer.render(
                board=info["board"],
                current_player=info["current_player"],
                last_action=info["last_action"],
            )

        while not (terminated or truncated):
            current_player = int(info["current_player"])
            participant = participants[current_player]
            active_player = participant.player
            active_player.last_policy_target = None
            if isinstance(active_player, PPOPlayer):
                active_player.last_action_info = None

            action = int(active_player.select_action(observation, info["valid_actions"], info))
            actions.append(action)
            _capture_learning_step(
                collector=collectors[current_player],
                observation=observation,
                action=action,
            )

            observation, _, terminated, truncated, info = env.step(action)

            if render and renderer is not None:
                renderer.render(
                    board=info["board"],
                    current_player=info["current_player"],
                    last_action=info["last_action"],
                )
                if move_delay > 0:
                    time.sleep(move_delay)

        if render and renderer is not None and post_game_delay > 0:
            end_time = time.time() + post_game_delay
            while time.time() < end_time:
                renderer.process_events()
                time.sleep(0.05)

        final_info = copy_info_dict(info)
        winner = final_info.get("winner")
        reason = str(final_info.get("reason", "unknown"))

        stored_items_by_color: dict[int, int] = {}
        buffer_sizes_by_color: dict[int, int] = {}

        for color, collector in collectors.items():
            participant = collector.participant
            trainer = participant.trainer
            algorithm = participant.algorithm

            if algorithm == "ppo" and isinstance(trainer, PPOTrainer):
                stored_steps = trainer.record_external_episode(
                    collector.ppo_steps,
                    controlled_player=color,
                    winner=winner,
                    reason=reason,
                    opponent_type=_participant_kind(participants[_opponent_color(color)]),
                )
                stored_items_by_color[color] = stored_steps
                buffer_sizes_by_color[color] = len(trainer.buffer)
                continue

            if algorithm == "alphazero" and isinstance(trainer, AlphaZeroTrainer):
                examples = [
                    AlphaZeroExample(
                        observation=step.observation.copy(),
                        policy_target=step.policy_target.copy(),
                        value_target=_winner_to_value(winner, step.acting_player),
                        info={
                            "winner": winner,
                            "reason": reason,
                            "acting_player": step.acting_player,
                            "last_action": step.action,
                            "matchup": matchup,
                            "opponent_type": _participant_kind(participants[_opponent_color(color)]),
                        },
                    )
                    for step in collector.alphazero_steps
                ]
                stored_items_by_color[color] = len(examples)
                buffer_sizes_by_color[color] = trainer.add_examples(examples) if examples else len(trainer.replay_buffer)

        update_performed_by_trainer: dict[int, bool] = {}
        update_metrics_by_trainer: dict[int, dict[str, float] | None] = {}
        grouped_trainers = _group_learning_trainers(participants)

        if train_after_game:
            for trainer_id, trainer_data in grouped_trainers.items():
                trainer = trainer_data["trainer"]
                colors = trainer_data["colors"]
                total_stored_items = sum(stored_items_by_color.get(color, 0) for color in colors)
                if total_stored_items <= 0:
                    update_performed_by_trainer[trainer_id] = False
                    update_metrics_by_trainer[trainer_id] = None
                    continue

                if isinstance(trainer, PPOTrainer):
                    metrics = trainer.update()
                    update_performed_by_trainer[trainer_id] = metrics is not None
                    update_metrics_by_trainer[trainer_id] = metrics
                else:
                    if alphazero_train_steps <= 0:
                        update_performed_by_trainer[trainer_id] = False
                        update_metrics_by_trainer[trainer_id] = None
                        continue
                    metrics_list = trainer.train_steps(alphazero_train_steps)
                    update_performed_by_trainer[trainer_id] = bool(metrics_list)
                    update_metrics_by_trainer[trainer_id] = _summarize_alphazero_metrics(metrics_list)

        participant_summaries: list[ParticipantTrainingSummary] = []
        for color in (BLACK, WHITE):
            participant = participants[color]
            trainer = participant.trainer
            algorithm = participant.algorithm
            if trainer is None or algorithm is None:
                continue

            trainer_id = id(trainer)
            participant_summaries.append(
                ParticipantTrainingSummary(
                    color=color,
                    color_label=_color_label(color),
                    player_name=participant.player.name,
                    algorithm=algorithm,
                    stored_items=stored_items_by_color.get(color, 0),
                    buffer_size=buffer_sizes_by_color.get(
                        color,
                        len(trainer.buffer) if isinstance(trainer, PPOTrainer) else len(trainer.replay_buffer),
                    ),
                    update_performed=update_performed_by_trainer.get(trainer_id, False),
                    update_metrics=update_metrics_by_trainer.get(trainer_id),
                )
            )

        return TrainingMatchResult(
            matchup=matchup,
            black_name=black_participant.player.name,
            white_name=white_participant.player.name,
            winner=winner,
            winner_name=_winner_name(winner, black_participant.player.name, white_participant.player.name),
            reason=reason,
            move_count=len(actions),
            actions=actions,
            ppo_steps_collected=sum(summary.stored_items for summary in participant_summaries if summary.algorithm == "ppo"),
            alphazero_examples_collected=sum(
                summary.stored_items for summary in participant_summaries if summary.algorithm == "alphazero"
            ),
            participant_summaries=participant_summaries,
            final_info=final_info,
        )
    finally:
        env.close()
        if render and renderer is not None and close_renderer:
            renderer.close()


def _validated_participant(participant: MatchParticipant) -> MatchParticipant:
    algorithm = _resolve_algorithm(participant)
    trainer = participant.trainer
    player = participant.player

    if trainer is None:
        if algorithm in {"ppo", "alphazero"}:
            raise ValueError("Learning participants require a trainer instance.")
        return MatchParticipant(player=player, trainer=None, algorithm=None)

    if isinstance(trainer, PPOTrainer):
        if algorithm != "ppo":
            raise ValueError("PPOTrainer must be paired with a PPO participant.")
        if not isinstance(player, PPOPlayer):
            raise TypeError("PPO learning participants must use PPOPlayer.")
        return MatchParticipant(player=player, trainer=trainer, algorithm="ppo")

    if isinstance(trainer, AlphaZeroTrainer):
        if algorithm != "alphazero":
            raise ValueError("AlphaZeroTrainer must be paired with an AlphaZero participant.")
        if not isinstance(player, RLPlayer):
            raise TypeError("AlphaZero learning participants must use RLPlayer.")
        return MatchParticipant(player=player, trainer=trainer, algorithm="alphazero")

    raise TypeError(f"Unsupported trainer type: {type(trainer).__name__}")


def _resolve_algorithm(participant: MatchParticipant) -> str | None:
    if participant.algorithm is not None:
        return participant.algorithm
    if isinstance(participant.trainer, PPOTrainer) or isinstance(participant.player, PPOPlayer):
        return "ppo"
    if isinstance(participant.trainer, AlphaZeroTrainer) or isinstance(participant.player, RLPlayer):
        return "alphazero"
    return None


def _participant_kind(participant: MatchParticipant) -> str:
    if participant.algorithm is not None:
        return participant.algorithm
    if isinstance(participant.player, HumanPlayer):
        return "human"
    if isinstance(participant.player, HeuristicPlayer):
        return "heuristic"
    return type(participant.player).__name__.lower()


def _capture_learning_step(
    *,
    collector: _ParticipantCollector,
    observation: np.ndarray,
    action: int,
) -> None:
    participant = collector.participant
    player = participant.player
    action_space_size = observation.shape[0] * observation.shape[1]

    if participant.algorithm == "ppo":
        action_info = getattr(player, "last_action_info", None)
        if not isinstance(action_info, PPOPlayerAction):
            raise RuntimeError("PPO learning participants must expose PPOPlayerAction metadata.")
        collector.ppo_steps.append(
            PPORolloutStep(
                observation=np.asarray(observation, dtype=np.int8).copy(),
                action_mask=np.asarray(action_info.action_mask, dtype=bool).copy(),
                action=int(action),
                log_prob=float(action_info.log_prob),
                value=float(action_info.value),
            )
        )
        return

    if participant.algorithm == "alphazero":
        collector.alphazero_steps.append(
            _PendingAlphaZeroStep(
                observation=np.asarray(observation, dtype=np.int8).copy(),
                policy_target=_coerce_policy_target(player.last_policy_target, action, action_space_size),
                acting_player=collector.color,
                action=int(action),
            )
        )


def _coerce_policy_target(
    policy_target: np.ndarray | None,
    action: int,
    action_space_size: int,
) -> np.ndarray:
    if policy_target is None:
        return _one_hot(action, action_space_size)

    normalized = np.asarray(policy_target, dtype=np.float32)
    if normalized.shape != (action_space_size,) or not np.isfinite(normalized).all():
        return _one_hot(action, action_space_size)
    return normalized.copy()


def _one_hot(action: int, action_space_size: int) -> np.ndarray:
    target = np.zeros(action_space_size, dtype=np.float32)
    target[int(action)] = 1.0
    return target


def _winner_to_value(winner: int | None, acting_player: int) -> float:
    if winner is None or winner == DRAW:
        return 0.0
    return 1.0 if winner == acting_player else -1.0


def _group_learning_trainers(
    participants: dict[int, MatchParticipant],
) -> dict[int, dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}
    for color, participant in participants.items():
        if participant.trainer is None:
            continue
        trainer_id = id(participant.trainer)
        entry = grouped.setdefault(
            trainer_id,
            {
                "trainer": participant.trainer,
                "colors": [],
            },
        )
        entry["colors"].append(color)
    return grouped


def _summarize_alphazero_metrics(metrics: list[dict[str, float]]) -> dict[str, float] | None:
    if not metrics:
        return None
    return {
        "total_loss": _mean_metric(metrics, "total_loss"),
        "policy_loss": _mean_metric(metrics, "policy_loss"),
        "value_loss": _mean_metric(metrics, "value_loss"),
        "train_steps": float(len(metrics)),
    }


def _mean_metric(metrics: list[dict[str, float]], key: str) -> float:
    return float(sum(metric[key] for metric in metrics) / len(metrics))


def _color_label(color: int) -> str:
    return "black" if color == BLACK else "white"


def _opponent_color(color: int) -> int:
    return WHITE if color == BLACK else BLACK


def _winner_name(winner: int | None, black_name: str, white_name: str) -> str:
    if winner == BLACK:
        return black_name
    if winner == WHITE:
        return white_name
    return "Draw"
