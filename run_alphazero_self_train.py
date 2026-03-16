"""Root entrypoint for AlphaZero Gomoku self-play training.

This script intentionally stays thin and composes the existing project
building blocks:
- `PolicyValueNet`
- `AlphaZeroTrainer`
- `AlphaZeroEvaluator`
- `TrainerConfig`
- `SelfPlayConfig`
- `MCTSConfig`
- `EvaluationConfig`

The default flow is:
1. create a policy/value network
2. run AlphaZero self-play + training for N cycles
3. optionally evaluate the current candidate against the promoted best model
   after each cycle

Example usage:
- `python run_alphazero_self_train.py`
- `python run_alphazero_self_train.py --cycles 10`
- `python run_alphazero_self_train.py --device cpu --games-per-cycle 20 --simulations 50`
- `python run_alphazero_self_train.py --with-eval`
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from gomoku_ai.alphazero import (
    AlphaZeroEvaluator,
    AlphaZeroTrainer,
    EvaluationConfig,
    EvaluationResult,
    MCTSConfig,
    PolicyValueNet,
    SelfPlayConfig,
    TrainerConfig,
    append_training_log_entry,
    build_training_log_entry,
    load_best_model_checkpoint,
    summarize_self_play_records,
    write_training_dashboard_from_log,
)


@dataclass(frozen=True)
class RunSummary:
    """Summary of one CLI-driven AlphaZero training run."""

    cycles_requested: int
    cycles_completed: int
    final_checkpoint_path: str | None
    best_model_path: str
    metrics_log_path: str | None
    dashboard_path: str | None
    eval_enabled: bool
    interrupted: bool


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer for argparse."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer.")
    return parsed


def _positive_float(value: str) -> float:
    """Parse a strictly positive float for argparse."""

    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be a positive float.")
    return parsed


def _non_negative_float(value: str) -> float:
    """Parse a non-negative float for argparse."""

    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be a non-negative float.")
    return parsed


def _zero_to_one_float(value: str) -> float:
    """Parse a float constrained to the inclusive range [0.0, 1.0]."""

    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be in the inclusive range [0.0, 1.0].")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the AlphaZero self-train entrypoint."""

    parser = argparse.ArgumentParser(
        description="Run AlphaZero self-play training for Gomoku.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cycles", type=_positive_int, default=10, help="Number of training cycles to run.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device string, for example 'cpu' or 'cuda'.")
    parser.add_argument(
        "--games-per-cycle",
        type=_positive_int,
        default=20,
        help="Number of self-play games collected in each training cycle.",
    )
    parser.add_argument(
        "--buffer-samples",
        type=_positive_int,
        default=2000,
        help="Maximum number of recent samples retained in the replay buffer.",
    )
    parser.add_argument("--batch-size", type=_positive_int, default=64, help="Training batch size.")
    parser.add_argument(
        "--epochs-per-cycle",
        type=_positive_int,
        default=3,
        help="Number of optimizer passes over the buffered samples per cycle.",
    )
    parser.add_argument(
        "--learning-rate",
        type=_positive_float,
        default=1e-3,
        help="AdamW learning rate used by the AlphaZero trainer.",
    )
    parser.add_argument(
        "--weight-decay",
        type=_non_negative_float,
        default=1e-4,
        help="AdamW weight decay used by the AlphaZero trainer.",
    )
    parser.add_argument(
        "--simulations",
        type=_positive_int,
        default=50,
        help="Number of MCTS simulations for self-play and evaluation.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/alphazero",
        help="Directory used for trainer checkpoints.",
    )
    parser.add_argument(
        "--with-eval",
        action="store_true",
        help="Evaluate the current candidate against the promoted best model after each cycle.",
    )
    parser.add_argument(
        "--eval-games",
        type=_positive_int,
        default=20,
        help="Number of games used for candidate-vs-best evaluation.",
    )
    parser.add_argument(
        "--promotion-threshold",
        type=_zero_to_one_float,
        default=0.55,
        help="Minimum candidate score required to replace the current best model.",
    )
    parser.add_argument(
        "--draw-score",
        type=_zero_to_one_float,
        default=0.5,
        help="Fractional score awarded to the candidate for a draw during evaluation.",
    )
    parser.add_argument(
        "--best-model-path",
        type=str,
        default="checkpoints/alphazero/best_model.pt",
        help="Path to the promoted best-model checkpoint.",
    )
    parser.add_argument(
        "--metrics-log-path",
        type=str,
        default=None,
        help="Optional JSONL path used to store per-cycle dashboard metrics.",
    )
    parser.add_argument(
        "--dashboard-path",
        type=str,
        default=None,
        help="Optional HTML path used for the generated training dashboard.",
    )
    parser.add_argument(
        "--append-metrics-log",
        action="store_true",
        help="Append to an existing metrics log instead of starting a fresh one.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Clamp cycle/game/simulation counts for a fast end-to-end validation run.",
    )
    return parser.parse_args(argv)


def apply_smoke_test_overrides(args: argparse.Namespace) -> argparse.Namespace:
    """Clamp expensive settings for a fast end-to-end smoke test run."""

    if not args.smoke_test:
        return args

    args.cycles = min(args.cycles, 2)
    args.games_per_cycle = min(args.games_per_cycle, 2)
    args.simulations = min(args.simulations, 4)
    args.epochs_per_cycle = min(args.epochs_per_cycle, 1)
    args.eval_games = min(args.eval_games, 2)
    return args


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _validate_device(device: str) -> str:
    """Validate the requested torch device string before model construction."""

    try:
        resolved = torch.device(device)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid torch device string: {device!r}.") from exc

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is False.")
    return str(resolved)


def _resolve_metrics_log_path(args: argparse.Namespace) -> Path:
    """Resolve the JSONL metrics log path for this training run."""

    if args.metrics_log_path:
        return Path(args.metrics_log_path)
    return Path(args.checkpoint_dir) / "training_metrics.jsonl"


def _resolve_dashboard_path(args: argparse.Namespace) -> Path:
    """Resolve the static HTML dashboard path for this training run."""

    if args.dashboard_path:
        return Path(args.dashboard_path)
    return Path(args.checkpoint_dir) / "training_dashboard.html"


def _prepare_metrics_outputs(metrics_log_path: Path, *, append: bool) -> None:
    """Prepare the metrics outputs for a new run."""

    metrics_log_path.parent.mkdir(parents=True, exist_ok=True)
    if not append and metrics_log_path.exists():
        metrics_log_path.unlink()


def build_trainer_from_args(args: argparse.Namespace) -> AlphaZeroTrainer:
    """Build an `AlphaZeroTrainer` using the CLI-provided hyperparameters."""

    trainer_config = TrainerConfig(
        num_self_play_games_per_cycle=args.games_per_cycle,
        max_buffer_samples=args.buffer_samples,
        batch_size=args.batch_size,
        epochs_per_cycle=args.epochs_per_cycle,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )
    self_play_config = SelfPlayConfig(
        opening_temperature_moves=10,
        opening_temperature=1.0,
        late_temperature=0.0,
        use_root_noise=True,
        game_id_prefix="selfplay",
    )
    mcts_config = MCTSConfig(num_simulations=args.simulations)
    return AlphaZeroTrainer(
        model=PolicyValueNet(),
        trainer_config=trainer_config,
        self_play_config=self_play_config,
        mcts_config=mcts_config,
    )


def build_evaluator_from_args(args: argparse.Namespace) -> AlphaZeroEvaluator:
    """Build an `AlphaZeroEvaluator` with deterministic evaluation settings."""

    eval_config = EvaluationConfig(
        num_eval_games=args.eval_games,
        replace_win_rate_threshold=args.promotion_threshold,
        draw_score=args.draw_score,
        eval_temperature=0.0,
        use_root_noise=False,
        eval_num_simulations=args.simulations,
        best_model_path=args.best_model_path,
    )
    return AlphaZeroEvaluator(config=eval_config)


def _build_bootstrap_result() -> EvaluationResult:
    """Create a synthetic evaluation result used when bootstrapping the first best model."""

    return EvaluationResult(
        num_games=1,
        candidate_wins=1,
        reference_wins=0,
        draws=0,
        candidate_score=1.0,
        candidate_win_rate=1.0,
        promoted=True,
        candidate_as_black_wins=1,
        candidate_as_white_wins=0,
        candidate_black_games=1,
        candidate_white_games=0,
    )


def _evaluate_and_maybe_promote(
    evaluator: AlphaZeroEvaluator,
    candidate_model: PolicyValueNet,
    cycle_index: int,
    device: str,
) -> tuple[EvaluationResult, bool, bool]:
    """Evaluate the current candidate against the best model and promote if needed.

    Returns:
        A tuple of `(result, promoted, bootstrapped)` where `bootstrapped`
        indicates that no prior best-model checkpoint existed.
    """

    best_model_path = Path(evaluator.config.best_model_path)
    if not best_model_path.exists():
        bootstrap_result = _build_bootstrap_result()
        promoted = evaluator.promote_if_better(candidate_model, cycle_index=cycle_index, result=bootstrap_result)
        return bootstrap_result, promoted, True

    reference_model = load_best_model_checkpoint(best_model_path, device=device)
    if reference_model is None:
        raise RuntimeError(f"Failed to load best model from {best_model_path}.")

    result = evaluator.play_match(candidate_model, reference_model)
    promoted = evaluator.promote_if_better(candidate_model, cycle_index=cycle_index, result=result)
    return result, promoted, False


def _print_training_metrics(cycle_index: int, metrics: dict[str, float | str]) -> None:
    """Print the required per-cycle training metrics in a readable format."""

    print(f"\n=== Cycle {cycle_index} Training ===")
    print(f"cycle_index={cycle_index}")
    print(f"num_self_play_games={int(metrics['num_self_play_games'])}")
    print(f"buffer_samples={int(metrics['buffer_samples'])}")
    if "avg_game_length" in metrics:
        print(f"avg_game_length={float(metrics['avg_game_length']):.3f}")
    if "black_win_rate" in metrics:
        print(f"black_win_rate={float(metrics['black_win_rate']):.6f}")
    if "white_win_rate" in metrics:
        print(f"white_win_rate={float(metrics['white_win_rate']):.6f}")
    if "draw_rate" in metrics:
        print(f"draw_rate={float(metrics['draw_rate']):.6f}")
    if "avg_policy_entropy" in metrics:
        print(f"avg_policy_entropy={float(metrics['avg_policy_entropy']):.6f}")
    print(f"policy_loss={float(metrics['policy_loss']):.6f}")
    print(f"value_loss={float(metrics['value_loss']):.6f}")
    print(f"total_loss={float(metrics['total_loss']):.6f}")
    if "policy_top1_accuracy" in metrics:
        print(f"policy_top1_accuracy={float(metrics['policy_top1_accuracy']):.6f}")
    if "value_outcome_accuracy" in metrics:
        print(f"value_outcome_accuracy={float(metrics['value_outcome_accuracy']):.6f}")
    if "value_mae" in metrics:
        print(f"value_mae={float(metrics['value_mae']):.6f}")
    print(f"checkpoint_path={metrics['checkpoint_path']}")


def _print_evaluation_metrics(
    cycle_index: int,
    result: EvaluationResult,
    promoted: bool,
    *,
    bootstrapped: bool,
    best_model_path: str,
) -> None:
    """Print evaluation and promotion metrics for one completed cycle."""

    print(f"=== Cycle {cycle_index} Evaluation ===")
    print(f"candidate_wins={result.candidate_wins}")
    print(f"reference_wins={result.reference_wins}")
    print(f"draws={result.draws}")
    print(f"candidate_score={result.candidate_score:.6f}")
    print(f"candidate_win_rate={result.candidate_win_rate:.6f}")
    print(f"promoted={promoted}")
    print(f"bootstrapped_best_model={bootstrapped}")
    print(f"best_model_path={best_model_path}")


def run_training_loop(
    trainer: AlphaZeroTrainer,
    args: argparse.Namespace,
    evaluator: AlphaZeroEvaluator | None = None,
    *,
    metrics_log_path: Path | None = None,
    dashboard_path: Path | None = None,
) -> RunSummary:
    """Run the requested number of AlphaZero training cycles.

    When `evaluator` is provided, the candidate model is evaluated against the
    current best model after each completed cycle. If no best model exists yet,
    the current candidate is promoted immediately via bootstrap.
    """

    cycles_completed = 0
    final_checkpoint_path: str | None = None
    interrupted = False

    try:
        for cycle_index in range(args.cycles):
            metrics = trainer.run_training_cycle(cycle_index)
            final_checkpoint_path = str(metrics["checkpoint_path"])
            self_play_summary = summarize_self_play_records(trainer.last_collected_games)
            enriched_metrics = dict(metrics)
            enriched_metrics.update(self_play_summary)
            _print_training_metrics(cycle_index, enriched_metrics)
            cycles_completed += 1

            evaluation_result: EvaluationResult | None = None
            promoted: bool | None = None
            bootstrapped: bool | None = None
            if evaluator is not None:
                evaluation_result, promoted, bootstrapped = _evaluate_and_maybe_promote(
                    evaluator=evaluator,
                    candidate_model=trainer.model,
                    cycle_index=cycle_index,
                    device=args.device,
                )
                _print_evaluation_metrics(
                    cycle_index=cycle_index,
                    result=evaluation_result,
                    promoted=bool(promoted),
                    bootstrapped=bool(bootstrapped),
                    best_model_path=evaluator.config.best_model_path,
                )

            if metrics_log_path is not None:
                entry = build_training_log_entry(
                    cycle_index,
                    metrics,
                    trainer.last_collected_games,
                    evaluation_result=evaluation_result,
                    promoted=promoted,
                    bootstrapped=bootstrapped,
                )
                append_training_log_entry(metrics_log_path, entry)
                if dashboard_path is not None:
                    write_training_dashboard_from_log(metrics_log_path, dashboard_path)
    except KeyboardInterrupt:
        interrupted = True
        print("\nTraining interrupted by user.")
        if trainer.last_checkpoint_path is not None:
            print(f"last_checkpoint_path={trainer.last_checkpoint_path}")

    return RunSummary(
        cycles_requested=args.cycles,
        cycles_completed=cycles_completed,
        final_checkpoint_path=final_checkpoint_path,
        best_model_path=args.best_model_path,
        metrics_log_path=None if metrics_log_path is None else str(metrics_log_path),
        dashboard_path=None if dashboard_path is None else str(dashboard_path),
        eval_enabled=bool(evaluator is not None),
        interrupted=interrupted,
    )


def _print_run_configuration(
    args: argparse.Namespace,
    *,
    metrics_log_path: Path,
    dashboard_path: Path,
) -> None:
    """Print the high-level run configuration before training starts."""

    print("=== AlphaZero Self-Train Configuration ===")
    print(f"cycles={args.cycles}")
    print(f"device={args.device}")
    print(f"games_per_cycle={args.games_per_cycle}")
    print(f"buffer_samples={args.buffer_samples}")
    print(f"batch_size={args.batch_size}")
    print(f"epochs_per_cycle={args.epochs_per_cycle}")
    print(f"learning_rate={args.learning_rate}")
    print(f"weight_decay={args.weight_decay}")
    print(f"simulations={args.simulations}")
    print(f"checkpoint_dir={args.checkpoint_dir}")
    print(f"with_eval={args.with_eval}")
    print(f"eval_games={args.eval_games}")
    print(f"promotion_threshold={args.promotion_threshold}")
    print(f"draw_score={args.draw_score}")
    print(f"best_model_path={args.best_model_path}")
    print(f"metrics_log_path={metrics_log_path}")
    print(f"dashboard_path={dashboard_path}")
    print(f"append_metrics_log={args.append_metrics_log}")
    print(f"seed={args.seed}")
    print(f"smoke_test={args.smoke_test}")


def _print_run_summary(summary: RunSummary) -> None:
    """Print the final summary required by the CLI contract."""

    print("\n=== AlphaZero Self-Train Summary ===")
    print(f"total_cycles_run={summary.cycles_completed}")
    print(f"final_checkpoint_path={summary.final_checkpoint_path}")
    print(f"best_model_path={summary.best_model_path}")
    print(f"metrics_log_path={summary.metrics_log_path}")
    print(f"dashboard_path={summary.dashboard_path}")
    print(f"eval_enabled={summary.eval_enabled}")
    print(f"interrupted={summary.interrupted}")


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for AlphaZero-only Gomoku self-play training."""

    args = apply_smoke_test_overrides(parse_args(argv))
    args.device = _validate_device(args.device)
    metrics_log_path = _resolve_metrics_log_path(args)
    dashboard_path = _resolve_dashboard_path(args)
    _prepare_metrics_outputs(metrics_log_path, append=args.append_metrics_log)
    set_global_seed(args.seed)

    trainer = build_trainer_from_args(args)
    evaluator = build_evaluator_from_args(args) if args.with_eval else None

    _print_run_configuration(args, metrics_log_path=metrics_log_path, dashboard_path=dashboard_path)
    summary = run_training_loop(
        trainer,
        args,
        evaluator=evaluator,
        metrics_log_path=metrics_log_path,
        dashboard_path=dashboard_path,
    )
    _print_run_summary(summary)


if __name__ == "__main__":
    main()
