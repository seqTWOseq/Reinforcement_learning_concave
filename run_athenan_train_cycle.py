"""End-to-end Athenan training-cycle runner.

One cycle does:
1. self-play N games
2. replay-buffer accumulation
3. M training epochs
4. random-baseline evaluation
5. train-search vs inference-search comparison
6. best-checkpoint update
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from gomoku_ai.athenan.eval import AthenanEvaluator, RandomLegalAgent
from gomoku_ai.athenan.network import AthenanValueNet, load_athenan_value_net
from gomoku_ai.athenan.replay import AthenanReplayBuffer
from gomoku_ai.athenan.search import AthenanSearcher
from gomoku_ai.athenan.trainer import AthenanSelfPlayRunner, AthenanTrainer


@dataclass(frozen=True)
class RunSummary:
    """Summary of one CLI-driven Athenan run."""

    cycles_requested: int
    cycles_completed: int
    final_cycle_checkpoint_path: str | None
    best_checkpoint_path: str
    metrics_log_path: str
    replay_buffer_size: int


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer.")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer.")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be a positive float.")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be a non-negative float.")
    return parsed


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run integrated Athenan self-play + train + eval cycles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cycles", type=_positive_int, default=10, help="Number of full train cycles.")
    parser.add_argument("--self-play-games", type=_positive_int, default=10, help="Self-play games per cycle.")
    parser.add_argument("--train-epochs", type=_positive_int, default=2, help="Training epochs per cycle.")
    parser.add_argument("--batch-size", type=_positive_int, default=64, help="Training batch size.")
    parser.add_argument("--replay-max-size", type=_positive_int, default=50_000, help="Replay buffer max size.")
    parser.add_argument("--aux-search-weight", type=_non_negative_float, default=0.1, help="Aux search-loss weight.")
    parser.add_argument("--learning-rate", type=_positive_float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=_non_negative_float, default=1e-4, help="Adam weight decay.")
    parser.add_argument("--train-search-depth", type=_positive_int, default=2, help="Training search depth.")
    parser.add_argument("--inference-search-depth", type=_positive_int, default=4, help="Inference search depth.")
    parser.add_argument("--candidate-limit", type=_positive_int, default=64, help="Search candidate limit.")
    parser.add_argument("--candidate-radius", type=_non_negative_int, default=2, help="Search candidate radius.")
    parser.add_argument(
        "--opening-random-steps",
        type=_non_negative_int,
        default=2,
        help="Opening random moves in self-play.",
    )
    parser.add_argument("--eval-games", type=_positive_int, default=10, help="Games used in each evaluation call.")
    parser.add_argument(
        "--inference-time-budget-sec",
        type=_non_negative_float,
        default=0.0,
        help="Inference time budget per search. 0 disables time budget.",
    )
    parser.add_argument(
        "--disable-inference-iterative",
        action="store_true",
        help="Disable inference iterative deepening.",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/athenan", help="Checkpoint directory.")
    parser.add_argument(
        "--best-checkpoint-path",
        type=str,
        default="checkpoints/athenan/best_model.pt",
        help="Best-model checkpoint path.",
    )
    parser.add_argument(
        "--metrics-log-path",
        type=str,
        default="checkpoints/athenan/train_metrics.jsonl",
        help="JSONL path for per-cycle metrics.",
    )
    parser.add_argument(
        "--append-metrics-log",
        action="store_true",
        help="Append to existing metrics log (default: overwrite).",
    )
    parser.add_argument(
        "--init-checkpoint-path",
        type=str,
        default="",
        help="Optional initial Athenan checkpoint to load before cycle 0.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device string.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Clamp expensive settings for a fast end-to-end check.",
    )
    return parser.parse_args(argv)


def _apply_smoke_overrides(args: argparse.Namespace) -> argparse.Namespace:
    if not args.smoke_test:
        return args
    args.cycles = min(args.cycles, 2)
    args.self_play_games = min(args.self_play_games, 2)
    args.train_epochs = min(args.train_epochs, 1)
    args.batch_size = min(args.batch_size, 16)
    args.eval_games = min(args.eval_games, 2)
    args.train_search_depth = min(args.train_search_depth, 1)
    args.inference_search_depth = min(args.inference_search_depth, 2)
    args.candidate_limit = min(args.candidate_limit, 12)
    return args


def _validate_device(device: str) -> str:
    try:
        resolved = torch.device(device)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid torch device string: {device!r}.") from exc
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but torch.cuda.is_available() is False.")
    return str(resolved)


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _prepare_outputs(metrics_log_path: Path, *, append: bool) -> None:
    metrics_log_path.parent.mkdir(parents=True, exist_ok=True)
    if not append and metrics_log_path.exists():
        metrics_log_path.unlink()


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=True, sort_keys=False) + "\n")


def _summarize_self_play(game_summaries: list[object]) -> dict[str, float | int]:
    if not game_summaries:
        return {
            "games": 0,
            "total_moves": 0,
            "avg_move_count": 0.0,
            "num_samples": 0,
            "forced_tactical_count": 0,
        }
    total_moves = sum(int(summary.move_count) for summary in game_summaries)
    total_samples = sum(int(summary.num_samples) for summary in game_summaries)
    total_forced = sum(int(summary.forced_tactical_count) for summary in game_summaries)
    return {
        "games": len(game_summaries),
        "total_moves": total_moves,
        "avg_move_count": float(total_moves / len(game_summaries)),
        "num_samples": total_samples,
        "forced_tactical_count": total_forced,
    }


def _bootstrap_best_win_rate(
    evaluator: AthenanEvaluator,
    *,
    best_checkpoint_path: Path,
    eval_games: int,
    inference_search_depth: int,
    candidate_limit: int,
    candidate_radius: int,
    iterative_deepening: bool,
    time_budget_sec: float | None,
    device: str,
    seed: int,
) -> dict[str, object] | None:
    if not best_checkpoint_path.exists():
        return None
    best_model = load_athenan_value_net(best_checkpoint_path, device=device)
    summary = evaluator.evaluate_inference_search_vs_random(
        model=best_model,
        num_games=eval_games,
        random_seed=seed,
        searcher_max_depth=inference_search_depth,
        searcher_candidate_limit=candidate_limit,
        searcher_candidate_radius=candidate_radius,
        iterative_deepening=iterative_deepening,
        time_budget_sec=time_budget_sec,
        device=device,
    )
    evaluator.best_win_rate = float(summary.win_rate)
    return asdict(summary)


def _build_train_searcher(args: argparse.Namespace, model: AthenanValueNet) -> AthenanSearcher:
    return AthenanSearcher(
        model=model,
        max_depth=args.train_search_depth,
        candidate_limit=args.candidate_limit,
        candidate_radius=args.candidate_radius,
        use_alpha_beta=True,
        device=args.device,
    )


def _print_cycle_overview(
    cycle_index: int,
    *,
    self_play_metrics: dict[str, float | int],
    train_metrics: dict[str, float | int],
    random_eval: dict[str, object],
    comparison: dict[str, object],
    best_decision: dict[str, object],
    replay_buffer_size: int,
    cycle_checkpoint_path: Path,
) -> None:
    print(f"\n=== Athenan Cycle {cycle_index} ===")
    print(f"self_play_games={self_play_metrics['games']} total_moves={self_play_metrics['total_moves']}")
    print(f"replay_buffer_size={replay_buffer_size}")
    print(
        "train_metrics: "
        f"total_loss={float(train_metrics['total_loss']):.6f} "
        f"value_loss={float(train_metrics['value_loss']):.6f} "
        f"aux_search_loss={float(train_metrics['aux_search_loss']):.6f}"
    )
    print(
        "eval_random: "
        f"win_rate={float(random_eval['win_rate']):.4f} "
        f"score_rate={float(random_eval['score_rate']):.4f}"
    )
    train_summary = comparison["train_summary"]
    infer_summary = comparison["inference_summary"]
    print(
        "train_vs_inference: "
        f"train_win_rate={float(train_summary['win_rate']):.4f} "
        f"infer_win_rate={float(infer_summary['win_rate']):.4f}"
    )
    print(
        "best_update: "
        f"updated={best_decision['updated']} "
        f"best_win_rate={float(best_decision['best_win_rate']):.4f}"
    )
    print(f"cycle_checkpoint_path={cycle_checkpoint_path}")


def _run_cycles(args: argparse.Namespace) -> RunSummary:
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_log_path = Path(args.metrics_log_path)
    best_checkpoint_path = Path(args.best_checkpoint_path)
    _prepare_outputs(metrics_log_path, append=args.append_metrics_log)

    replay_buffer = AthenanReplayBuffer(max_size=args.replay_max_size)
    trainer = AthenanTrainer(
        model=AthenanValueNet(),
        replay_buffer=replay_buffer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        aux_search_weight=args.aux_search_weight,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )
    # Use trainer-owned reference to avoid divergence if constructor behavior changes.
    replay_buffer = trainer.replay_buffer
    if args.init_checkpoint_path.strip():
        trainer.load_model_checkpoint(args.init_checkpoint_path.strip())

    evaluator = AthenanEvaluator(best_checkpoint_path=best_checkpoint_path)
    inference_time_budget_sec = None if args.inference_time_budget_sec <= 0.0 else args.inference_time_budget_sec
    best_bootstrap = _bootstrap_best_win_rate(
        evaluator,
        best_checkpoint_path=best_checkpoint_path,
        eval_games=args.eval_games,
        inference_search_depth=args.inference_search_depth,
        candidate_limit=args.candidate_limit,
        candidate_radius=args.candidate_radius,
        iterative_deepening=not args.disable_inference_iterative,
        time_budget_sec=inference_time_budget_sec,
        device=args.device,
        seed=args.seed + 50_000,
    )
    if best_bootstrap is not None:
        _append_jsonl(
            metrics_log_path,
            {
                "event": "best_checkpoint_bootstrap",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "best_checkpoint_path": str(best_checkpoint_path),
                "summary": best_bootstrap,
            },
        )

    final_cycle_checkpoint_path: Path | None = None
    cycles_completed = 0
    for cycle_index in range(args.cycles):
        # 1) self-play N games
        self_play_runner = AthenanSelfPlayRunner(
            searcher=_build_train_searcher(args, trainer.model),
            replay_buffer=replay_buffer,
            opening_random_steps=args.opening_random_steps,
            seed=args.seed + cycle_index,
        )
        game_summaries = self_play_runner.play_games(args.self_play_games)
        self_play_metrics = _summarize_self_play(game_summaries)

        # 2) replay buffer is accumulated through shared `replay_buffer`.
        replay_buffer_size = len(replay_buffer)

        # 3) train for M epochs
        train_history = trainer.train_loop(
            num_epochs=args.train_epochs,
            batch_size=args.batch_size,
            shuffle=True,
        )
        train_metrics = dict(train_history[-1])

        cycle_checkpoint_path = checkpoint_dir / f"cycle_{cycle_index:04d}.pt"
        trainer.save_model_checkpoint(
            cycle_checkpoint_path,
            metadata={
                "cycle_index": cycle_index,
                "self_play_metrics": self_play_metrics,
                "train_metrics": train_metrics,
            },
        )

        # 4) evaluate against random baseline
        random_eval_summary = evaluator.evaluate_model_vs_random(
            trainer.model,
            num_games=args.eval_games,
            random_seed=args.seed + 1000 + cycle_index,
            searcher_max_depth=args.train_search_depth,
            searcher_candidate_limit=args.candidate_limit,
            searcher_candidate_radius=args.candidate_radius,
            searcher_use_alpha_beta=True,
            device=args.device,
        )
        random_eval = asdict(random_eval_summary)

        # 5) compare train-search vs inference-search
        comparison_summary = evaluator.evaluate_train_vs_inference(
            model=trainer.model,
            opponent_agent=RandomLegalAgent(seed=args.seed + 2000 + cycle_index),
            num_games=args.eval_games,
            train_search_kwargs={
                "max_depth": args.train_search_depth,
                "candidate_limit": args.candidate_limit,
                "candidate_radius": args.candidate_radius,
                "use_alpha_beta": True,
                "device": args.device,
            },
            inference_search_kwargs={
                "max_depth": args.inference_search_depth,
                "candidate_limit": args.candidate_limit,
                "candidate_radius": args.candidate_radius,
                "use_alpha_beta": True,
                "iterative_deepening": not args.disable_inference_iterative,
                "time_budget_sec": inference_time_budget_sec,
                "device": args.device,
            },
        )
        comparison = asdict(comparison_summary)

        # 6) update best checkpoint
        best_eval_summary, best_decision_obj = evaluator.evaluate_and_update_best_checkpoint(
            model=trainer.model,
            opponent_agent=RandomLegalAgent(seed=args.seed + 3000 + cycle_index),
            num_games=args.eval_games,
            candidate_checkpoint_path=cycle_checkpoint_path,
            use_inference_search=True,
            searcher_max_depth=args.inference_search_depth,
            searcher_candidate_limit=args.candidate_limit,
            searcher_candidate_radius=args.candidate_radius,
            searcher_use_alpha_beta=True,
            iterative_deepening=not args.disable_inference_iterative,
            time_budget_sec=inference_time_budget_sec,
            device=args.device,
        )
        best_eval = asdict(best_eval_summary)
        best_decision = asdict(best_decision_obj)

        cycle_payload = {
            "event": "cycle",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "cycle_index": cycle_index,
            "settings": {
                "self_play_games": args.self_play_games,
                "train_epochs": args.train_epochs,
                "batch_size": args.batch_size,
                "aux_search_weight": args.aux_search_weight,
                "train_search_depth": args.train_search_depth,
                "inference_search_depth": args.inference_search_depth,
                "candidate_limit": args.candidate_limit,
                "opening_random_steps": args.opening_random_steps,
            },
            "self_play": self_play_metrics,
            "replay": {
                "buffer_size": replay_buffer_size,
            },
            "train": train_metrics,
            "eval_random_baseline": random_eval,
            "eval_train_vs_inference": comparison,
            "best_checkpoint_eval": best_eval,
            "best_checkpoint_decision": best_decision,
            "paths": {
                "cycle_checkpoint_path": str(cycle_checkpoint_path),
                "best_checkpoint_path": str(best_checkpoint_path),
            },
        }
        _append_jsonl(metrics_log_path, cycle_payload)
        _print_cycle_overview(
            cycle_index,
            self_play_metrics=self_play_metrics,
            train_metrics=train_metrics,
            random_eval=random_eval,
            comparison=comparison,
            best_decision=best_decision,
            replay_buffer_size=replay_buffer_size,
            cycle_checkpoint_path=cycle_checkpoint_path,
        )

        final_cycle_checkpoint_path = cycle_checkpoint_path
        cycles_completed += 1

    return RunSummary(
        cycles_requested=args.cycles,
        cycles_completed=cycles_completed,
        final_cycle_checkpoint_path=(
            None if final_cycle_checkpoint_path is None else str(final_cycle_checkpoint_path)
        ),
        best_checkpoint_path=str(best_checkpoint_path),
        metrics_log_path=str(metrics_log_path),
        replay_buffer_size=len(replay_buffer),
    )


def _print_config(args: argparse.Namespace) -> None:
    print("=== Athenan Train Cycle Configuration ===")
    print(f"cycles={args.cycles}")
    print(f"self_play_games={args.self_play_games}")
    print(f"train_epochs={args.train_epochs}")
    print(f"batch_size={args.batch_size}")
    print(f"replay_max_size={args.replay_max_size}")
    print(f"aux_search_weight={args.aux_search_weight}")
    print(f"learning_rate={args.learning_rate}")
    print(f"weight_decay={args.weight_decay}")
    print(f"train_search_depth={args.train_search_depth}")
    print(f"inference_search_depth={args.inference_search_depth}")
    print(f"candidate_limit={args.candidate_limit}")
    print(f"candidate_radius={args.candidate_radius}")
    print(f"opening_random_steps={args.opening_random_steps}")
    print(f"eval_games={args.eval_games}")
    print(f"inference_time_budget_sec={args.inference_time_budget_sec}")
    print(f"disable_inference_iterative={args.disable_inference_iterative}")
    print(f"checkpoint_dir={args.checkpoint_dir}")
    print(f"best_checkpoint_path={args.best_checkpoint_path}")
    print(f"metrics_log_path={args.metrics_log_path}")
    print(f"append_metrics_log={args.append_metrics_log}")
    print(f"init_checkpoint_path={args.init_checkpoint_path}")
    print(f"device={args.device}")
    print(f"seed={args.seed}")
    print(f"smoke_test={args.smoke_test}")


def _print_run_summary(summary: RunSummary) -> None:
    print("\n=== Athenan Run Summary ===")
    print(f"cycles_requested={summary.cycles_requested}")
    print(f"cycles_completed={summary.cycles_completed}")
    print(f"final_cycle_checkpoint_path={summary.final_cycle_checkpoint_path}")
    print(f"best_checkpoint_path={summary.best_checkpoint_path}")
    print(f"metrics_log_path={summary.metrics_log_path}")
    print(f"replay_buffer_size={summary.replay_buffer_size}")


def main(argv: Sequence[str] | None = None) -> None:
    args = _apply_smoke_overrides(_parse_args(argv))
    args.device = _validate_device(args.device)
    _set_global_seed(args.seed)
    _print_config(args)
    summary = _run_cycles(args)
    _print_run_summary(summary)


if __name__ == "__main__":
    main()
