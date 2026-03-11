from __future__ import annotations

import argparse
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gomoku_project.rl.ppo_trainer import PPOTrainer
from gomoku_project.utils.training_logger import append_csv_record, append_jsonl_record


def _resolve_logging_targets(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    log_dir = Path(args.log_dir)
    if not args.log_csv and not args.log_jsonl:
        args.log_csv = True
        args.log_jsonl = True

    csv_path = log_dir / "ppo_training_metrics.csv" if args.log_csv else None
    jsonl_path = log_dir / "ppo_training_metrics.jsonl" if args.log_jsonl else None
    return csv_path, jsonl_path


def _format_optional(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_recent_rates(values: deque[float]) -> str:
    if not values:
        return "n/a"
    return ", ".join(f"{value:.2f}" for value in values)


def _build_record(
    *,
    iteration: int,
    total_iterations: int,
    collection: dict[str, Any],
    metrics: dict[str, Any] | None,
    evaluation: dict[str, Any] | None,
    checkpoint_path: Path,
    best_updated: bool,
    best_metric_name: str | None,
    best_metric_value: float | None,
) -> dict[str, Any]:
    metrics = metrics or {}
    evaluation = evaluation or {}
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "iteration": iteration,
        "total_iterations": total_iterations,
        "games": collection["games"],
        "self_games": collection["self_games"],
        "heuristic_games": collection["heuristic_games"],
        "moves": collection["moves"],
        "avg_moves": collection["avg_moves"],
        "self_avg_moves": collection["self_avg_moves"],
        "heuristic_avg_moves": collection["heuristic_avg_moves"],
        "black_wins": collection["black_wins"],
        "white_wins": collection["white_wins"],
        "draws": collection["draws"],
        "ppo_black_episodes": collection["ppo_black_episodes"],
        "ppo_white_episodes": collection["ppo_white_episodes"],
        "ppo_wins": collection["ppo_wins"],
        "ppo_losses": collection["ppo_losses"],
        "ppo_draws": collection["ppo_draws"],
        "ppo_win_rate": collection["ppo_win_rate"],
        "ppo_loss_rate": collection["ppo_loss_rate"],
        "ppo_draw_rate": collection["ppo_draw_rate"],
        "self_ppo_wins": collection["self_ppo_wins"],
        "self_ppo_losses": collection["self_ppo_losses"],
        "self_ppo_draws": collection["self_ppo_draws"],
        "self_ppo_win_rate": collection["self_ppo_win_rate"],
        "heuristic_ppo_wins": collection["heuristic_ppo_wins"],
        "heuristic_ppo_losses": collection["heuristic_ppo_losses"],
        "heuristic_ppo_draws": collection["heuristic_ppo_draws"],
        "heuristic_ppo_win_rate": collection["heuristic_ppo_win_rate"],
        "invalid_move_games": collection["invalid_move_games"],
        "self_invalid_move_games": collection["self_invalid_move_games"],
        "heuristic_invalid_move_games": collection["heuristic_invalid_move_games"],
        "ppo_invalid_moves": collection["ppo_invalid_moves"],
        "opponent_invalid_moves": collection["opponent_invalid_moves"],
        "invalid_move_rate": collection["invalid_move_rate"],
        "action_entropy_mean": collection["action_entropy_mean"],
        "selected_action_prob_mean": collection["selected_action_prob_mean"],
        "top1_action_prob_mean": collection["top1_action_prob_mean"],
        "top3_action_prob_sum_mean": collection["top3_action_prob_sum_mean"],
        "valid_action_count_mean": collection["valid_action_count_mean"],
        "center_near_ratio": collection["center_near_ratio"],
        "opening_center_near_ratio": collection["opening_center_near_ratio"],
        "immediate_win_available_count": collection["immediate_win_available_count"],
        "immediate_win_chosen_count": collection["immediate_win_chosen_count"],
        "immediate_win_pick_rate": collection["immediate_win_pick_rate"],
        "immediate_block_available_count": collection["immediate_block_available_count"],
        "immediate_block_chosen_count": collection["immediate_block_chosen_count"],
        "immediate_block_pick_rate": collection["immediate_block_pick_rate"],
        "open_three_create_rate": collection["open_three_create_rate"],
        "strong_four_create_rate": collection["strong_four_create_rate"],
        "total_loss": metrics.get("total_loss"),
        "policy_loss": metrics.get("policy_loss"),
        "value_loss": metrics.get("value_loss"),
        "entropy": metrics.get("entropy"),
        "approx_kl": metrics.get("approx_kl"),
        "clip_fraction": metrics.get("clip_fraction"),
        "advantage_mean": metrics.get("advantage_mean"),
        "advantage_std": metrics.get("advantage_std"),
        "return_mean": metrics.get("return_mean"),
        "return_std": metrics.get("return_std"),
        "value_prediction_mean": metrics.get("value_prediction_mean"),
        "value_prediction_std": metrics.get("value_prediction_std"),
        "num_samples": metrics.get("num_samples"),
        "eval_games": evaluation.get("games"),
        "eval_ppo_wins": evaluation.get("ppo_wins"),
        "eval_ppo_losses": evaluation.get("ppo_losses"),
        "eval_ppo_draws": evaluation.get("ppo_draws"),
        "eval_ppo_win_rate": evaluation.get("ppo_win_rate"),
        "eval_invalid_move_games": evaluation.get("invalid_move_games"),
        "checkpoint_saved": True,
        "checkpoint_path": str(checkpoint_path),
        "best_metric_name": best_metric_name or "",
        "best_metric_value": best_metric_value,
        "best_metric_updated": best_updated,
    }


def _print_iteration_report(
    *,
    iteration: int,
    total_iterations: int,
    collection: dict[str, Any],
    metrics: dict[str, Any] | None,
    evaluation: dict[str, Any] | None,
    save_path: Path,
    best_updated: bool,
    best_metric_name: str | None,
    best_metric_value: float | None,
    recent_eval_rates: deque[float],
    verbose_episodes: bool,
) -> None:
    metrics = metrics or {}
    print(f"iteration {iteration}/{total_iterations}")
    print(
        "  games: "
        f"total={collection['games']} self={collection['self_games']} heuristic={collection['heuristic_games']} "
        f"avg_moves={collection['avg_moves']:.2f} "
        f"self_avg={collection['self_avg_moves']:.2f} heuristic_avg={collection['heuristic_avg_moves']:.2f}"
    )
    print(
        "  colors: "
        f"ppo_black={collection['ppo_black_episodes']} "
        f"ppo_white={collection['ppo_white_episodes']}"
    )
    print(
        "  ppo_result: "
        f"win={collection['ppo_wins']} lose={collection['ppo_losses']} draw={collection['ppo_draws']} "
        f"win_rate={collection['ppo_win_rate']:.3f}"
    )
    print(
        "  matchup_result: "
        f"self=win:{collection['self_ppo_wins']}/lose:{collection['self_ppo_losses']}/draw:{collection['self_ppo_draws']} "
        f"heuristic=win:{collection['heuristic_ppo_wins']}/lose:{collection['heuristic_ppo_losses']}/draw:{collection['heuristic_ppo_draws']}"
    )
    print(
        "  invalid_moves: "
        f"games={collection['invalid_move_games']} "
        f"self={collection['self_invalid_move_games']} heuristic={collection['heuristic_invalid_move_games']} "
        f"ppo={collection['ppo_invalid_moves']} opponent={collection['opponent_invalid_moves']}"
    )
    print(
        "  policy: "
        f"action_entropy={collection['action_entropy_mean']:.4f} "
        f"selected_p={collection['selected_action_prob_mean']:.4f} "
        f"top1_p={collection['top1_action_prob_mean']:.4f} "
        f"top3_sum={collection['top3_action_prob_sum_mean']:.4f} "
        f"valid_actions={collection['valid_action_count_mean']:.2f}"
    )
    print(
        "  move_quality: "
        f"opening_center={collection['opening_center_near_ratio']:.3f} "
        f"center={collection['center_near_ratio']:.3f} "
        f"immediate_win={collection['immediate_win_pick_rate']:.3f} "
        f"({collection['immediate_win_chosen_count']}/{collection['immediate_win_available_count']}) "
        f"immediate_block={collection['immediate_block_pick_rate']:.3f} "
        f"({collection['immediate_block_chosen_count']}/{collection['immediate_block_available_count']}) "
        f"open_three={collection['open_three_create_rate']:.3f}"
    )
    print(
        "  update: "
        f"total={_format_optional(metrics.get('total_loss'))} "
        f"policy={_format_optional(metrics.get('policy_loss'))} "
        f"value={_format_optional(metrics.get('value_loss'))} "
        f"entropy={_format_optional(metrics.get('entropy'))} "
        f"approx_kl={_format_optional(metrics.get('approx_kl'))} "
        f"clip_frac={_format_optional(metrics.get('clip_fraction'))}"
    )
    print(
        "  returns: "
        f"adv_mean={_format_optional(metrics.get('advantage_mean'))} "
        f"adv_std={_format_optional(metrics.get('advantage_std'))} "
        f"ret_mean={_format_optional(metrics.get('return_mean'))} "
        f"ret_std={_format_optional(metrics.get('return_std'))} "
        f"value_mean={_format_optional(metrics.get('value_prediction_mean'))} "
        f"value_std={_format_optional(metrics.get('value_prediction_std'))}"
    )
    if evaluation is not None:
        print(
            "  eval_vs_heuristic: "
            f"games={evaluation['games']} "
            f"win={evaluation['ppo_wins']} lose={evaluation['ppo_losses']} draw={evaluation['ppo_draws']} "
            f"win_rate={evaluation['ppo_win_rate']:.3f} "
            f"recent={_format_recent_rates(recent_eval_rates)}"
        )
    print(
        "  checkpoint: "
        f"saved={save_path} "
        f"best_updated={best_updated} "
        f"best_metric={best_metric_name or 'none'} "
        f"best_value={_format_optional(best_metric_value)}"
    )
    if verbose_episodes:
        for episode_index, episode_log in enumerate(collection["episode_logs"], start=1):
            print(f"    episode {episode_index}: {episode_log}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PPO self-play training for Gomoku.")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games-per-iteration", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--save-path", type=str, default="gomoku_project/models/ppo/ppo_checkpoint.pt")
    parser.add_argument("--reset-model", action="store_true")
    parser.add_argument("--use-heuristic-pool", action="store_true", help="Sample HeuristicPlayer as an opponent during PPO data collection.")
    parser.add_argument("--heuristic-prob", type=float, default=0.3, help="Probability of PPO vs Heuristic episodes when heuristic pool is enabled.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for opponent sampling and heuristic tie-breaks.")
    parser.add_argument("--log-dir", type=str, default="gomoku_project/logs")
    parser.add_argument("--log-csv", action="store_true", help="Append iteration metrics to CSV.")
    parser.add_argument("--log-jsonl", action="store_true", help="Append iteration metrics to JSONL.")
    parser.add_argument("--eval-every", type=int, default=0, help="Run deterministic PPO vs Heuristic evaluation every N iterations. 0 disables evaluation.")
    parser.add_argument("--eval-games", type=int, default=10)
    parser.add_argument("--deterministic-eval", action="store_true", help="Use deterministic PPO actions during periodic evaluation.")
    parser.add_argument("--verbose-episodes", action="store_true", help="Print per-episode collection logs.")
    parser.add_argument(
        "--save-best-on",
        type=str,
        choices=("heuristic_win_rate", "ppo_win_rate"),
        default=None,
        help="Also save a *_best checkpoint when the selected metric improves.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.heuristic_prob <= 1.0:
        parser.error("--heuristic-prob must be between 0.0 and 1.0.")
    if args.eval_every < 0:
        parser.error("--eval-every must be >= 0.")
    if args.eval_games <= 0:
        parser.error("--eval-games must be > 0.")

    opponent_mode = "self_play_with_heuristic_pool" if args.use_heuristic_pool else "self_play_only"
    opponent_probs = None
    if args.use_heuristic_pool:
        opponent_probs = {
            "self": 1.0 - args.heuristic_prob,
            "heuristic": args.heuristic_prob,
        }

    trainer = PPOTrainer(
        device=args.device,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        opponent_mode=opponent_mode,
        opponent_probs=opponent_probs,
        seed=args.seed,
    )
    save_path = Path(args.save_path)
    best_path = save_path.with_name(f"{save_path.stem}_best{save_path.suffix}")
    csv_path, jsonl_path = _resolve_logging_targets(args)

    loaded = False
    if not args.reset_model:
        loaded = trainer.load_if_exists(save_path)
    print(f"loaded_existing_model={loaded}")

    best_metric_value: float | None = None
    recent_eval_rates: deque[float] = deque(maxlen=10)

    for iteration in range(1, args.iterations + 1):
        collection = trainer.collect_self_play_games(args.games_per_iteration)
        metrics = trainer.update()

        evaluation = None
        if args.eval_every > 0 and iteration % args.eval_every == 0:
            evaluation = trainer.evaluate_against_heuristic(
                args.eval_games,
                deterministic=args.deterministic_eval,
            )
            recent_eval_rates.append(float(evaluation["ppo_win_rate"]))

        trainer.save(save_path)

        best_updated = False
        tracked_metric_name = args.save_best_on
        tracked_metric_value: float | None = None
        if tracked_metric_name == "heuristic_win_rate" and evaluation is not None:
            tracked_metric_value = float(evaluation["ppo_win_rate"])
        elif tracked_metric_name == "ppo_win_rate":
            tracked_metric_value = float(collection["ppo_win_rate"])

        if tracked_metric_value is not None and (best_metric_value is None or tracked_metric_value > best_metric_value):
            best_metric_value = tracked_metric_value
            trainer.save(best_path)
            best_updated = True

        _print_iteration_report(
            iteration=iteration,
            total_iterations=args.iterations,
            collection=collection,
            metrics=metrics,
            evaluation=evaluation,
            save_path=save_path,
            best_updated=best_updated,
            best_metric_name=tracked_metric_name,
            best_metric_value=best_metric_value,
            recent_eval_rates=recent_eval_rates,
            verbose_episodes=args.verbose_episodes,
        )

        record = _build_record(
            iteration=iteration,
            total_iterations=args.iterations,
            collection=collection,
            metrics=metrics,
            evaluation=evaluation,
            checkpoint_path=save_path,
            best_updated=best_updated,
            best_metric_name=tracked_metric_name,
            best_metric_value=best_metric_value,
        )
        if csv_path is not None:
            append_csv_record(csv_path, record)
        if jsonl_path is not None:
            append_jsonl_record(jsonl_path, record)

    print(f"saved_model={save_path}")


if __name__ == "__main__":
    main()
