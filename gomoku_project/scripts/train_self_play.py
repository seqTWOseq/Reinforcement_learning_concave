from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gomoku_project.rl.trainer import AlphaZeroTrainer


def _mean_metric(metrics: list[dict[str, float]], key: str) -> float | None:
    if not metrics:
        return None
    return sum(item[key] for item in metrics) / len(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AlphaZero-style self-play training.")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games-per-iteration", type=int, default=4)
    parser.add_argument("--train-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mcts-simulations", type=int, default=32)
    parser.add_argument("--save-path", type=str, default="gomoku_project/models/alphazero_checkpoint.pt")
    parser.add_argument("--reset-model", action="store_true")
    args = parser.parse_args()

    trainer = AlphaZeroTrainer(
        batch_size=args.batch_size,
        mcts_simulations=args.mcts_simulations,
    )
    save_path = Path(args.save_path)

    loaded = False
    if not args.reset_model:
        loaded = trainer.load_if_exists(save_path)
    print(f"loaded_existing_model={loaded}")

    for iteration in range(1, args.iterations + 1):
        buffer_size = trainer.collect_self_play_games(args.games_per_iteration)
        metrics = trainer.train_steps(args.train_steps)
        print(
            f"iteration={iteration} "
            f"buffer_size={buffer_size} "
            f"total_loss={_mean_metric(metrics, 'total_loss')} "
            f"policy_loss={_mean_metric(metrics, 'policy_loss')} "
            f"value_loss={_mean_metric(metrics, 'value_loss')}"
        )

    trainer.save(save_path)
    print(f"saved_model={save_path}")


if __name__ == "__main__":
    main()
