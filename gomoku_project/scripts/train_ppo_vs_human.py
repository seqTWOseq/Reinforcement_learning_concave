from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gomoku_project.arena.training_match import MatchParticipant, run_training_match
from gomoku_project.envs.gomoku_env import GomokuEnv
from gomoku_project.players.human_player import HumanPlayer
from gomoku_project.rl.ppo_trainer import PPOTrainer
from gomoku_project.ui.tkinter_renderer import TkinterRenderer
from gomoku_project.utils.checkpoint_utils import load_checkpoint_or_maybe_warn


def _format_optional(value: float | None) -> str:
    return "None" if value is None else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play Human vs PPO and update PPO from the PPO turns only.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", type=str, default="gomoku_project/models/ppo/ppo_checkpoint.pt")
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--reset-model", action="store_true")
    parser.add_argument("--human-color", choices=("black", "white"), default="black")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--deterministic-ppo", action="store_true")
    parser.add_argument("--move-delay", type=float, default=0.1)
    parser.add_argument("--post-game-delay", type=float, default=3.0)
    args = parser.parse_args()

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
    )
    checkpoint_path, _ = load_checkpoint_or_maybe_warn(
        trainer=trainer,
        path=args.model_path,
        model_label="PPO",
        cli_flag="model-path",
        allow_random_init=args.allow_random_init,
        reset_model=args.reset_model,
    )

    renderer = TkinterRenderer(title="Train PPO vs Human")
    human_player = HumanPlayer(renderer=renderer, name="Human")
    ppo_player = trainer.build_player(deterministic=args.deterministic_ppo, name="PPO")

    if args.human_color == "black":
        black_participant = MatchParticipant(player=human_player)
        white_participant = MatchParticipant(player=ppo_player, trainer=trainer, algorithm="ppo")
    else:
        black_participant = MatchParticipant(player=ppo_player, trainer=trainer, algorithm="ppo")
        white_participant = MatchParticipant(player=human_player)

    result = run_training_match(
        black_participant=black_participant,
        white_participant=white_participant,
        matchup="ppo_vs_human",
        env=GomokuEnv(),
        render=True,
        renderer=renderer,
        move_delay=args.move_delay,
        post_game_delay=args.post_game_delay,
        train_after_game=True,
    )
    trainer.save(checkpoint_path)

    metrics = next(
        (summary.update_metrics for summary in result.participant_summaries if summary.algorithm == "ppo"),
        None,
    ) or {}
    print(
        f"matchup={result.matchup} "
        f"black={result.black_name} "
        f"white={result.white_name} "
        f"winner={result.winner_name} "
        f"reason={result.reason} "
        f"moves={result.move_count}"
    )
    print(
        f"ppo_steps={result.ppo_steps_collected} "
        f"alphazero_examples={result.alphazero_examples_collected}"
    )
    print(
        f"ppo_update={bool(metrics)} "
        f"ppo_total_loss={_format_optional(metrics.get('total_loss'))} "
        f"ppo_policy_loss={_format_optional(metrics.get('policy_loss'))} "
        f"ppo_value_loss={_format_optional(metrics.get('value_loss'))} "
        f"checkpoint_saved=True "
        f"checkpoint={checkpoint_path}"
    )


if __name__ == "__main__":
    main()
