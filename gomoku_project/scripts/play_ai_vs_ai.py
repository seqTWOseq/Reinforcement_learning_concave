from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gomoku_project.arena.match_runner import run_match
from gomoku_project.arena.training_match import MatchParticipant, run_training_match
from gomoku_project.envs.gomoku_env import GomokuEnv
from gomoku_project.players.heuristic_player import HeuristicPlayer
from gomoku_project.rl.ppo_trainer import PPOTrainer
from gomoku_project.rl.trainer import AlphaZeroTrainer
from gomoku_project.ui.tkinter_renderer import TkinterRenderer
from gomoku_project.utils.checkpoint_utils import load_checkpoint_or_maybe_warn


def _format_optional(value: float | None) -> str:
    return "None" if value is None else f"{value:.4f}"


def _build_player(kind: str, name: str, args: argparse.Namespace):
    if kind == "heuristic":
        return HeuristicPlayer(name=name), None, None, None
    if kind == "ppo":
        trainer = PPOTrainer(
            device=args.device,
            update_epochs=args.ppo_update_epochs,
            minibatch_size=args.ppo_minibatch_size,
        )
        checkpoint_path, _ = load_checkpoint_or_maybe_warn(
            trainer=trainer,
            path=args.ppo_model_path,
            model_label=name,
            cli_flag="ppo-model-path",
            allow_random_init=args.allow_random_init,
        )
        return (
            trainer.build_player(deterministic=args.deterministic_ai, name=name),
            trainer,
            "ppo",
            checkpoint_path,
        )

    trainer = AlphaZeroTrainer(
        batch_size=args.batch_size,
        mcts_simulations=args.mcts_simulations,
        device=args.device,
    )
    checkpoint_path, _ = load_checkpoint_or_maybe_warn(
        trainer=trainer,
        path=args.alphazero_model_path,
        model_label=name,
        cli_flag="alphazero-model-path",
        allow_random_init=args.allow_random_init,
    )
    return (
        trainer.build_player(
            deterministic=args.deterministic_ai,
            name=name,
            use_root_noise=not args.disable_root_noise and not args.deterministic_ai,
        ),
        trainer,
        "alphazero",
        checkpoint_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play Gomoku: AI vs AI. Training is disabled unless --train-after-game is set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--black", choices=("heuristic", "ppo", "alphazero"), default="heuristic")
    parser.add_argument("--white", choices=("heuristic", "ppo", "alphazero"), default="alphazero")
    parser.add_argument(
        "--alphazero-model-path",
        type=str,
        default="gomoku_project/models/alphazero_checkpoint.pt",
        help="AlphaZero checkpoint path for any alphazero side.",
    )
    parser.add_argument(
        "--ppo-model-path",
        type=str,
        default="gomoku_project/models/ppo/ppo_checkpoint.pt",
        help="PPO checkpoint path for any ppo side.",
    )
    parser.add_argument(
        "--allow-random-init",
        action="store_true",
        help="Allow PPO/AlphaZero players to start from random initialization if their checkpoint is missing or invalid.",
    )
    parser.add_argument("--train-after-game", action="store_true", help="Store AI-only match data and update learned participants after the game.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mcts-simulations", type=int, default=24)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--move-delay", type=float, default=0.1)
    parser.add_argument("--post-game-delay", type=float, default=1.0)
    parser.add_argument("--alphazero-train-steps", type=int, default=8)
    parser.add_argument("--ppo-update-epochs", type=int, default=4)
    parser.add_argument("--ppo-minibatch-size", type=int, default=128)
    parser.add_argument("--deterministic-ai", action="store_true")
    parser.add_argument("--disable-root-noise", action="store_true")
    args = parser.parse_args()

    black_player, black_trainer, black_algorithm, black_checkpoint = _build_player(args.black, f"{args.black.title()}_Black", args)
    white_player, white_trainer, white_algorithm, white_checkpoint = _build_player(args.white, f"{args.white.title()}_White", args)

    if args.train_after_game and black_trainer is None and white_trainer is None:
        parser.error("--train-after-game requires at least one PPO or AlphaZero participant.")

    if args.train_after_game:
        result = run_training_match(
            black_participant=MatchParticipant(player=black_player, trainer=black_trainer, algorithm=black_algorithm),
            white_participant=MatchParticipant(player=white_player, trainer=white_trainer, algorithm=white_algorithm),
            matchup=f"{args.black}_vs_{args.white}",
            env=GomokuEnv(),
            render=args.render,
            renderer=TkinterRenderer(title="Gomoku AI vs AI") if args.render else None,
            move_delay=args.move_delay,
            post_game_delay=args.post_game_delay if args.render else 0.0,
            train_after_game=True,
            alphazero_train_steps=args.alphazero_train_steps,
        )

        saved_trainers: set[int] = set()
        for trainer, checkpoint_path in ((black_trainer, black_checkpoint), (white_trainer, white_checkpoint)):
            if trainer is None or checkpoint_path is None or id(trainer) in saved_trainers:
                continue
            trainer.save(checkpoint_path)
            saved_trainers.add(id(trainer))

        print(
            f"mode=training "
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
        for summary in result.participant_summaries:
            if summary.algorithm == "ppo":
                print(
                    f"{summary.player_name}_update={summary.update_performed} "
                    f"ppo_total_loss={_format_optional((summary.update_metrics or {}).get('total_loss'))} "
                    f"ppo_policy_loss={_format_optional((summary.update_metrics or {}).get('policy_loss'))} "
                    f"ppo_value_loss={_format_optional((summary.update_metrics or {}).get('value_loss'))}"
                )
            else:
                print(
                    f"{summary.player_name}_update={summary.update_performed} "
                    f"alphazero_total_loss={_format_optional((summary.update_metrics or {}).get('total_loss'))} "
                    f"alphazero_policy_loss={_format_optional((summary.update_metrics or {}).get('policy_loss'))} "
                    f"alphazero_value_loss={_format_optional((summary.update_metrics or {}).get('value_loss'))}"
                )
        print(
            f"checkpoint_saved={bool(saved_trainers)} "
            f"black_checkpoint={black_checkpoint} "
            f"white_checkpoint={white_checkpoint}"
        )
        return

    renderer = TkinterRenderer(title="Gomoku AI vs AI") if args.render else None
    result = run_match(
        black_player=black_player,
        white_player=white_player,
        env=GomokuEnv(),
        render=args.render,
        renderer=renderer,
        move_delay=args.move_delay,
        post_game_delay=args.post_game_delay if args.render else 0.0,
    )
    print(
        f"mode=play_only "
        f"matchup={args.black}_vs_{args.white} "
        f"winner={result.winner_name} "
        f"reason={result.reason} "
        f"moves={result.move_count}"
    )
    print("ppo_steps=0 alphazero_examples=0")
    print("update_performed=False checkpoint_saved=False")


if __name__ == "__main__":
    main()
