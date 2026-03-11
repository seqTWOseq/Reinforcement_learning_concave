from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gomoku_project.arena.match_runner import run_match
from gomoku_project.arena.training_match import MatchParticipant, run_training_match
from gomoku_project.core.constants import BLACK, WHITE
from gomoku_project.envs.gomoku_env import GomokuEnv
from gomoku_project.players.center_player import CenterPlayer
from gomoku_project.players.heuristic_player import HeuristicPlayer
from gomoku_project.players.human_player import HumanPlayer
from gomoku_project.rl.ppo_trainer import PPOTrainer
from gomoku_project.rl.trainer import AlphaZeroTrainer
from gomoku_project.ui.tkinter_renderer import TkinterRenderer
from gomoku_project.utils.checkpoint_utils import load_checkpoint_or_maybe_warn

DEFAULT_ALPHAZERO_MODEL_PATH = "gomoku_project/models/alphazero_checkpoint.pt"
DEFAULT_PPO_MODEL_PATH = "gomoku_project/models/ppo/ppo_checkpoint.pt"


def _winner_label(winner: int | None, black_name: str, white_name: str) -> str:
    if winner == BLACK:
        return black_name
    if winner == WHITE:
        return white_name
    return "Draw"


def _mean_metric(metrics: list[dict[str, float]], key: str) -> float | None:
    if not metrics:
        return None
    return sum(item[key] for item in metrics) / len(metrics)


def _format_optional(value: float | None) -> str:
    return "None" if value is None else f"{value:.4f}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play Gomoku: human vs selected AI. Training is disabled unless --train-after-game is set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--opponent",
        choices=("alphazero", "rl", "ppo", "heuristic", "center"),
        default="alphazero",
        help="'rl' is kept as a legacy alias for 'alphazero'.",
    )
    parser.add_argument("--human-color", choices=("black", "white"), default="black")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Generic checkpoint path for the selected learned opponent (alphazero/ppo). Do not combine with the explicit model-path option for that opponent.",
    )
    parser.add_argument(
        "--alphazero-model-path",
        type=str,
        default=None,
        help=f"AlphaZero checkpoint path. Used only with --opponent alphazero/rl. Default: {DEFAULT_ALPHAZERO_MODEL_PATH}",
    )
    parser.add_argument(
        "--ppo-model-path",
        type=str,
        default=None,
        help=f"PPO checkpoint path. Used only with --opponent ppo. Default: {DEFAULT_PPO_MODEL_PATH}",
    )
    parser.add_argument(
        "--allow-random-init",
        action="store_true",
        help="Allow continuing with random-initialized PPO/AlphaZero weights if checkpoint loading is skipped, missing, or fails.",
    )
    parser.add_argument(
        "--reset-model",
        action="store_true",
        help="Skip checkpoint loading for PPO/AlphaZero and start from random initialization. Requires --allow-random-init.",
    )
    parser.add_argument("--train-after-game", action="store_true", help="Store AI-only match data and update the learned AI after the game.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--post-game-delay", type=float, default=3.0)
    parser.add_argument("--move-delay", type=float, default=0.1)
    parser.add_argument("--self-play-games", type=int, default=0)
    parser.add_argument("--self-play-train-steps", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mcts-simulations", type=int, default=24)
    parser.add_argument("--ppo-update-epochs", type=int, default=4)
    parser.add_argument("--ppo-minibatch-size", type=int, default=128)
    parser.add_argument("--deterministic-ai", action="store_true", help="Use deterministic AI moves instead of stochastic training-time play.")
    parser.add_argument(
        "--disable-root-noise",
        action="store_true",
        help="Disable AlphaZero root noise when AlphaZero is used in a training-enabled match.",
    )
    return parser


def _normalize_and_validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    args.opponent = "alphazero" if args.opponent == "rl" else args.opponent

    if args.model_path is not None:
        if args.opponent == "alphazero":
            if args.alphazero_model_path is not None:
                parser.error("Use either --model-path or --alphazero-model-path, not both.")
            args.alphazero_model_path = args.model_path
        elif args.opponent == "ppo":
            if args.ppo_model_path is not None:
                parser.error("Use either --model-path or --ppo-model-path, not both.")
            args.ppo_model_path = args.model_path
        else:
            parser.error("--model-path is only valid with --opponent alphazero/rl or --opponent ppo.")

    if args.opponent == "alphazero":
        if args.ppo_model_path is not None:
            parser.error("--ppo-model-path is only valid with --opponent ppo. Use --alphazero-model-path for AlphaZero.")
    elif args.opponent == "ppo":
        if args.alphazero_model_path is not None:
            parser.error("--alphazero-model-path is only valid with --opponent alphazero/rl. Use --ppo-model-path for PPO.")
    else:
        if args.alphazero_model_path is not None or args.ppo_model_path is not None:
            parser.error(
                "Model checkpoint options are only valid with --opponent alphazero/rl or --opponent ppo."
            )
        if args.reset_model:
            parser.error("--reset-model is only valid with --opponent alphazero or --opponent ppo.")

    if args.reset_model and args.opponent in {"alphazero", "ppo"} and not args.allow_random_init:
        parser.error("--reset-model starts from random initialization. Add --allow-random-init to confirm.")
    if args.train_after_game and args.opponent not in {"alphazero", "ppo"}:
        parser.error("--train-after-game is only supported for --opponent alphazero/rl or --opponent ppo.")
    if args.self_play_games < 0 or args.self_play_train_steps < 0 or args.train_steps < 0:
        parser.error("Training step counts must be >= 0.")

    args.alphazero_model_path = args.alphazero_model_path or DEFAULT_ALPHAZERO_MODEL_PATH
    args.ppo_model_path = args.ppo_model_path or DEFAULT_PPO_MODEL_PATH
    return args


def _build_ai_player(args: argparse.Namespace):
    if args.opponent == "alphazero":
        trainer = AlphaZeroTrainer(
            batch_size=args.batch_size,
            mcts_simulations=args.mcts_simulations,
            device=args.device,
        )
        model_path, loaded = load_checkpoint_or_maybe_warn(
            trainer=trainer,
            path=args.alphazero_model_path,
            model_label="AlphaZero",
            cli_flag="alphazero-model-path",
            allow_random_init=args.allow_random_init,
            reset_model=args.reset_model,
        )

        if args.self_play_games > 0:
            buffer_size = trainer.collect_self_play_games(args.self_play_games)
            warmup_metrics = trainer.train_steps(args.self_play_train_steps)
            print(
                f"alphazero_warmup_buffer_size={buffer_size} "
                f"alphazero_warmup_total_loss={_mean_metric(warmup_metrics, 'total_loss')} "
                f"alphazero_warmup_policy_loss={_mean_metric(warmup_metrics, 'policy_loss')} "
                f"alphazero_warmup_value_loss={_mean_metric(warmup_metrics, 'value_loss')} "
                f"alphazero_loaded_existing_model={loaded}"
            )

        return (
            trainer.build_player(
                deterministic=args.deterministic_ai,
                name="AlphaZero",
                use_root_noise=not args.disable_root_noise and not args.deterministic_ai,
            ),
            trainer,
            model_path,
            args.opponent,
        )

    if args.opponent == "ppo":
        trainer = PPOTrainer(
            device=args.device,
            update_epochs=args.ppo_update_epochs,
            minibatch_size=args.ppo_minibatch_size,
        )
        model_path, _ = load_checkpoint_or_maybe_warn(
            trainer=trainer,
            path=args.ppo_model_path,
            model_label="PPO",
            cli_flag="ppo-model-path",
            allow_random_init=args.allow_random_init,
            reset_model=args.reset_model,
        )
        return (
            trainer.build_player(deterministic=args.deterministic_ai, name="PPO"),
            trainer,
            model_path,
            args.opponent,
        )

    if args.opponent == "heuristic":
        return HeuristicPlayer(name="Heuristic"), None, None, args.opponent

    return CenterPlayer(name="Center"), None, None, args.opponent


def main() -> None:
    parser = _build_parser()
    args = _normalize_and_validate_args(parser, parser.parse_args())

    renderer = TkinterRenderer(title="Gomoku Human vs AI")
    human_player = HumanPlayer(renderer=renderer, name="Human")
    ai_player, trainer, save_path, opponent = _build_ai_player(args)

    if args.human_color == "black":
        black_player = human_player
        white_player = ai_player
        black_participant = MatchParticipant(player=human_player)
        white_participant = MatchParticipant(
            player=ai_player,
            trainer=trainer,
            algorithm="alphazero" if isinstance(trainer, AlphaZeroTrainer) else "ppo" if isinstance(trainer, PPOTrainer) else None,
        )
    else:
        black_player = ai_player
        white_player = human_player
        black_participant = MatchParticipant(
            player=ai_player,
            trainer=trainer,
            algorithm="alphazero" if isinstance(trainer, AlphaZeroTrainer) else "ppo" if isinstance(trainer, PPOTrainer) else None,
        )
        white_participant = MatchParticipant(player=human_player)

    if args.train_after_game and trainer is not None:
        result = run_training_match(
            black_participant=black_participant,
            white_participant=white_participant,
            matchup=f"{opponent}_vs_human",
            env=GomokuEnv(),
            render=True,
            renderer=renderer,
            move_delay=args.move_delay,
            post_game_delay=args.post_game_delay,
            train_after_game=True,
            alphazero_train_steps=args.train_steps,
        )
        trainer.save(save_path)

        metrics = next(
            (
                summary.update_metrics
                for summary in result.participant_summaries
                if summary.algorithm in {"ppo", "alphazero"}
            ),
            None,
        ) or {}
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
        if isinstance(trainer, PPOTrainer):
            print(
                f"ppo_update={bool(metrics)} "
                f"ppo_total_loss={_format_optional(metrics.get('total_loss'))} "
                f"ppo_policy_loss={_format_optional(metrics.get('policy_loss'))} "
                f"ppo_value_loss={_format_optional(metrics.get('value_loss'))}"
            )
        else:
            print(
                f"alphazero_update={bool(metrics)} "
                f"alphazero_total_loss={_format_optional(metrics.get('total_loss'))} "
                f"alphazero_policy_loss={_format_optional(metrics.get('policy_loss'))} "
                f"alphazero_value_loss={_format_optional(metrics.get('value_loss'))}"
            )
        print(f"checkpoint_saved=True checkpoint={save_path}")
        return

    result = run_match(
        black_player=black_player,
        white_player=white_player,
        env=GomokuEnv(),
        render=True,
        renderer=renderer,
        move_delay=args.move_delay,
        post_game_delay=args.post_game_delay,
    )
    print(
        f"mode=play_only "
        f"matchup={opponent}_vs_human "
        f"winner={_winner_label(result.winner, result.black_name, result.white_name)} "
        f"reason={result.reason} "
        f"moves={result.move_count}"
    )
    print("ppo_steps=0 alphazero_examples=0")
    print("update_performed=False checkpoint_saved=False")


if __name__ == "__main__":
    main()
