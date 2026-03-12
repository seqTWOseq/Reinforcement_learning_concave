from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gomoku_project.arena.match_runner import run_match
from gomoku_project.core.constants import BLACK, WHITE
from gomoku_project.envs.gomoku_env import GomokuEnv
from gomoku_project.players.human_player import HumanPlayer
from gomoku_project.rl.ppo_trainer import PPOTrainer
from gomoku_project.ui.tkinter_renderer import TkinterRenderer
from gomoku_project.utils.checkpoint_utils import load_checkpoint_or_maybe_warn


def _winner_label(winner: int | None, black_name: str, white_name: str) -> str:
    if winner == BLACK:
        return black_name
    if winner == WHITE:
        return white_name
    return "Draw"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play Gomoku: human vs PPO. The PPO checkpoint is required unless --allow-random-init is set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", type=str, default="gomoku_project/models/ppo/ppo_checkpoint.pt")
    parser.add_argument(
        "--allow-random-init",
        action="store_true",
        help="Allow PPO to use random initialization if the checkpoint is missing or invalid.",
    )
    parser.add_argument("--human-color", choices=("black", "white"), default="black")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--move-delay", type=float, default=0.1)
    parser.add_argument("--post-game-delay", type=float, default=3.0)
    parser.set_defaults(use_heuristic_prior=True)
    parser.add_argument(
        "--use-heuristic-prior",
        dest="use_heuristic_prior",
        action="store_true",
        help="Enable heuristic prior injection during PPO play.",
    )
    parser.add_argument(
        "--disable-heuristic-prior",
        dest="use_heuristic_prior",
        action="store_false",
        help="Disable heuristic prior injection during PPO play.",
    )
    parser.add_argument("--heuristic-prior-beta-start", type=float, default=1.5)
    parser.add_argument("--heuristic-prior-beta-end", type=float, default=1.5)
    parser.add_argument("--heuristic-prior-decay-updates", type=int, default=1)
    parser.add_argument("--heuristic-prior-score-clip", type=float, default=2.5)
    args = parser.parse_args()

    trainer = PPOTrainer(
        device=args.device,
        use_heuristic_prior=args.use_heuristic_prior,
        heuristic_prior_beta_start=args.heuristic_prior_beta_start,
        heuristic_prior_beta_end=args.heuristic_prior_beta_end,
        heuristic_prior_decay_updates=args.heuristic_prior_decay_updates,
        heuristic_prior_score_clip=args.heuristic_prior_score_clip,
    )
    load_checkpoint_or_maybe_warn(
        trainer=trainer,
        path=args.model_path,
        model_label="PPO",
        cli_flag="model-path",
        allow_random_init=args.allow_random_init,
    )

    renderer = TkinterRenderer(title="Gomoku Human vs PPO")
    human_player = HumanPlayer(renderer=renderer, name="Human")
    ppo_player = trainer.build_player(deterministic=True, name="PPO")

    if args.human_color == "black":
        black_player = human_player
        white_player = ppo_player
    else:
        black_player = ppo_player
        white_player = human_player

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
        f"winner={_winner_label(result.winner, result.black_name, result.white_name)} "
        f"reason={result.reason} "
        f"moves={result.move_count}"
    )


if __name__ == "__main__":
    main()
