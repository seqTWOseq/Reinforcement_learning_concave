from __future__ import annotations

from pathlib import Path

from gomoku_project.core.constants import BLACK, WHITE
from gomoku_project.envs.gomoku_env import GomokuEnv
from gomoku_project.players.human_player import HumanPlayer
from gomoku_project.rl.self_play import play_recorded_game
from gomoku_project.rl.trainer import AlphaZeroTrainer
from gomoku_project.ui.tkinter_renderer import TkinterRenderer

CHECKPOINT_PATH = Path("gomoku_project/models/alphazero_checkpoint.pt")
PRE_GAME_SELF_PLAY_GAMES = 1
PRE_GAME_TRAIN_STEPS = 4
POST_GAME_TRAIN_STEPS = 8
MCTS_SIMULATIONS = 24


def _mean_metric(metrics: list[dict[str, float]], key: str) -> float | None:
    if not metrics:
        return None
    return sum(item[key] for item in metrics) / len(metrics)


def main() -> None:
    trainer = AlphaZeroTrainer(
        mcts_simulations=MCTS_SIMULATIONS,
        batch_size=32,
    )
    loaded = trainer.load_if_exists(CHECKPOINT_PATH)

    print(f"loaded_existing_checkpoint={loaded}")
    print(f"=== pre_game_self_play games={PRE_GAME_SELF_PLAY_GAMES} ===")
    buffer_size = trainer.collect_self_play_games(PRE_GAME_SELF_PLAY_GAMES)
    pre_metrics = trainer.train_steps(PRE_GAME_TRAIN_STEPS)
    print(
        f"self_play_buffer_size={buffer_size} "
        f"pre_total_loss={_mean_metric(pre_metrics, 'total_loss')} "
        f"pre_policy_loss={_mean_metric(pre_metrics, 'policy_loss')} "
        f"pre_value_loss={_mean_metric(pre_metrics, 'value_loss')}"
    )

    env = GomokuEnv()
    renderer = TkinterRenderer(title="AlphaZero Gomoku Arena")
    agent1 = HumanPlayer(renderer=renderer, name="Human_Black")
    agent2 = trainer.build_player(
        deterministic=True,
        name="AlphaZero_White",
        use_root_noise=False,
    )

    print(f"=== {agent1.name} vs {agent2.name} match start ===")
    examples, final_info = play_recorded_game(
        black_player=agent1,
        white_player=agent2,
        env=env,
        render=True,
        renderer=renderer,
        move_delay=0.1,
        post_game_delay=3.0,
    )

    buffer_size = trainer.add_examples(examples)
    post_metrics = trainer.train_steps(POST_GAME_TRAIN_STEPS)
    trainer.save(CHECKPOINT_PATH)

    print("\n=== Match finished ===")
    if final_info.get("winner") == BLACK:
        print(f"{agent1.name} wins!")
    elif final_info.get("winner") == WHITE:
        print(f"{agent2.name} wins!")
    else:
        print("Draw!")
    print(
        f"examples_added={len(examples)} "
        f"reason={final_info.get('reason')} "
        f"buffer_size={buffer_size} "
        f"post_total_loss={_mean_metric(post_metrics, 'total_loss')} "
        f"post_policy_loss={_mean_metric(post_metrics, 'policy_loss')} "
        f"post_value_loss={_mean_metric(post_metrics, 'value_loss')} "
        f"saved_checkpoint={CHECKPOINT_PATH}"
    )


if __name__ == "__main__":
    main()
