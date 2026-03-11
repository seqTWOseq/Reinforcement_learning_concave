from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from gomoku_project.core.constants import BOARD_SIZE
from gomoku_project.envs.gomoku_env import GomokuEnv
from gomoku_project.players.rl_player import RLPlayer
from gomoku_project.rl.network import AlphaZeroPolicyValueNet
from gomoku_project.rl.replay_buffer import AlphaZeroExample, ReplayBuffer
from gomoku_project.rl.self_play import play_self_play_game


class AlphaZeroTrainer:
    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        *,
        device: str = "cpu",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        buffer_capacity: int = 100_000,
        board_size: int = BOARD_SIZE,
        mcts_simulations: int = 32,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.03,
        dirichlet_epsilon: float = 0.25,
        temperature_drop_move: int = 10,
    ) -> None:
        self.device = torch.device(device)
        self.model = model or AlphaZeroPolicyValueNet(board_size=board_size)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.board_size = board_size
        self.mcts_simulations = mcts_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.temperature_drop_move = temperature_drop_move
        self.training_steps = 0
        self.self_play_games = 0

    def build_player(
        self,
        *,
        deterministic: bool,
        name: str,
        use_root_noise: bool = False,
    ) -> RLPlayer:
        return RLPlayer(
            model=self.model,
            device=str(self.device),
            deterministic=deterministic,
            name=name,
            num_simulations=self.mcts_simulations,
            c_puct=self.c_puct,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon,
            use_root_noise=use_root_noise,
            temperature_drop_move=self.temperature_drop_move,
        )

    def collect_self_play_games(self, num_games: int = 1) -> int:
        for _ in range(num_games):
            black_player = self.build_player(
                deterministic=False,
                name="AlphaZeroSelfPlayBlack",
                use_root_noise=True,
            )
            white_player = self.build_player(
                deterministic=False,
                name="AlphaZeroSelfPlayWhite",
                use_root_noise=True,
            )
            examples, _ = play_self_play_game(
                black_player=black_player,
                white_player=white_player,
                env=GomokuEnv(board_size=self.board_size),
            )
            self.replay_buffer.extend(examples)
            self.self_play_games += 1
        return len(self.replay_buffer)

    def add_examples(self, examples: Sequence[AlphaZeroExample]) -> int:
        self.replay_buffer.extend(list(examples))
        return len(self.replay_buffer)

    def train_step(self) -> Optional[dict[str, float]]:
        if len(self.replay_buffer) == 0:
            return None

        batch = self.replay_buffer.sample(min(self.batch_size, len(self.replay_buffer)))
        observations = np.stack([example.observation for example in batch]).astype(np.float32)
        policy_targets = np.stack([example.policy_target for example in batch]).astype(np.float32)
        value_targets = np.array([example.value_target for example in batch], dtype=np.float32)

        observation_tensor = torch.as_tensor(observations, dtype=torch.float32, device=self.device).unsqueeze(1)
        policy_target_tensor = torch.as_tensor(policy_targets, dtype=torch.float32, device=self.device)
        value_target_tensor = torch.as_tensor(value_targets, dtype=torch.float32, device=self.device)

        self.model.train()
        policy_logits, predicted_values = self.model(observation_tensor)

        log_probs = F.log_softmax(policy_logits, dim=-1)
        policy_loss = -(policy_target_tensor * log_probs).sum(dim=1).mean()
        value_loss = F.mse_loss(predicted_values, value_target_tensor)
        total_loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.training_steps += 1

        return {
            "total_loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
        }

    def train_steps(self, num_steps: int) -> list[dict[str, float]]:
        metrics: list[dict[str, float]] = []
        for _ in range(num_steps):
            metric = self.train_step()
            if metric is not None:
                metrics.append(metric)
        return metrics

    def load_if_exists(self, path: str | Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False

        try:
            self.load(path)
        except Exception as exc:
            print(f"checkpoint_load_failed={exc}")
            return False
        return True

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "training_steps": self.training_steps,
                "self_play_games": self.self_play_games,
                "board_size": self.board_size,
                "mcts_simulations": self.mcts_simulations,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device)
        if "model_state" not in checkpoint:
            raise ValueError("Checkpoint format is not compatible with AlphaZeroTrainer.")

        self.model.load_state_dict(checkpoint["model_state"])
        optimizer_state = checkpoint.get("optimizer_state")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        self.training_steps = int(checkpoint.get("training_steps", 0))
        self.self_play_games = int(checkpoint.get("self_play_games", 0))
        self.model.to(self.device)
        self.model.eval()


SelfPlayTrainer = AlphaZeroTrainer
