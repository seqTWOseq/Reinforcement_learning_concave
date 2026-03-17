"""Match-based evaluator for Athenan value-search agents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from gomoku_ai.athenan.eval.play import play_agent_game
from gomoku_ai.athenan.network import AthenanValueNet, save_athenan_value_net
from gomoku_ai.athenan.search import AthenanInferenceSearcher, AthenanSearcher
from gomoku_ai.common.agents import BaseAgent, BaseSearcher
from gomoku_ai.env import BLACK, DRAW, GomokuEnv, WHITE


class RandomLegalAgent(BaseAgent):
    """Simple random legal-move baseline for evaluation."""

    def __init__(self, *, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def select_action(self, env: GomokuEnv) -> int:
        legal = np.flatnonzero(np.asarray(env.get_valid_moves(), dtype=bool))
        if legal.size == 0:
            raise RuntimeError("No legal moves available for RandomLegalAgent.")
        return int(self._rng.choice(legal))


class SearcherAgent(BaseAgent):
    """Wrap `BaseSearcher` as `BaseAgent` for evaluation games."""

    def __init__(self, searcher: BaseSearcher) -> None:
        self.searcher = searcher

    def select_action(self, env: GomokuEnv) -> int:
        result = self.searcher.search(env)
        action = int(result.best_action)
        if action < 0:
            raise RuntimeError("Searcher returned best_action < 0 on a non-terminal evaluation turn.")
        legal_mask = np.asarray(env.get_valid_moves(), dtype=bool)
        if not (0 <= action < legal_mask.size and bool(legal_mask[action])):
            raise RuntimeError(f"Searcher produced illegal action during evaluation: {action}.")
        return action


class AlphaZeroAgentAdapter(BaseAgent):
    """Scaffold adapter for future AlphaZero-vs-Athenan evaluation.

    TODO: Wire AlphaZero MCTS + checkpoint inference once cross-engine match
    protocol is fixed.
    """

    def __init__(self, *, checkpoint_path: str | Path | None = None) -> None:
        self.checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path)

    def select_action(self, env: GomokuEnv) -> int:
        raise NotImplementedError(
            "AlphaZeroAgentAdapter is a scaffold. TODO: implement AlphaZero inference adapter."
        )


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregated head-to-head evaluation metrics."""

    games: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    score_rate: float
    avg_move_count: float


@dataclass(frozen=True)
class BestCheckpointDecision:
    """Result of best-checkpoint update decision."""

    updated: bool
    candidate_win_rate: float
    best_win_rate: float
    best_checkpoint_path: str | None


@dataclass(frozen=True)
class TrainInferenceComparison:
    """Side-by-side summaries for train-search and inference-search settings."""

    train_summary: EvaluationSummary
    inference_summary: EvaluationSummary


class AthenanEvaluator:
    """Evaluate agents by match score-rate/win-rate and manage best checkpoints."""

    def __init__(
        self,
        *,
        env_factory: Callable[[], GomokuEnv] | None = None,
        best_checkpoint_path: str | Path | None = None,
    ) -> None:
        self.env_factory = env_factory or GomokuEnv
        self.best_checkpoint_path = None if best_checkpoint_path is None else Path(best_checkpoint_path)
        self.best_win_rate: float = -1.0

    def evaluate_agents(
        self,
        *,
        target_agent: BaseAgent,
        opponent_agent: BaseAgent,
        num_games: int = 10,
    ) -> EvaluationSummary:
        """Evaluate `target_agent` against `opponent_agent` with color alternation."""

        if num_games <= 0:
            raise ValueError("num_games must be positive.")

        wins = 0
        losses = 0
        draws = 0
        total_moves = 0

        for game_index in range(num_games):
            if game_index % 2 == 0:
                black_agent = target_agent
                white_agent = opponent_agent
                target_color = BLACK
            else:
                black_agent = opponent_agent
                white_agent = target_agent
                target_color = WHITE

            game_result = play_agent_game(
                black_agent,
                white_agent,
                env_factory=self.env_factory,
            )
            total_moves += int(game_result.move_count)

            if game_result.winner == DRAW:
                draws += 1
            elif game_result.winner == target_color:
                wins += 1
            else:
                losses += 1

        return EvaluationSummary(
            games=num_games,
            wins=wins,
            losses=losses,
            draws=draws,
            win_rate=float(wins / num_games),
            score_rate=float((wins + 0.5 * draws) / num_games),
            avg_move_count=float(total_moves / num_games),
        )

    def evaluate_searcher(
        self,
        *,
        target_searcher: BaseSearcher,
        opponent_agent: BaseAgent,
        num_games: int = 10,
    ) -> EvaluationSummary:
        """Evaluate a searcher-backed target against a given opponent agent."""

        return self.evaluate_agents(
            target_agent=SearcherAgent(target_searcher),
            opponent_agent=opponent_agent,
            num_games=num_games,
        )

    def evaluate_model_vs_random(
        self,
        model: AthenanValueNet | None,
        *,
        num_games: int = 10,
        random_seed: int | None = None,
        searcher_max_depth: int = 1,
        searcher_candidate_limit: int | None = 32,
        searcher_candidate_radius: int = 2,
        searcher_use_alpha_beta: bool = True,
        device: str | None = None,
    ) -> EvaluationSummary:
        """Evaluate train-search settings against random legal baseline."""

        searcher = self.build_train_searcher(
            model=model,
            max_depth=searcher_max_depth,
            candidate_limit=searcher_candidate_limit,
            candidate_radius=searcher_candidate_radius,
            use_alpha_beta=searcher_use_alpha_beta,
            device=device,
        )
        return self.evaluate_searcher(
            target_searcher=searcher,
            opponent_agent=RandomLegalAgent(seed=random_seed),
            num_games=num_games,
        )

    def evaluate_inference_search_vs_random(
        self,
        model: AthenanValueNet | None,
        *,
        num_games: int = 10,
        random_seed: int | None = None,
        searcher_max_depth: int = 4,
        searcher_candidate_limit: int | None = 64,
        searcher_candidate_radius: int = 2,
        searcher_use_alpha_beta: bool = True,
        iterative_deepening: bool = True,
        time_budget_sec: float | None = None,
        device: str | None = None,
    ) -> EvaluationSummary:
        """Evaluate stronger inference-search settings against random baseline."""

        searcher = self.build_inference_searcher(
            model=model,
            max_depth=searcher_max_depth,
            candidate_limit=searcher_candidate_limit,
            candidate_radius=searcher_candidate_radius,
            use_alpha_beta=searcher_use_alpha_beta,
            iterative_deepening=iterative_deepening,
            time_budget_sec=time_budget_sec,
            device=device,
        )
        return self.evaluate_searcher(
            target_searcher=searcher,
            opponent_agent=RandomLegalAgent(seed=random_seed),
            num_games=num_games,
        )

    def evaluate_train_vs_inference(
        self,
        model: AthenanValueNet | None,
        *,
        opponent_agent: BaseAgent | None = None,
        num_games: int = 10,
        train_search_kwargs: Mapping[str, object] | None = None,
        inference_search_kwargs: Mapping[str, object] | None = None,
    ) -> TrainInferenceComparison:
        """Compare train-search and inference-search with the same model/opponent."""

        opponent = opponent_agent or RandomLegalAgent(seed=0)
        resolved_train_kwargs = dict(train_search_kwargs or {})
        resolved_infer_kwargs = dict(inference_search_kwargs or {})
        train_searcher = self.build_train_searcher(model=model, **resolved_train_kwargs)
        inference_searcher = self.build_inference_searcher(model=model, **resolved_infer_kwargs)
        train_summary = self.evaluate_searcher(
            target_searcher=train_searcher,
            opponent_agent=opponent,
            num_games=num_games,
        )
        inference_summary = self.evaluate_searcher(
            target_searcher=inference_searcher,
            opponent_agent=opponent,
            num_games=num_games,
        )
        return TrainInferenceComparison(
            train_summary=train_summary,
            inference_summary=inference_summary,
        )

    def evaluate_and_update_best_checkpoint(
        self,
        *,
        model: AthenanValueNet,
        opponent_agent: BaseAgent,
        num_games: int = 10,
        candidate_checkpoint_path: str | Path | None = None,
        use_inference_search: bool = True,
        searcher_max_depth: int = 4,
        searcher_candidate_limit: int | None = 64,
        searcher_candidate_radius: int = 2,
        searcher_use_alpha_beta: bool = True,
        iterative_deepening: bool = True,
        time_budget_sec: float | None = None,
        device: str | None = None,
    ) -> tuple[EvaluationSummary, BestCheckpointDecision]:
        """Evaluate model by win-rate and update best checkpoint when improved."""

        if use_inference_search:
            target_searcher: BaseSearcher = self.build_inference_searcher(
                model=model,
                max_depth=searcher_max_depth,
                candidate_limit=searcher_candidate_limit,
                candidate_radius=searcher_candidate_radius,
                use_alpha_beta=searcher_use_alpha_beta,
                iterative_deepening=iterative_deepening,
                time_budget_sec=time_budget_sec,
                device=device,
            )
        else:
            target_searcher = self.build_train_searcher(
                model=model,
                max_depth=searcher_max_depth,
                candidate_limit=searcher_candidate_limit,
                candidate_radius=searcher_candidate_radius,
                use_alpha_beta=searcher_use_alpha_beta,
                device=device,
            )

        summary = self.evaluate_searcher(
            target_searcher=target_searcher,
            opponent_agent=opponent_agent,
            num_games=num_games,
        )
        improved = summary.win_rate > self.best_win_rate
        if improved:
            self.best_win_rate = summary.win_rate
            if self.best_checkpoint_path is not None:
                source_path = None if candidate_checkpoint_path is None else str(Path(candidate_checkpoint_path))
                save_athenan_value_net(
                    model,
                    self.best_checkpoint_path,
                    metadata={
                        "eval_win_rate": summary.win_rate,
                        "source_checkpoint_path": source_path,
                    },
                )

        return summary, BestCheckpointDecision(
            updated=improved,
            candidate_win_rate=summary.win_rate,
            best_win_rate=self.best_win_rate,
            best_checkpoint_path=None if self.best_checkpoint_path is None else str(self.best_checkpoint_path),
        )

    @staticmethod
    def build_train_searcher(
        *,
        model: AthenanValueNet | None,
        max_depth: int = 2,
        candidate_limit: int | None = 32,
        candidate_radius: int = 2,
        use_alpha_beta: bool = True,
        device: str | None = None,
        **_: object,
    ) -> AthenanSearcher:
        """Construct a shallow training-oriented searcher preset."""

        return AthenanSearcher(
            model=model,
            max_depth=max_depth,
            candidate_limit=candidate_limit,
            candidate_radius=candidate_radius,
            use_alpha_beta=use_alpha_beta,
            device=device,
        )

    @staticmethod
    def build_inference_searcher(
        *,
        model: AthenanValueNet | None,
        max_depth: int = 4,
        candidate_limit: int | None = 64,
        candidate_radius: int = 2,
        use_alpha_beta: bool = True,
        iterative_deepening: bool = True,
        time_budget_sec: float | None = None,
        device: str | None = None,
        **_: object,
    ) -> AthenanInferenceSearcher:
        """Construct a deeper inference-oriented searcher preset."""

        return AthenanInferenceSearcher(
            model=model,
            max_depth=max_depth,
            candidate_limit=candidate_limit,
            candidate_radius=candidate_radius,
            use_alpha_beta=use_alpha_beta,
            iterative_deepening=iterative_deepening,
            time_budget_sec=time_budget_sec,
            device=device,
        )
