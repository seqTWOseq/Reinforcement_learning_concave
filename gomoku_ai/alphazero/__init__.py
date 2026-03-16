"""Public AlphaZero-compatible specs and helpers."""

from gomoku_ai.alphazero.checkpoint_manager import (
    BEST_MODEL_CHECKPOINT_FORMAT_VERSION,
    BEST_MODEL_CHECKPOINT_TYPE,
    load_best_model_checkpoint,
    save_best_model_checkpoint,
)
from gomoku_ai.alphazero.checkpoint_loader import load_model_for_inference, resolve_model_checkpoint_path
from gomoku_ai.alphazero.evaluation import AlphaZeroEvaluator, EvaluationConfig, EvaluationResult
from gomoku_ai.alphazero.evaluation_utils import (
    build_evaluation_mcts_config,
    compute_candidate_score,
    play_single_evaluation_game,
)
from gomoku_ai.alphazero.human_play import HumanAITurnRecord, HumanPlayConfig, HumanVsAlphaZeroGameRunner
from gomoku_ai.alphazero.human_play_utils import (
    REQUIRED_HUMAN_PLAY_METADATA_KEYS,
    build_human_play_game_id,
    build_human_play_mcts_config,
    finalize_human_ai_turn_records,
    resolve_human_and_ai_colors,
)
from gomoku_ai.alphazero.human_training import extract_human_play_samples, maybe_extract_ai_turn_samples
from gomoku_ai.alphazero.inference import predict_single
from gomoku_ai.alphazero.mcts import MCTS, MCTSConfig, MCTSNode
from gomoku_ai.alphazero.mcts_utils import opponent_of, resolve_env_player_to_move, terminal_value_for_player
from gomoku_ai.alphazero.model import (
    PolicyValueNet,
    PolicyValueNetConfig,
    count_parameters,
    load_policy_value_net,
    save_policy_value_net,
)
from gomoku_ai.alphazero.specs import (
    ACTION_SIZE,
    POLICY_DTYPE,
    POLICY_SHAPE,
    STATE_CHANNELS,
    STATE_DTYPE,
    STATE_SHAPE,
    VALUE_SHAPE,
    GameRecord,
    GameStepSample,
    NetworkInputSpec,
    PolicyValueOutputSpec,
)
from gomoku_ai.alphazero.self_play import SelfPlayConfig, SelfPlayGameGenerator, SelfPlayTurnRecord
from gomoku_ai.alphazero.self_play_utils import (
    REQUIRED_SELF_PLAY_METADATA_KEYS,
    build_self_play_game_id,
    finalize_turn_records,
    get_temperature_for_move,
)
from gomoku_ai.alphazero.dataset import AlphaZeroSampleDataset
from gomoku_ai.alphazero.trainer import AlphaZeroTrainer, RecentGameBuffer, TrainerConfig
from gomoku_ai.alphazero.trainer_utils import (
    build_dataloader_from_buffer,
    compute_policy_loss,
    compute_value_loss,
    trim_samples,
)
from gomoku_ai.alphazero.training_dashboard import (
    append_training_log_entry,
    build_training_log_entry,
    load_training_log_entries,
    summarize_self_play_records,
    write_training_dashboard,
    write_training_dashboard_from_log,
)
from gomoku_ai.alphazero.utils import (
    build_game_step_sample,
    mask_policy_logits,
    normalize_visit_counts,
    policy_logits_to_probs,
    winner_to_value_target,
)

__all__ = [
    "ACTION_SIZE",
    "AlphaZeroEvaluator",
    "HumanAITurnRecord",
    "HumanPlayConfig",
    "HumanVsAlphaZeroGameRunner",
    "MCTS",
    "MCTSConfig",
    "MCTSNode",
    "PolicyValueNet",
    "PolicyValueNetConfig",
    "BEST_MODEL_CHECKPOINT_FORMAT_VERSION",
    "BEST_MODEL_CHECKPOINT_TYPE",
    "POLICY_DTYPE",
    "POLICY_SHAPE",
    "STATE_CHANNELS",
    "STATE_DTYPE",
    "STATE_SHAPE",
    "VALUE_SHAPE",
    "EvaluationConfig",
    "EvaluationResult",
    "GameRecord",
    "GameStepSample",
    "NetworkInputSpec",
    "PolicyValueOutputSpec",
    "AlphaZeroSampleDataset",
    "AlphaZeroTrainer",
    "RecentGameBuffer",
    "REQUIRED_HUMAN_PLAY_METADATA_KEYS",
    "REQUIRED_SELF_PLAY_METADATA_KEYS",
    "SelfPlayConfig",
    "SelfPlayGameGenerator",
    "SelfPlayTurnRecord",
    "TrainerConfig",
    "build_evaluation_mcts_config",
    "build_dataloader_from_buffer",
    "build_game_step_sample",
    "build_human_play_game_id",
    "build_human_play_mcts_config",
    "build_self_play_game_id",
    "compute_candidate_score",
    "compute_policy_loss",
    "compute_value_loss",
    "count_parameters",
    "extract_human_play_samples",
    "finalize_turn_records",
    "finalize_human_ai_turn_records",
    "get_temperature_for_move",
    "load_best_model_checkpoint",
    "load_model_for_inference",
    "load_policy_value_net",
    "mask_policy_logits",
    "maybe_extract_ai_turn_samples",
    "normalize_visit_counts",
    "opponent_of",
    "play_single_evaluation_game",
    "policy_logits_to_probs",
    "predict_single",
    "resolve_human_and_ai_colors",
    "resolve_env_player_to_move",
    "resolve_model_checkpoint_path",
    "save_best_model_checkpoint",
    "save_policy_value_net",
    "append_training_log_entry",
    "build_training_log_entry",
    "load_training_log_entries",
    "summarize_self_play_records",
    "terminal_value_for_player",
    "trim_samples",
    "write_training_dashboard",
    "write_training_dashboard_from_log",
    "winner_to_value_target",
]
