"""Athenan search scaffolding."""

from gomoku_ai.athenan.search.minimax import AthenanInferenceSearcher
from gomoku_ai.athenan.search.move_ordering import order_actions, score_action
from gomoku_ai.athenan.search.searcher import AthenanSearcher
from gomoku_ai.athenan.search.tactical_rules import (
    apply_forced_tactical_rule,
    find_immediate_blocking_actions,
    find_immediate_winning_actions,
    generate_proximity_candidates,
)

__all__ = [
    "AthenanInferenceSearcher",
    "AthenanSearcher",
    "apply_forced_tactical_rule",
    "find_immediate_blocking_actions",
    "find_immediate_winning_actions",
    "generate_proximity_candidates",
    "order_actions",
    "score_action",
]
