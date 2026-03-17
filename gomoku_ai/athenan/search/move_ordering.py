"""Move-ordering helpers for shallow Athenan search baselines."""

from __future__ import annotations

from gomoku_ai.env import BLACK, DIRECTIONS, GomokuEnv, WHITE


def score_action(
    env: GomokuEnv,
    action: int,
    *,
    player: int | None = None,
) -> float:
    """Return a heuristic ordering score for one legal action."""

    if env.done:
        raise ValueError("Cannot score actions on a finished game.")

    player_to_move = env.current_player if player is None else int(player)
    env.action_to_coord(action)
    legal_moves = env.get_valid_moves()
    if not legal_moves[action]:
        raise ValueError(f"Action {action} is not legal in the current position.")

    opponent = WHITE if player_to_move == BLACK else BLACK
    own_win_bonus = 1_000_000.0 if _is_immediate_win_for_player(env, action, player_to_move) else 0.0
    block_bonus = 500_000.0 if _is_blocking_opponent_immediate_win(env, action, player_to_move) else 0.0
    line_bonus = float(_line_length_after_move(env, action, player_to_move) * 1_000)

    row, col = env.action_to_coord(action)
    friendly_neighbors = _count_neighbors(env, row, col, player_to_move, radius=2)
    opponent_neighbors = _count_neighbors(env, row, col, opponent, radius=2)
    neighbor_bonus = float(friendly_neighbors * 10 + opponent_neighbors * 4)

    center = (env.board_size - 1) / 2.0
    center_distance = abs(row - center) + abs(col - center)
    center_bonus = float((env.board_size * 2) - center_distance)

    return float(own_win_bonus + block_bonus + line_bonus + neighbor_bonus + center_bonus)


def order_actions(
    env: GomokuEnv,
    actions: list[int] | tuple[int, ...],
    *,
    player: int | None = None,
    candidate_limit: int | None = None,
) -> list[int]:
    """Sort candidate actions by descending ordering score."""

    if candidate_limit is not None and candidate_limit <= 0:
        raise ValueError("candidate_limit must be positive when provided.")

    legal_moves = env.get_valid_moves()
    unique_legal_actions: list[int] = []
    seen: set[int] = set()
    for action in actions:
        normalized = int(action)
        if normalized in seen:
            continue
        if 0 <= normalized < legal_moves.size and legal_moves[normalized]:
            unique_legal_actions.append(normalized)
            seen.add(normalized)

    scored = [(action, score_action(env, action, player=player)) for action in unique_legal_actions]
    scored.sort(key=lambda item: (-item[1], item[0]))
    ordered = [action for action, _ in scored]
    if candidate_limit is not None:
        return ordered[:candidate_limit]
    return ordered


def _is_blocking_opponent_immediate_win(env: GomokuEnv, action: int, player: int) -> bool:
    """Return `True` when `player` blocks opponent one-move win at `action`."""

    opponent = WHITE if player == BLACK else BLACK
    return _is_immediate_win_for_player(env, action, opponent)


def _is_immediate_win_for_player(env: GomokuEnv, action: int, player: int) -> bool:
    """Return `True` when `player` wins by playing `action` immediately."""

    cloned = env.clone()
    cloned.current_player = player
    cloned.done = False
    cloned.winner = None
    try:
        cloned.apply_move(action)
    except ValueError:
        return False
    return bool(cloned.done and cloned.winner == player)


def _line_length_after_move(env: GomokuEnv, action: int, player: int) -> int:
    """Compute the longest line length created by `player` at `action`."""

    cloned = env.clone()
    cloned.current_player = player
    cloned.done = False
    cloned.winner = None
    cloned.apply_move(action)
    row, col = cloned.action_to_coord(action)
    return _max_line_length_from_point(cloned, row, col, player)


def _max_line_length_from_point(env: GomokuEnv, row: int, col: int, player: int) -> int:
    """Return the max contiguous line through `(row, col)` for `player`."""

    max_length = 1
    for delta_row, delta_col in DIRECTIONS:
        length = 1
        length += _count_direction(env, row, col, player, delta_row, delta_col)
        length += _count_direction(env, row, col, player, -delta_row, -delta_col)
        if length > max_length:
            max_length = length
    return max_length


def _count_direction(
    env: GomokuEnv,
    row: int,
    col: int,
    player: int,
    delta_row: int,
    delta_col: int,
) -> int:
    """Count contiguous stones for `player` in one direction."""

    count = 0
    next_row = row + delta_row
    next_col = col + delta_col
    while (
        0 <= next_row < env.board_size
        and 0 <= next_col < env.board_size
        and int(env.board[next_row, next_col]) == player
    ):
        count += 1
        next_row += delta_row
        next_col += delta_col
    return count


def _count_neighbors(env: GomokuEnv, row: int, col: int, player: int, *, radius: int) -> int:
    """Count nearby stones for `player` around `(row, col)`."""

    count = 0
    for delta_row in range(-radius, radius + 1):
        for delta_col in range(-radius, radius + 1):
            if delta_row == 0 and delta_col == 0:
                continue
            next_row = row + delta_row
            next_col = col + delta_col
            if 0 <= next_row < env.board_size and 0 <= next_col < env.board_size:
                if int(env.board[next_row, next_col]) == player:
                    count += 1
    return count
