"""Training-log helpers and static HTML dashboard generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from gomoku_ai.alphazero.evaluation import EvaluationResult
from gomoku_ai.alphazero.specs import GameRecord
from gomoku_ai.env import BLACK, DRAW, WHITE


def summarize_self_play_records(records: Sequence[GameRecord]) -> dict[str, float]:
    """Summarize one cycle of self-play records into dashboard-friendly metrics."""

    if not records:
        return {
            "avg_game_length": 0.0,
            "min_game_length": 0.0,
            "max_game_length": 0.0,
            "black_win_rate": 0.0,
            "white_win_rate": 0.0,
            "draw_rate": 0.0,
            "avg_policy_entropy": 0.0,
        }

    lengths = [float(len(record.moves)) for record in records]
    black_wins = sum(1 for record in records if record.winner == BLACK)
    white_wins = sum(1 for record in records if record.winner == WHITE)
    draws = sum(1 for record in records if record.winner == DRAW)
    total_games = float(len(records))

    entropies: list[float] = []
    for record in records:
        for sample in record.samples:
            probs = np.asarray(sample.policy_target, dtype=np.float32)
            positive_probs = probs[probs > 0.0]
            if positive_probs.size == 0:
                continue
            entropy = -float(np.sum(positive_probs * np.log(positive_probs)))
            entropies.append(entropy)

    return {
        "avg_game_length": float(mean(lengths)),
        "min_game_length": float(min(lengths)),
        "max_game_length": float(max(lengths)),
        "black_win_rate": black_wins / total_games,
        "white_win_rate": white_wins / total_games,
        "draw_rate": draws / total_games,
        "avg_policy_entropy": float(mean(entropies)) if entropies else 0.0,
    }


def build_training_log_entry(
    cycle_index: int,
    training_metrics: Mapping[str, float | str],
    self_play_records: Sequence[GameRecord],
    *,
    evaluation_result: EvaluationResult | None = None,
    promoted: bool | None = None,
    bootstrapped: bool | None = None,
) -> dict[str, Any]:
    """Build one JSON-serializable dashboard entry for a completed cycle."""

    if cycle_index < 0:
        raise ValueError("cycle_index must be non-negative.")

    entry: dict[str, Any] = {
        "cycle_index": cycle_index,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    for key, value in training_metrics.items():
        if isinstance(value, str):
            entry[key] = value
        else:
            entry[key] = float(value)
    entry.update(summarize_self_play_records(self_play_records))

    if evaluation_result is not None:
        entry.update(
            {
                "eval_num_games": float(evaluation_result.num_games),
                "eval_candidate_wins": float(evaluation_result.candidate_wins),
                "eval_reference_wins": float(evaluation_result.reference_wins),
                "eval_draws": float(evaluation_result.draws),
                "eval_candidate_score": float(evaluation_result.candidate_score),
                "eval_candidate_win_rate": float(evaluation_result.candidate_win_rate),
                "eval_promoted": bool(promoted if promoted is not None else evaluation_result.promoted),
                "eval_bootstrapped_best_model": bool(bootstrapped) if bootstrapped is not None else False,
            }
        )

    return entry


def append_training_log_entry(path: str | Path, entry: Mapping[str, Any]) -> Path:
    """Append one dashboard entry to a JSONL log file."""

    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(entry), ensure_ascii=True, sort_keys=True))
        handle.write("\n")
    return log_path


def load_training_log_entries(path: str | Path) -> list[dict[str, Any]]:
    """Load all dashboard entries from a JSONL log file."""

    log_path = Path(path)
    if not log_path.exists():
        return []

    entries: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {log_path}.") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected a JSON object on line {line_number} of {log_path}.")
            entries.append(payload)

    entries.sort(key=lambda entry: int(entry["cycle_index"]))
    return entries


def write_training_dashboard(
    entries: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    title: str = "AlphaZero Training Dashboard",
) -> Path:
    """Render a static HTML dashboard from parsed log entries."""

    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    html = _build_dashboard_html(entries, title=title)
    resolved_output_path.write_text(html, encoding="utf-8")
    return resolved_output_path


def write_training_dashboard_from_log(
    log_path: str | Path,
    output_path: str | Path,
    *,
    title: str = "AlphaZero Training Dashboard",
) -> Path:
    """Load a JSONL training log and render a static HTML dashboard."""

    return write_training_dashboard(load_training_log_entries(log_path), output_path, title=title)


def _build_dashboard_html(entries: Sequence[Mapping[str, Any]], *, title: str) -> str:
    """Build the full HTML document for the static dashboard."""

    latest = entries[-1] if entries else {}
    summary_cards = [
        ("Cycles", str(len(entries))),
        ("Latest Total Loss", _format_decimal(latest.get("total_loss"))),
        ("Latest Policy Acc", _format_percentage(latest.get("policy_top1_accuracy"))),
        ("Latest Value Acc", _format_percentage(latest.get("value_outcome_accuracy"))),
        ("Latest Avg Length", _format_decimal(latest.get("avg_game_length"))),
        ("Latest Eval Score", _format_percentage(latest.get("eval_candidate_score"))),
    ]

    charts = [
        _build_line_chart(
            entries,
            title="Loss",
            series=(
                ("policy_loss", "Policy Loss", "#c44536"),
                ("value_loss", "Value Loss", "#4b6584"),
                ("total_loss", "Total Loss", "#0f8b8d"),
            ),
        ),
        _build_line_chart(
            entries,
            title="Training Accuracy",
            series=(
                ("policy_top1_accuracy", "Policy Top-1", "#1e3799"),
                ("value_outcome_accuracy", "Value Outcome", "#38ada9"),
            ),
            y_min=0.0,
            y_max=1.0,
            percentage_axis=True,
        ),
        _build_line_chart(
            entries,
            title="Self-Play Outcomes",
            series=(
                ("black_win_rate", "Black Win Rate", "#2d3436"),
                ("white_win_rate", "White Win Rate", "#f5cd79"),
                ("draw_rate", "Draw Rate", "#778ca3"),
            ),
            y_min=0.0,
            y_max=1.0,
            percentage_axis=True,
        ),
        _build_line_chart(
            entries,
            title="Game Length",
            series=(("avg_game_length", "Average Moves", "#8854d0"),),
        ),
        _build_line_chart(
            entries,
            title="Evaluation",
            series=(
                ("eval_candidate_score", "Candidate Score", "#eb3b5a"),
                ("eval_candidate_win_rate", "Candidate Win Rate", "#20bf6b"),
            ),
            y_min=0.0,
            y_max=1.0,
            percentage_axis=True,
        ),
    ]

    table_rows = "".join(_build_table_row(entry) for entry in entries)
    cards_html = "".join(
        f"<div class='card'><div class='label'>{escape(label)}</div><div class='value'>{escape(value)}</div></div>"
        for label, value in summary_cards
    )
    charts_html = "".join(f"<section class='panel chart-panel'>{chart}</section>" for chart in charts)
    empty_message = ""
    if not entries:
        empty_message = "<p class='empty'>No training cycles have been logged yet.</p>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1f2a44;
      --muted: #5f6b85;
      --paper: #f6f8fb;
      --panel: #ffffff;
      --line: #d7ddea;
      --accent: #0f8b8d;
      --shadow: 0 14px 32px rgba(31, 42, 68, 0.08);
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      background:
        radial-gradient(circle at top right, rgba(15, 139, 141, 0.12), transparent 30%),
        linear-gradient(180deg, #edf2fb 0%, var(--paper) 55%, #eef4f9 100%);
      color: var(--ink);
    }}
    main {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 2rem;
      letter-spacing: 0.02em;
    }}
    .subtitle {{
      margin: 0 0 24px;
      color: var(--muted);
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-bottom: 20px;
    }}
    .card, .panel {{
      background: var(--panel);
      border: 1px solid rgba(215, 221, 234, 0.9);
      border-radius: 18px;
      box-shadow: var(--shadow);
    }}
    .card {{
      padding: 16px 18px;
    }}
    .label {{
      color: var(--muted);
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .value {{
      margin-top: 8px;
      font-size: 1.55rem;
      font-weight: 700;
    }}
    .charts {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
      margin-bottom: 20px;
    }}
    .chart-panel {{
      padding: 16px 16px 10px;
    }}
    .chart-title {{
      margin: 0 0 12px;
      font-size: 1rem;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 0.84rem;
    }}
    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .legend i {{
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 999px;
    }}
    svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .axis-label {{
      font-size: 11px;
      fill: var(--muted);
    }}
    .table-panel {{
      overflow: hidden;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
    }}
    thead {{
      background: #eef4ff;
    }}
    th, td {{
      padding: 12px 10px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    .empty {{
      color: var(--muted);
      margin: 18px 0;
    }}
    @media (max-width: 720px) {{
      main {{
        padding: 20px 12px 32px;
      }}
      h1 {{
        font-size: 1.6rem;
      }}
      .value {{
        font-size: 1.3rem;
      }}
      th, td {{
        padding: 10px 8px;
        font-size: 0.82rem;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>{escape(title)}</h1>
    <p class="subtitle">Static dashboard generated from self-play and training logs.</p>
    <section class="cards">{cards_html}</section>
    {empty_message}
    <section class="charts">{charts_html}</section>
    <section class="panel table-panel">
      <table>
        <thead>
          <tr>
            <th>Cycle</th>
            <th>Total Loss</th>
            <th>Policy Acc</th>
            <th>Value Acc</th>
            <th>Avg Length</th>
            <th>Black Win</th>
            <th>White Win</th>
            <th>Draw</th>
            <th>Eval Score</th>
            <th>Promoted</th>
          </tr>
        </thead>
        <tbody>{table_rows}</tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""


def _build_table_row(entry: Mapping[str, Any]) -> str:
    """Build one HTML table row for a dashboard entry."""

    return (
        "<tr>"
        f"<td>{int(entry['cycle_index'])}</td>"
        f"<td>{escape(_format_decimal(entry.get('total_loss')))}</td>"
        f"<td>{escape(_format_percentage(entry.get('policy_top1_accuracy')))}</td>"
        f"<td>{escape(_format_percentage(entry.get('value_outcome_accuracy')))}</td>"
        f"<td>{escape(_format_decimal(entry.get('avg_game_length')))}</td>"
        f"<td>{escape(_format_percentage(entry.get('black_win_rate')))}</td>"
        f"<td>{escape(_format_percentage(entry.get('white_win_rate')))}</td>"
        f"<td>{escape(_format_percentage(entry.get('draw_rate')))}</td>"
        f"<td>{escape(_format_percentage(entry.get('eval_candidate_score')))}</td>"
        f"<td>{escape(_format_bool(entry.get('eval_promoted')))}</td>"
        "</tr>"
    )


def _build_line_chart(
    entries: Sequence[Mapping[str, Any]],
    *,
    title: str,
    series: Sequence[tuple[str, str, str]],
    y_min: float | None = None,
    y_max: float | None = None,
    percentage_axis: bool = False,
) -> str:
    """Render a lightweight SVG line chart for selected entry keys."""

    width = 540
    height = 240
    padding_left = 48
    padding_right = 12
    padding_top = 12
    padding_bottom = 28
    plot_width = width - padding_left - padding_right
    plot_height = height - padding_top - padding_bottom

    prepared_series: list[tuple[str, str, str, list[tuple[float, float]]]] = []
    all_values: list[float] = []
    for key, label, color in series:
        values: list[tuple[float, float]] = []
        for index, entry in enumerate(entries):
            if key not in entry or entry[key] is None:
                continue
            value = float(entry[key])
            values.append((float(index), value))
            all_values.append(value)
        prepared_series.append((key, label, color, values))

    legend_html = "".join(
        f"<span><i style='background:{escape(color)}'></i>{escape(label)}</span>"
        for _, label, color, values in prepared_series
        if values
    )

    if not entries or not all_values:
        return f"<h2 class='chart-title'>{escape(title)}</h2><div class='legend'>{legend_html}</div><p class='empty'>No data.</p>"

    resolved_y_min = min(all_values) if y_min is None else y_min
    resolved_y_max = max(all_values) if y_max is None else y_max
    if resolved_y_min == resolved_y_max:
        delta = 1.0 if resolved_y_min == 0.0 else abs(resolved_y_min) * 0.1
        resolved_y_min -= delta
        resolved_y_max += delta

    x_denominator = max(1, len(entries) - 1)
    y_range = resolved_y_max - resolved_y_min

    grid_lines = []
    for tick_index in range(5):
        tick_ratio = tick_index / 4
        tick_y = padding_top + plot_height * tick_ratio
        tick_value = resolved_y_max - y_range * tick_ratio
        label = _format_percentage(tick_value) if percentage_axis else _format_decimal(tick_value)
        grid_lines.append(
            f"<line x1='{padding_left}' y1='{tick_y:.2f}' x2='{padding_left + plot_width}' y2='{tick_y:.2f}' "
            "stroke='#e3e9f3' stroke-width='1' />"
        )
        grid_lines.append(
            f"<text class='axis-label' x='{padding_left - 8}' y='{tick_y + 4:.2f}' text-anchor='end'>{escape(label)}</text>"
        )

    x_labels = []
    for index, entry in enumerate(entries):
        x = padding_left + plot_width * (index / x_denominator)
        if len(entries) > 10 and index not in {0, len(entries) - 1} and index % 2 == 1:
            continue
        x_labels.append(
            f"<text class='axis-label' x='{x:.2f}' y='{height - 8}' text-anchor='middle'>{int(entry['cycle_index'])}</text>"
        )

    polylines = []
    for _, label, color, values in prepared_series:
        if not values:
            continue
        points = []
        for x_index, value in values:
            x = padding_left + plot_width * (x_index / x_denominator)
            y = padding_top + plot_height * ((resolved_y_max - value) / y_range)
            points.append((x, y))
        points_attr = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        circles = "".join(
            f"<circle cx='{x:.2f}' cy='{y:.2f}' r='3.2' fill='{escape(color)}'>"
            f"<title>{escape(label)}: {escape(_format_percentage(value) if percentage_axis else _format_decimal(value))}</title>"
            "</circle>"
            for (x, y), (_, value) in zip(points, values, strict=False)
        )
        polylines.append(
            f"<polyline fill='none' stroke='{escape(color)}' stroke-width='2.6' points='{points_attr}' />{circles}"
        )

    svg = (
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{escape(title)} chart'>"
        f"{''.join(grid_lines)}"
        f"<line x1='{padding_left}' y1='{padding_top + plot_height:.2f}' x2='{padding_left + plot_width}' "
        f"y2='{padding_top + plot_height:.2f}' stroke='#b9c4d7' stroke-width='1.2' />"
        f"<line x1='{padding_left}' y1='{padding_top}' x2='{padding_left}' y2='{padding_top + plot_height:.2f}' "
        "stroke='#b9c4d7' stroke-width='1.2' />"
        f"{''.join(polylines)}"
        f"{''.join(x_labels)}"
        "</svg>"
    )

    return f"<h2 class='chart-title'>{escape(title)}</h2><div class='legend'>{legend_html}</div>{svg}"


def _format_decimal(value: Any) -> str:
    """Format a scalar value for compact dashboard display."""

    if value is None:
        return "-"
    return f"{float(value):.3f}"


def _format_percentage(value: Any) -> str:
    """Format a ratio in `[0, 1]` as a percentage string."""

    if value is None:
        return "-"
    return f"{float(value) * 100.0:.1f}%"


def _format_bool(value: Any) -> str:
    """Format a boolean-like dashboard cell."""

    if value is None:
        return "-"
    return "yes" if bool(value) else "no"
