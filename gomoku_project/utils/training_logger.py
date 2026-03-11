from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def append_csv_record(path: str | Path, record: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    with output_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(record)
    return output_path


def append_jsonl_record(path: str | Path, record: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return output_path
