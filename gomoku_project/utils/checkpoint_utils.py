from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _warn_random_init(model_label: str, checkpoint_path: Path, reason: str) -> None:
    print(
        f"WARNING: continuing with random-initialized {model_label}. "
        f"{reason}. checkpoint={checkpoint_path}",
        file=sys.stderr,
        flush=True,
    )


def load_checkpoint_or_maybe_warn(
    *,
    trainer: Any,
    path: str | Path,
    model_label: str,
    cli_flag: str,
    allow_random_init: bool,
    reset_model: bool = False,
) -> tuple[Path, bool]:
    checkpoint_path = Path(path)

    if reset_model:
        if not allow_random_init:
            raise SystemExit(
                f"error: --reset-model skips {model_label} checkpoint loading.\n"
                f"Add --allow-random-init to confirm random initialization, or remove --reset-model."
            )
        _warn_random_init(
            model_label=model_label,
            checkpoint_path=checkpoint_path,
            reason="--reset-model was provided; checkpoint loading was skipped",
        )
        return checkpoint_path, False

    if checkpoint_path.exists():
        try:
            trainer.load(checkpoint_path)
        except Exception as exc:
            if allow_random_init:
                _warn_random_init(
                    model_label=model_label,
                    checkpoint_path=checkpoint_path,
                    reason=f"checkpoint exists but failed to load ({exc})",
                )
                return checkpoint_path, False
            raise SystemExit(
                f"error: failed to load {model_label} checkpoint from '{checkpoint_path}': {exc}\n"
                f"Provide a valid --{cli_flag} PATH or add --allow-random-init to continue with random initialization."
            )

        print(f"{model_label}_loaded_existing_model=True")
        return checkpoint_path, True

    if allow_random_init:
        _warn_random_init(
            model_label=model_label,
            checkpoint_path=checkpoint_path,
            reason="checkpoint was not found",
        )
        return checkpoint_path, False

    raise SystemExit(
        f"error: {model_label} checkpoint not found at '{checkpoint_path}'.\n"
        f"Provide --{cli_flag} PATH or add --allow-random-init to continue with random initialization."
    )
