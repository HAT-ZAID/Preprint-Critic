"""Shared helpers for the Re^2 LoRA artifact contract suite.

The suite reads the artifacts that ship with the repository only. It never
downloads the base model, never mutates a file, and never needs a GPU.
"""

import json
from pathlib import Path

#: Repository root, resolved from this file so tests work from any cwd.
REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json(relative_path: str):
    """Load a UTF-8 JSON document located at ``relative_path`` under ``REPO_ROOT``."""
    path = REPO_ROOT / relative_path
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
