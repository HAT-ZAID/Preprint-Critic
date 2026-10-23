"""Shared helpers for the Re^2 LoRA artifact contract suite.

The artifact tests read only files committed to the repository. The dataset
contract tests additionally require the gitignored ``rc_project_datasets_V3/``
and skip when it is absent. Nothing here downloads the base model, mutates a
file, or needs a GPU.
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
