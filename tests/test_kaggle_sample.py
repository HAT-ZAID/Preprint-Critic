"""Contract test for the committed 25-paper Kaggle inference sample.

The Kaggle notebook downloads this file when no test-split dataset is attached,
so its schema and paper-level uniqueness are part of the runnable surface.
"""

import json

from conftest import REPO_ROOT

SAMPLE_PATH = REPO_ROOT / "comparison" / "kaggle_sample_papers.jsonl"
EXPECTED_ROWS = 25


def test_sample_schema():
    rows = [json.loads(line) for line in SAMPLE_PATH.open(encoding="utf-8") if line.strip()]
    assert len(rows) == EXPECTED_ROWS
    assert len({row["paper_id"] for row in rows}) == EXPECTED_ROWS
    for row in rows:
        assert set(row) == {"paper_id", "whole_paper", "review_content"}
        assert isinstance(row["paper_id"], str) and row["paper_id"]
        assert isinstance(row["whole_paper"], str) and row["whole_paper"].strip()
        assert isinstance(row["review_content"], str) and len(row["review_content"]) >= 100
