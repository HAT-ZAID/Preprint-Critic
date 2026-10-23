"""Contract tests for the baseline-vs-fine-tune comparison artifacts.

Three JSON files describe the same ten held-out papers, each from a different
angle (paired comparison, baseline-only run, fine-tune-only run). These tests
protect their schema, the fact that they describe the identical ten papers in
the same order, and the textual identity of the reviews they share.
"""

import pytest
from tokenizers import Tokenizer

from conftest import REPO_ROOT, load_json

COMPARISON_DIR = REPO_ROOT / "comparison"
COMPARISON_PATH = COMPARISON_DIR / "comparison_v3a.json"
BASELINE_PATH = COMPARISON_DIR / "baseline_results_v3a.json"
FINETUNE_PATH = REPO_ROOT / "rc_project_datasets_V3" / "finetune_results_v3a.json"
EXTRACTED_TEXTS_PATH = COMPARISON_DIR / "extracted_paper_texts.json"
TOKENIZER_PATH = REPO_ROOT / "rc_project_adapter_V3" / "tokenizer.json"

EXPECTED_ROWS = 10
MAX_REVIEW_TOKENS = 16_384

COMPARISON = load_json("comparison/comparison_v3a.json")
BASELINE = load_json("comparison/baseline_results_v3a.json")
FINETUNE = load_json("rc_project_datasets_V3/finetune_results_v3a.json")

COMPARISON_KEYS = {
    "paper_id",
    "base_review",
    "fine_review",
    "reference_review",
    "base_output_tokens",
    "fine_output_tokens",
    "base_hit_cap",
    "fine_hit_cap",
}
BASELINE_KEYS = {
    "paper_id",
    "base_review",
    "reference_review",
    "input_tokens",
    "base_output_tokens",
    "base_hit_cap",
}
FINETUNE_KEYS = {
    "paper_id",
    "fine_review",
    "reference_review",
    "input_tokens",
    "fine_output_tokens",
    "fine_hit_cap",
}

FILES = [
    ("comparison_v3a.json", COMPARISON, COMPARISON_KEYS),
    ("baseline_results_v3a.json", BASELINE, BASELINE_KEYS),
    ("finetune_results_v3a.json", FINETUNE, FINETUNE_KEYS),
]


@pytest.mark.parametrize(
    "filename,rows,expected_keys",
    FILES,
    ids=[filename for filename, _, _ in FILES],
)
def test_schema_and_row_count(filename, rows, expected_keys):
    """Each artifact must be a ten-row list with exactly its documented schema.

    The column sets were produced by different notebook cells; if one drifts,
    the three files can no longer be cross-checked row for row.
    """
    assert isinstance(rows, list), (
        f"{filename}: expected a JSON list, found {type(rows).__name__}"
    )
    assert len(rows) == EXPECTED_ROWS, (
        f"{filename}: expected {EXPECTED_ROWS} rows, found {len(rows)}"
    )

    for row_index, row in enumerate(rows):
        assert isinstance(row, dict), (
            f"{filename} row {row_index}: expected a JSON object, "
            f"found {type(row).__name__}"
        )
        assert set(row) == expected_keys, (
            f"{filename} row {row_index}: expected keys {sorted(expected_keys)!r}, "
            f"found {sorted(row)!r}"
        )


def test_same_papers_in_same_order():
    """All three artifacts must cover the identical ten papers in identical order.

    Row-wise comparisons elsewhere in this module are only meaningful under this
    ordering guarantee.
    """
    ids = [[row["paper_id"] for row in rows] for _, rows, _ in FILES]
    labels = [filename for filename, _, _ in FILES]

    for filename, paper_ids in zip(labels, ids):
        assert len(set(paper_ids)) == EXPECTED_ROWS, (
            f"{filename}: expected {EXPECTED_ROWS} distinct paper ids, "
            f"found {len(set(paper_ids))}"
        )

    assert ids[0] == ids[1] == ids[2], (
        "expected comparison/baseline/finetune artifacts to list the same paper ids "
        f"in the same order, found {dict(zip(labels, ids))!r}"
    )


@pytest.mark.parametrize("row_index", range(EXPECTED_ROWS))
def test_reference_reviews_are_byte_identical(row_index):
    """The human reference review must be the same text in all three artifacts.

    A differing reference means the baseline or the fine-tune was scored against
    a different target, which invalidates the comparison.
    """
    paper_id = COMPARISON[row_index]["paper_id"]
    reference = COMPARISON[row_index]["reference_review"]

    for filename, rows in (
        ("baseline_results_v3a.json", BASELINE),
        ("finetune_results_v3a.json", FINETUNE),
    ):
        other = rows[row_index]["reference_review"]
        assert other == reference, (
            f"paper {paper_id!r}: expected reference_review in {filename} to be "
            f"byte-identical to comparison_v3a.json, found a different string "
            f"({len(other)} vs {len(reference)} chars)"
        )


@pytest.mark.parametrize("row_index", range(EXPECTED_ROWS))
def test_shared_reviews_are_identical(row_index):
    """The duplicated review columns must agree with the paired comparison file.

    ``comparison_v3a.json`` embeds both generations; the per-arm files repeat
    one each. Divergence would mean the published comparison shows text that no
    arm actually produced.
    """
    paper_id = COMPARISON[row_index]["paper_id"]

    pairs = [
        ("base_review", BASELINE[row_index], "baseline_results_v3a.json"),
        ("fine_review", FINETUNE[row_index], "finetune_results_v3a.json"),
    ]
    for field, other_row, filename in pairs:
        expected = COMPARISON[row_index][field]
        found = other_row[field]
        assert found == expected, (
            f"paper {paper_id!r}: expected {field} in {filename} to match "
            f"comparison_v3a.json, found a different string "
            f"({len(found)} vs {len(expected)} chars)"
        )


@pytest.mark.parametrize(
    "filename,rows,field",
    [
        ("comparison_v3a.json", COMPARISON, "base_review"),
        ("comparison_v3a.json", COMPARISON, "fine_review"),
        ("comparison_v3a.json", COMPARISON, "reference_review"),
        ("baseline_results_v3a.json", BASELINE, "base_review"),
        ("baseline_results_v3a.json", BASELINE, "reference_review"),
        ("finetune_results_v3a.json", FINETUNE, "fine_review"),
        ("finetune_results_v3a.json", FINETUNE, "reference_review"),
    ],
    ids=[
        "comparison-base",
        "comparison-fine",
        "comparison-reference",
        "baseline-base",
        "baseline-reference",
        "finetune-fine",
        "finetune-reference",
    ],
)
def test_reviews_are_non_empty(filename, rows, field):
    """Every stored review must be non-empty prose, never a blank generation.

    An empty fine-tune review would otherwise look like a valid row in the
    comparison table.
    """
    for row_index, row in enumerate(rows):
        value = row[field]
        assert isinstance(value, str), (
            f"{filename} row {row_index}: expected {field} to be str, "
            f"found {type(value).__name__}"
        )
        assert value.strip(), (
            f"{filename} row {row_index}: expected non-empty {field}, found {value!r}"
        )


def test_reference_reviews_are_distinct():
    """The ten reference reviews must be ten different texts.

    Duplicated references would indicate a join bug that silently re-scored the
    same paper under several ids.
    """
    references = [row["reference_review"] for row in COMPARISON]
    assert len(set(references)) == EXPECTED_ROWS, (
        f"expected {EXPECTED_ROWS} distinct reference reviews, "
        f"found {len(set(references))}"
    )


@pytest.mark.parametrize(
    "filename,rows,fields",
    [
        ("comparison_v3a.json", COMPARISON, ("base_hit_cap", "fine_hit_cap")),
        ("baseline_results_v3a.json", BASELINE, ("base_hit_cap",)),
        ("finetune_results_v3a.json", FINETUNE, ("fine_hit_cap",)),
    ],
    ids=["comparison", "baseline", "finetune"],
)
def test_no_generation_hit_the_cap(filename, rows, fields):
    """No recorded generation may have been truncated by the token cap.

    These ten reviews are the qualitative evidence for the fine-tune; a capped
    generation is incomplete and cannot be compared fairly.
    """
    for row_index, row in enumerate(rows):
        for field in fields:
            assert row[field] is False, (
                f"{filename} row {row_index} (paper {row['paper_id']!r}): "
                f"expected {field} to be False, found {row[field]!r}"
            )


@pytest.fixture(scope="module")
def tokenizer():
    """The adapter's own tokenizer, used to measure review lengths."""
    assert TOKENIZER_PATH.exists(), f"expected tokenizer at {TOKENIZER_PATH}"
    return Tokenizer.from_file(str(TOKENIZER_PATH))


@pytest.mark.parametrize("row_index", range(EXPECTED_ROWS))
def test_reviews_fit_the_context_window(tokenizer, row_index):
    """Every stored review must re-encode to fewer than 16,384 tokens.

    The recorded ``*_output_tokens`` fields are padded batch widths (a known
    notebook bug), so they are deliberately not asserted against real counts.
    Re-encoding the text with the shipped tokenizer is the honest check that
    nothing exceeded the model's context window.
    """
    row = COMPARISON[row_index]
    for field in ("base_review", "fine_review"):
        token_count = len(tokenizer.encode(row[field], add_special_tokens=False).ids)
        assert token_count < MAX_REVIEW_TOKENS, (
            f"paper {row['paper_id']!r}: expected {field} to encode to fewer than "
            f"{MAX_REVIEW_TOKENS} tokens, found {token_count}"
        )


@pytest.mark.skipif(
    not EXTRACTED_TEXTS_PATH.exists(),
    reason="comparison/extracted_paper_texts.json is not shipped in this checkout",
)
def test_extracted_paper_texts_cover_every_compared_paper():
    """The 79 MB paper-text cache must contain every paper in the comparison.

    It is only needed to regenerate the reviews on a machine with the base
    model, but a missing key would make the ten published rows irreproducible.
    The test skips cleanly when the large cache is absent.
    """
    extracted = load_json("comparison/extracted_paper_texts.json")
    assert isinstance(extracted, dict), (
        "expected extracted_paper_texts.json to be a JSON object mapping paper_id "
        f"to text, found {type(extracted).__name__}"
    )

    missing = [row["paper_id"] for row in COMPARISON if row["paper_id"] not in extracted]
    assert not missing, (
        "expected every comparison paper_id to be a key in "
        f"extracted_paper_texts.json, missing {missing!r}"
    )

    for paper_id in [row["paper_id"] for row in COMPARISON]:
        text = extracted[paper_id]
        assert isinstance(text, str) and text.strip(), (
            f"paper {paper_id!r}: expected non-empty paper text in "
            f"extracted_paper_texts.json, found {type(text).__name__}"
        )
