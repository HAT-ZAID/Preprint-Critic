"""Contract tests for the six ``datasets`` splits in ``rc_project_datasets_V3``.

The suite pins the shapes that the notebook's data pipeline produced: split
sizes, column schemas, text (not list) payloads, paper-level leakage barriers
and the causal-LM label masking of the tokenized variants. It only samples
rows, so it stays fast and read-only.
"""

import json

import pytest
from datasets import load_from_disk

from conftest import REPO_ROOT

DATASETS_DIR = REPO_ROOT / "rc_project_datasets_V3"
SPLITS = ("train", "val", "test")
VARIANTS = ("untokenized", "tokenized")
EXPECTED_ROWS = {"train": 2800, "val": 280, "test": 280}
EXPECTED_TEST_PAPERS = 172

UNTOKENIZED_COLUMNS = ["unit_id", "paper_id", "whole_paper", "review_content", "meta"]
TOKENIZED_REQUIRED_COLUMNS = {"input_ids", "attention_mask", "labels"}
MAX_SEQUENCE_LENGTH = 18_432
SAMPLE_SIZE = 32

JSONL_PATH = DATASETS_DIR / "test_dataset_untokenized.jsonl"

pytestmark = pytest.mark.skipif(
    not DATASETS_DIR.is_dir(),
    reason="rc_project_datasets_V3/ is gitignored; regenerate it with the notebook",
)


@pytest.fixture(scope="session")
def datasets_by_variant():
    """Load every split once; ``datasets`` uses zero-copy Arrow backing."""
    loaded = {}
    for variant in VARIANTS:
        loaded[variant] = {}
        for split in SPLITS:
            path = DATASETS_DIR / f"{split}_dataset_{variant}"
            assert path.is_dir(), f"expected dataset directory {path} to exist"
            loaded[variant][split] = load_from_disk(str(path))
    return loaded


def evenly_spaced_indices(size, count):
    """Deterministic sample spanning the whole split, without touching every row."""
    assert size >= count, f"cannot sample {count} rows from a split of size {size}"
    return [index * size // count for index in range(count)]


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("split", SPLITS)
def test_row_counts(datasets_by_variant, variant, split):
    """Both variants of a split must have the same, documented number of units.

    Train/val/test of 2800/280/280 is the 90/5/5 unit split the results and the
    recorded 350 optimizer steps are derived from.
    """
    dataset = datasets_by_variant[variant][split]
    expected = EXPECTED_ROWS[split]
    assert len(dataset) == expected, (
        f"{split}_dataset_{variant}: expected {expected} rows, found {len(dataset)}"
    )


@pytest.mark.parametrize("split", SPLITS)
def test_untokenized_column_schema(datasets_by_variant, split):
    """Untokenized splits must expose exactly the five documented columns, in order.

    Downstream prompt construction indexes these names directly, so an extra or
    renamed column is a breaking change even when the rows still load.
    """
    dataset = datasets_by_variant["untokenized"][split]
    assert dataset.column_names == UNTOKENIZED_COLUMNS, (
        f"{split}_dataset_untokenized: expected columns {UNTOKENIZED_COLUMNS!r}, "
        f"found {dataset.column_names!r}"
    )


@pytest.mark.parametrize("split", SPLITS)
def test_untokenized_text_fields_are_non_empty_strings(datasets_by_variant, split):
    """``whole_paper``/``review_content`` must be real text, and ``meta`` a scored dict.

    This is a regression guard for a historical bug where those fields were
    stored as lists of strings, which trained the model on Python reprs instead
    of review prose. Sampling is deterministic and covers the whole split.
    """
    dataset = datasets_by_variant["untokenized"][split]
    sample = dataset.select(evenly_spaced_indices(len(dataset), SAMPLE_SIZE))

    for row_index, row in enumerate(sample):
        for field in ("whole_paper", "review_content"):
            value = row[field]
            assert isinstance(value, str), (
                f"{split}_dataset_untokenized row {row_index}: expected {field} to be "
                f"str, found {type(value).__name__}"
            )
            assert value.strip(), (
                f"{split}_dataset_untokenized row {row_index}: expected non-empty "
                f"{field}, found {value!r}"
            )

        meta = row["meta"]
        assert isinstance(meta, dict), (
            f"{split}_dataset_untokenized row {row_index}: expected meta to be dict, "
            f"found {type(meta).__name__}"
        )
        assert "final_score" in meta, (
            f"{split}_dataset_untokenized row {row_index}: expected meta to contain "
            f"'final_score', found keys {sorted(meta)!r}"
        )


@pytest.mark.parametrize(
    "left_split,right_split",
    [("train", "val"), ("train", "test"), ("val", "test")],
)
def test_no_paper_level_leakage(datasets_by_variant, left_split, right_split):
    """No paper may appear in two splits: the split is by paper, not by review unit.

    Sharing a paper would leak test prose into training and inflate every
    reported evaluation number.
    """
    untokenized = datasets_by_variant["untokenized"]
    left = set(untokenized[left_split]["paper_id"])
    right = set(untokenized[right_split]["paper_id"])
    overlap = left & right
    assert not overlap, (
        f"expected no paper_id shared between {left_split} and {right_split}, "
        f"found {len(overlap)} overlapping papers, e.g. {sorted(overlap)[:5]}"
    )


def test_test_split_paper_count(datasets_by_variant):
    """The test split must cover exactly 172 distinct papers across its 280 units.

    Several reviews of one paper are legitimate (multi-review papers); this
    pins how many papers the held-out comparison set draws from.
    """
    test_dataset = datasets_by_variant["untokenized"]["test"]
    papers = set(test_dataset["paper_id"])
    assert len(papers) == EXPECTED_TEST_PAPERS, (
        f"expected {EXPECTED_TEST_PAPERS} distinct papers in test_dataset_untokenized, "
        f"found {len(papers)}"
    )


@pytest.mark.parametrize("split", SPLITS)
def test_tokenized_required_columns(datasets_by_variant, split):
    """Tokenized splits must keep the causal-LM columns the trainer consumes."""
    dataset = datasets_by_variant["tokenized"][split]
    missing = TOKENIZED_REQUIRED_COLUMNS - set(dataset.column_names)
    assert not missing, (
        f"{split}_dataset_tokenized: expected columns to include "
        f"{sorted(TOKENIZED_REQUIRED_COLUMNS)}, missing {sorted(missing)}"
    )


@pytest.mark.parametrize("split", SPLITS)
def test_tokenized_rows_are_well_formed(datasets_by_variant, split):
    """Sampled tokenized rows must be length-aligned, capped at 18432 and prompt-masked.

    The first label is -100 (the prompt contributes no loss), at least one label
    supervises the completion, and every supervised label repeats the input id
    at that position. A mismatch means the model is trained against shifted or
    unrelated targets.
    """
    dataset = datasets_by_variant["tokenized"][split]
    sample = dataset.select(evenly_spaced_indices(len(dataset), SAMPLE_SIZE))

    for row_index, row in enumerate(sample):
        input_ids = row["input_ids"]
        attention_mask = row["attention_mask"]
        labels = row["labels"]

        assert len(input_ids) == len(attention_mask) == len(labels), (
            f"{split}_dataset_tokenized row {row_index}: expected input_ids/"
            f"attention_mask/labels to have equal lengths, found "
            f"{len(input_ids)}/{len(attention_mask)}/{len(labels)}"
        )
        assert len(input_ids) <= MAX_SEQUENCE_LENGTH, (
            f"{split}_dataset_tokenized row {row_index}: expected at most "
            f"{MAX_SEQUENCE_LENGTH} tokens, found {len(input_ids)}"
        )
        assert labels[0] == -100, (
            f"{split}_dataset_tokenized row {row_index}: expected the first label to "
            f"be -100 (masked prompt), found {labels[0]!r}"
        )

        supervised = [position for position, label in enumerate(labels) if label != -100]
        assert supervised, (
            f"{split}_dataset_tokenized row {row_index}: expected at least one "
            "supervised (non -100) label, found none"
        )

        mismatched = next(
            (
                position
                for position in supervised
                if labels[position] != input_ids[position]
            ),
            None,
        )
        assert mismatched is None, (
            f"{split}_dataset_tokenized row {row_index}: expected every supervised "
            f"label to equal input_ids at the same position, found label "
            f"{labels[mismatched]!r} != input_ids {input_ids[mismatched]!r} at "
            f"position {mismatched}"
        )


def _read_jsonl_objects(path):
    """Parse the JSONL export, ignoring a trailing blank line.

    Splitting is done on ``"\\n"`` rather than ``str.splitlines()``: paper text
    contains Unicode line separators (e.g. U+2028) inside JSON strings, and
    ``splitlines()`` would split mid-string and corrupt the document.
    """
    raw = path.read_text(encoding="utf-8")
    lines = [line for line in raw.split("\n") if line.strip()]
    return lines, [json.loads(line) for line in lines]


def test_jsonl_export_has_280_objects_with_text():
    """The JSONL export must mirror the test split's unit count and text payloads.

    It is the format external readers consume, so it has to stay in lockstep
    with the Arrow dataset rather than drifting into a stale copy.
    """
    assert JSONL_PATH.exists(), f"expected JSONL export at {JSONL_PATH}"
    _, objects = _read_jsonl_objects(JSONL_PATH)

    assert len(objects) == EXPECTED_ROWS["test"], (
        f"expected {EXPECTED_ROWS['test']} JSON objects in {JSONL_PATH.name}, "
        f"found {len(objects)}"
    )

    for line_number, obj in enumerate(objects, start=1):
        assert isinstance(obj, dict), (
            f"{JSONL_PATH.name} line {line_number}: expected a JSON object, "
            f"found {type(obj).__name__}"
        )
        keys = set(obj)
        assert keys <= set(UNTOKENIZED_COLUMNS), (
            f"{JSONL_PATH.name} line {line_number}: expected only untokenized "
            f"columns {UNTOKENIZED_COLUMNS!r}, found unexpected keys "
            f"{sorted(keys - set(UNTOKENIZED_COLUMNS))!r}"
        )
        for required in ("paper_id", "whole_paper", "review_content"):
            assert required in keys, (
                f"{JSONL_PATH.name} line {line_number}: expected key {required!r}, "
                f"found keys {sorted(keys)!r}"
            )
            assert isinstance(obj[required], str) and obj[required].strip(), (
                f"{JSONL_PATH.name} line {line_number}: expected non-empty str for "
                f"{required!r}, found {type(obj[required]).__name__}"
            )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known artifact deviation: test_dataset_untokenized.jsonl carries only "
        "paper_id/whole_paper/review_content; unit_id and meta are absent although "
        "the dataset itself has all five columns."
    ),
)
def test_jsonl_export_keeps_all_five_untokenized_keys():
    """Spec contract: every JSONL object should carry all five untokenized columns.

    The shipped export drops ``unit_id`` and ``meta``. This test records the
    expected contract and is expected to fail until the export is regenerated;
    once fixed, the strict xfail will surface as a failure and can be removed.
    """
    assert JSONL_PATH.exists(), f"expected JSONL export at {JSONL_PATH}"
    _, objects = _read_jsonl_objects(JSONL_PATH)
    assert objects, f"expected JSON objects in {JSONL_PATH.name}, found none"

    expected_keys = set(UNTOKENIZED_COLUMNS)
    for line_number, obj in enumerate(objects, start=1):
        assert set(obj) == expected_keys, (
            f"{JSONL_PATH.name} line {line_number}: expected the five untokenized "
            f"keys {sorted(expected_keys)!r}, found {sorted(obj)!r}"
        )
