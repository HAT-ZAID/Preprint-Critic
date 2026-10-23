"""Tests for ``scripts/evaluate.py`` and the committed V3a metrics report."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import evaluate  # noqa: E402

METRICS_PATH = REPO_ROOT / "comparison" / "metrics_v3a.json"
SCRIPT_PATH = REPO_ROOT / "scripts" / "evaluate.py"


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------
def test_rouge_l_identical_texts_is_one():
    text = "The paper proposes a new pruning method for image super resolution."
    assert evaluate.rouge_l(text, text) == pytest.approx(1.0)


def test_rouge_l_disjoint_texts_is_zero():
    assert evaluate.rouge_l("alpha beta gamma", "delta epsilon zeta") == 0.0


def test_rouge_l_partial_overlap_is_strictly_between():
    candidate = "the cat sat on the mat in the sun"
    reference = "the cat sat on a mat near the door"
    score = evaluate.rouge_l(candidate, reference)
    assert 0.0 < score < 1.0


def test_rouge_l_empty_reference_is_zero():
    assert evaluate.rouge_l("anything at all", "") == 0.0


# ---------------------------------------------------------------------------
# Repetition
# ---------------------------------------------------------------------------
def test_ngram_repeat_rate_repeated_text_beats_distinct_text():
    repeated = "the cat sat on the mat the cat sat on the mat"
    distinct = "one two three four five six seven eight nine ten"
    repeated_rate = evaluate.ngram_repeat_rate(repeated)
    distinct_rate = evaluate.ngram_repeat_rate(distinct)
    assert repeated_rate > distinct_rate
    assert repeated_rate > 0.0
    assert distinct_rate == 0.0


def test_ngram_repeat_rate_short_text_is_zero():
    assert evaluate.ngram_repeat_rate("too short here") == 0.0


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------
def test_citations_normalises_table_and_figure():
    found = evaluate.citations("See Table 2 and Fig. 3 for details.")
    assert "Table 2" in found
    assert "Figure 3" in found


def test_citations_ignores_section_without_number():
    assert evaluate.citations("This section discusses the method.") == []


def test_grounded_citations_only_counts_mentions_in_the_paper():
    found = evaluate.citations("See Table 2 and Figure 3.")
    paper = "Results in Table 2 show the improvement."
    grounded = evaluate.grounded_citations(found, paper)
    assert grounded == ["Table 2"]


def test_grounded_citations_empty_when_paper_does_not_mention_them():
    found = evaluate.citations("See Table 2.")
    assert evaluate.grounded_citations(found, "No numbered floats here.") == []


# ---------------------------------------------------------------------------
# Template keys
# ---------------------------------------------------------------------------
def test_template_keys_handles_bold_summary_and_heading_strengths():
    text = (
        "**summary_of_the_paper:** The paper studies pruning.\n"
        "## Strengths\n"
        "- solid experiments\n"
        "### Weaknesses\n"
        "- limited baselines\n"
        "Questions:\n"
        "1. How many seeds were used?\n"
    )
    keys = evaluate.template_keys_present(text)
    assert keys == list(evaluate.TEMPLATE_KEYS)
    assert evaluate.has_all_four_template_keys(text) is True


def test_template_keys_ignores_plain_prose():
    text = "This review has no template headings at all."
    assert evaluate.template_keys_present(text) == []
    assert evaluate.has_all_four_template_keys(text) is False


# ---------------------------------------------------------------------------
# CLI integration on a small synthetic fixture
# ---------------------------------------------------------------------------
def _write_wordlevel_tokenizer(path: Path) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(path))


def _fixture_rows():
    base = (
        "**Peer Review of the paper**\n"
        "The authors study a problem and report results in Table 2.\n"
    )
    fine = (
        "summary_of_the_paper: The paper studies a problem.\n"
        "strengths: The results in Section 3 are clear.\n"
        "weaknesses: The evaluation is narrow.\n"
        "questions: How many seeds were used?\n"
    )
    reference = (
        "summary_of_the_paper: The paper studies a problem.\n"
        "strengths: The results in Section 3 are clear.\n"
        "weaknesses: The evaluation is narrow.\n"
        "questions: How many seeds were used?\n"
    )
    return [
        {
            "paper_id": pid,
            "base_review": base,
            "fine_review": fine,
            "reference_review": reference,
            "base_output_tokens": 64,
            "fine_output_tokens": 32,
            "base_hit_cap": False,
            "fine_hit_cap": False,
        }
        for pid in ("paper_a", "paper_b", "paper_c")
    ]


def test_cli_writes_report_for_three_row_fixture(tmp_path):
    comparison_path = tmp_path / "comparison.json"
    papers_path = tmp_path / "papers.json"
    tokenizer_path = tmp_path / "tokenizer.json"
    out_path = tmp_path / "metrics.json"

    rows = _fixture_rows()
    comparison_path.write_text(json.dumps(rows), encoding="utf-8")
    papers_path.write_text(
        json.dumps(
            {
                "paper_a": "Results are shown in Table 2 and Section 3.",
                "paper_b": "Nothing numbered is mentioned here.",
                "paper_c": "Section 3 and Table 2 are referenced.",
            }
        ),
        encoding="utf-8",
    )
    _write_wordlevel_tokenizer(tokenizer_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--comparison",
            str(comparison_path),
            "--papers",
            str(papers_path),
            "--tokenizer",
            str(tokenizer_path),
            "--out",
            str(out_path),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert out_path.exists()

    report = json.loads(out_path.read_text(encoding="utf-8"))

    assert report["script"] == "scripts/evaluate.py"
    assert set(report["inputs"]) == {"comparison", "papers", "tokenizer"}
    assert report["n_samples"] == 3
    assert set(report["definitions"]) == {
        "rouge_l",
        "template_keys",
        "citation_grounding",
        "ngram_repeat_rate",
    }
    assert set(report["arms"]) == {"base", "fine", "reference"}
    assert set(report["head_to_head"]) == {"fine_wins", "base_wins", "ties"}
    assert len(report["per_sample"]) == 3
    assert isinstance(report["notes"], str) and report["notes"]

    for arm in ("base", "fine", "reference"):
        values = report["arms"][arm]
        assert values["n"] == 3
        assert len(values["real_output_tokens"]) == 3
        for field in ("mean_words", "median_words", "mean_rouge_l", "min_rouge_l",
                      "max_rouge_l", "template_count", "has_all_four_count",
                      "citation_total", "grounded_citation_total",
                      "mean_ngram_repeat_rate", "mean_real_output_tokens"):
            assert field in values

    for sample in report["per_sample"]:
        assert set(sample) >= {"paper_id", "base", "fine", "reference"}
        for arm in ("base", "fine", "reference"):
            assert isinstance(sample[arm]["words"], int)
            assert isinstance(sample[arm]["chars"], int)
            assert isinstance(sample[arm]["rouge_l"], float)
            assert isinstance(sample[arm]["has_all_four"], bool)
            assert isinstance(sample[arm]["citations"], list)
            assert 0.0 <= sample[arm]["ngram_repeat_rate"] <= 1.0

    # Same rows written twice must give byte-identical output.
    rerun_path = tmp_path / "metrics_rerun.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--comparison",
            str(comparison_path),
            "--papers",
            str(papers_path),
            "--tokenizer",
            str(tokenizer_path),
            "--out",
            str(rerun_path),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert out_path.read_bytes() == rerun_path.read_bytes()


# ---------------------------------------------------------------------------
# Committed artefact
# ---------------------------------------------------------------------------
def test_committed_metrics_report_exists_with_ten_samples():
    if not METRICS_PATH.exists():
        pytest.skip(
            "comparison/metrics_v3a.json is not present; "
            "run `python scripts/evaluate.py` to generate it"
        )

    report = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    assert report["n_samples"] == 10
    assert set(report["arms"]) == {"base", "fine", "reference"}
    assert len(report["per_sample"]) == 10
    assert len({sample["paper_id"] for sample in report["per_sample"]}) == 10

    for arm in ("base", "fine", "reference"):
        values = report["arms"][arm]
        assert values["n"] == 10
        assert len(values["real_output_tokens"]) == 10
        assert 0.0 <= values["mean_rouge_l"] <= 1.0
        assert 0.0 <= values["mean_ngram_repeat_rate"] <= 1.0

    assert report["arms"]["reference"]["mean_rouge_l"] == pytest.approx(1.0)
    assert report["head_to_head"]["fine_wins"] + report["head_to_head"]["base_wins"] + \
        report["head_to_head"]["ties"] == 10
