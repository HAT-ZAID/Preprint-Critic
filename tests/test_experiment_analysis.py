"""Focused tests for the Phase 1 non-generative comparison.

Only a handful of fixed-value checks are intentional: this suite is a guard
rail, not an exhaustive matrix.  No network call is made and no test requires
the Ollama server to be running.
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.analysis.load_runs import (  # noqa: E402
    COMMON_10,
    EXPECTED_RUNS,
    EXPERIMENTS_DIR,
    expected_arms,
    load_all,
    prompt_family,
    validate_all,
)
from experiments.analysis.local_metrics import (  # noqa: E402
    OllamaEmbedder,
    _cache_key,
    add_all_reference_metrics,
    build_metrics,
    chunk_text,
    count_words,
    length_fit,
    load_all_references,
    load_papers,
    loop_stats,
    normalise,
    repeat_stats,
    structure_flags,
    structure_score,
)

REVIEW_COLUMNS = [
    "paper_id",
    "arm",
    "review",
    "reference_review",
    "prompt_tokens",
    "output_tokens",
    "seconds",
    "words",
    "rouge_l",
    "repeat5",
    "hit_cap",
    "seed",
    "max_new_tokens",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "no_repeat_ngram_size",
]


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _review_row(paper_id: str, arm: str, **overrides) -> dict:
    row = {
        "paper_id": paper_id,
        "arm": arm,
        "review": f"Review of {paper_id} by the {arm} arm.",
        "reference_review": f"Human reference review of {paper_id}.",
        "prompt_tokens": 1000,
        "output_tokens": 150,
        "seconds": 1.5,
        "words": 120,
        "rouge_l": 0.1,
        "repeat5": 0.0,
        "hit_cap": "False",
        "seed": 42,
        "max_new_tokens": 4096,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.15,
        "no_repeat_ngram_size": 0,
    }
    row.update(overrides)
    return row


def _summary_rows(rows: list[dict]) -> list[dict]:
    summary = []
    for arm in sorted({row["arm"] for row in rows}):
        arm_rows = [row for row in rows if row["arm"] == arm]
        n = len(arm_rows)
        summary.append(
            {
                "arm": arm,
                "n": n,
                "mean_words": sum(float(row["words"]) for row in arm_rows) / n,
                "mean_output_tokens": sum(
                    float(row["output_tokens"]) for row in arm_rows
                )
                / n,
                "mean_rouge_l": sum(float(row["rouge_l"]) for row in arm_rows) / n,
                "mean_repeat5": sum(float(row["repeat5"]) for row in arm_rows) / n,
                "mean_seconds": sum(float(row["seconds"]) for row in arm_rows) / n,
                "cap_reached": sum(
                    1
                    for row in arm_rows
                    if str(row["hit_cap"]).strip().lower() in {"true", "1"}
                ),
            }
        )
    return summary


def _csv_text(columns: list[str], rows: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _write_run(
    root: Path,
    run: str,
    paper_ids: list[str],
    rows: list[dict],
    *,
    arms: tuple[str, ...] | None = None,
    config_overrides: dict | None = None,
) -> Path:
    config = {
        "base_model": "Qwen/Qwen3-4B-Instruct-2507",
        "adapter_dir": "adapter",
        "n_papers": len(paper_ids),
        "paper_ids": list(paper_ids),
        "max_paper_tokens": 17500,
        "max_new_tokens": 4096,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.15,
        "no_repeat_ngram_size": 0,
        "do_sample": True,
    }
    config.update(config_overrides or {})
    summary_columns = [
        "arm",
        "n",
        "mean_words",
        "mean_output_tokens",
        "mean_rouge_l",
        "mean_repeat5",
        "mean_seconds",
        "cap_reached",
    ]
    path = root / f"{run}.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("reviews.csv", _csv_text(REVIEW_COLUMNS, rows))
        archive.writestr(
            "summary_by_arm.csv", _csv_text(summary_columns, _summary_rows(rows))
        )
        archive.writestr("reviews.jsonl", "")
        archive.writestr("run_config.json", json.dumps(config))
    return path


def _default_rows(paper_ids: list[str], arms: tuple[str, ...]) -> list[dict]:
    return [
        _review_row(paper_id, arm) for arm in arms for paper_id in paper_ids
    ]


def _synthetic_root(
    tmp_path: Path,
    *,
    rows_mutator=None,
    arms: tuple[str, ...] = ("base", "adapter"),
    papers: tuple[str, ...] = ("P1", "P2"),
) -> Path:
    rows = _default_rows(list(papers), arms)
    if rows_mutator is not None:
        rows = rows_mutator(rows)
    _write_run(tmp_path, "original_prompt_synthetic", list(papers), rows, arms=arms)
    return tmp_path


# ---------------------------------------------------------------------------
# Real archives
# ---------------------------------------------------------------------------


real_data = pytest.mark.skipif(
    not EXPERIMENTS_DIR.is_dir(),
    reason=f"external experiment directory is absent: {EXPERIMENTS_DIR}",
)


def test_common_ten_matches_protocol_order() -> None:
    assert COMMON_10 == (
        "UgNQM-LcVpN",
        "N78I92JIqOJ",
        "wClmeg9u7G",
        "pW2Q2xLwIMD",
        "fdyxLGHE6bU",
        "g4szfsQUdy3",
        "rylq7mF8IS",
        "Jpxd93u2vK-",
        "mF5tmqUfdsw",
        "dZEZu7zxJBF",
    )


def test_prompt_family_and_expected_arms() -> None:
    assert prompt_family("a_bit_altered_prompt_60_papers") == "a_bit"
    assert prompt_family("new_prompt_with_conservative_settings") == "new"
    assert prompt_family("very_big_prompt_very_high_temp") == "very_big"
    assert expected_arms("a_bit_altered_prompt_conservative_settings") == ("adapter",)
    assert expected_arms("original_prompt_original_settings") == ("adapter", "base")


@real_data
def test_real_archives_validate_cleanly() -> None:
    assert validate_all() == []


@real_data
def test_real_load_shape() -> None:
    runs, reviews = load_all()
    assert len(runs) == 15
    assert len(reviews) == 960
    assert set(reviews["run"]) == set(EXPECTED_RUNS)
    assert reviews.duplicated(["run", "paper_id", "arm"]).sum() == 0
    assert int(runs["n_papers"].sum()) == 580


# ---------------------------------------------------------------------------
# Validation failures on synthetic data
# ---------------------------------------------------------------------------


def test_synthetic_fixture_is_valid(tmp_path: Path) -> None:
    root = _synthetic_root(tmp_path)
    assert validate_all(root, expected_runs=["original_prompt_synthetic"]) == []
    runs, reviews = load_all(root, expected_runs=["original_prompt_synthetic"])
    assert len(runs) == 1
    assert len(reviews) == 4


def test_duplicate_rows_are_reported(tmp_path: Path) -> None:
    root = _synthetic_root(
        tmp_path, rows_mutator=lambda rows: rows + [dict(rows[0])]
    )
    errors = "\n".join(validate_all(root, expected_runs=["original_prompt_synthetic"]))
    assert "duplicate" in errors
    assert "row count" in errors


def test_wrong_row_count_is_reported(tmp_path: Path) -> None:
    root = _synthetic_root(tmp_path, rows_mutator=lambda rows: rows[:-1])
    errors = "\n".join(validate_all(root, expected_runs=["original_prompt_synthetic"]))
    assert "row count" in errors


def test_wrong_paper_order_is_reported(tmp_path: Path) -> None:
    root = _synthetic_root(tmp_path, rows_mutator=lambda rows: list(reversed(rows)))
    errors = "\n".join(validate_all(root, expected_runs=["original_prompt_synthetic"]))
    assert "paper order" in errors


def test_reference_drift_is_reported(tmp_path: Path) -> None:
    second_rows = _default_rows(["P1", "P2"], ("base", "adapter"))
    second_rows[0]["reference_review"] = "A different human reference."
    _write_run(tmp_path, "original_prompt_synthetic_two", ["P1", "P2"], second_rows)
    root = _synthetic_root(tmp_path)
    errors = "\n".join(
        validate_all(
            root,
            expected_runs=["original_prompt_synthetic", "original_prompt_synthetic_two"],
        )
    )
    assert "reference review differs" in errors


def test_missing_archive_is_reported(tmp_path: Path) -> None:
    errors = "\n".join(validate_all(tmp_path, expected_runs=["original_prompt_absent"]))
    assert "missing archive" in errors


# ---------------------------------------------------------------------------
# Local lexical metrics
# ---------------------------------------------------------------------------


def test_normalise_and_count_words() -> None:
    assert normalise("  a\n b\t c ") == "a b c"
    assert count_words(" one\ntwo   three ") == 3
    assert count_words("   ") == 0


def test_repeat_stats_fixed_value() -> None:
    assert repeat_stats("a b c a b c a b c", 3) == pytest.approx(4 / 7)
    assert repeat_stats("a b c", 5) == 0.0


def test_loop_stats_detects_verbatim_loop() -> None:
    words = [f"w{index}" for index in range(10)]
    text = " ".join(words * 4)
    rate, max_count, degenerate = loop_stats(text, 10)
    assert max_count == 4
    assert degenerate is True
    assert rate > 0.5


def test_loop_stats_accepts_normal_prose() -> None:
    text = " ".join(f"word{index}" for index in range(120))
    rate, max_count, degenerate = loop_stats(text, 10)
    assert max_count == 1
    assert rate == 0.0
    assert degenerate is False


def test_structure_flags_fixed_values() -> None:
    text = (
        "Summary: the paper studies X. Strengths: clear writing. "
        "Weaknesses: no baselines in Section 3. Table 2 and Figure 1 support it. "
        "Equation (4) is unclear. Verdict: major revision. "
        "I haven't read the appendix."
    )
    flags = structure_flags(text)
    assert flags["summary_opening"] is True
    assert flags["has_strengths"] is True
    assert flags["has_weaknesses"] is True
    assert flags["has_verdict"] is True
    assert flags["section_refs"] == 1
    assert flags["table_refs"] == 1
    assert flags["figure_refs"] == 1
    assert flags["equation_refs"] == 1
    assert flags["location_refs"] == 4
    assert flags["location_any"] is True
    assert flags["meta_commentary"] is True
    assert "i haven't read" in flags["meta_phrases"]


def test_structure_flags_handles_section_sign_and_apostrophes() -> None:
    flags = structure_flags("See \u00a73 and \u00a74.2 for details. I haven\u2019t read the proof.")
    assert flags["section_refs"] == 2
    assert flags["location_any"] is True
    assert flags["meta_commentary"] is True

    # "as an ai" must not fire on ordinary prose such as "serves as an aid".
    benign = structure_flags("The ablation serves as an aid to readers.")
    assert benign["meta_commentary"] is False
    assert not benign["meta_phrases"]


def test_length_fit_and_structure_score() -> None:
    assert length_fit(99, 99) == pytest.approx(1.0)
    assert length_fit(999, 99) < 0.3
    assert length_fit(0, 99) == 0.0
    full = {
        "summary_opening": True,
        "has_strengths": True,
        "has_weaknesses": True,
        "location_any": True,
    }
    empty = {key: False for key in full}
    assert structure_score(full) == pytest.approx(1.0)
    assert structure_score(empty) == 0.0


def test_chunk_text_respects_cap_and_size() -> None:
    chunks = chunk_text("x" * 10_000, size=100, overlap=10, max_chunks=3)
    assert len(chunks) == 3
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert chunk_text("   ") == []
    with pytest.raises(ValueError):
        chunk_text("abc", size=10, overlap=10)


# ---------------------------------------------------------------------------
# Embedding client and per-review metric table
# ---------------------------------------------------------------------------


def test_cache_key_is_stable_and_model_specific() -> None:
    assert _cache_key("model", "text") == _cache_key("model", "text")
    assert _cache_key("model", "text") != _cache_key("model", "other")
    assert _cache_key("model", "text") != _cache_key("other", "text")


def test_embedding_cache_avoids_second_request(tmp_path: Path, monkeypatch) -> None:
    embedder = OllamaEmbedder(cache_path=tmp_path / "cache.sqlite")
    calls: list[int] = []

    def fake_post(texts):
        calls.append(len(texts))
        base = np.arange(len(texts) * 4, dtype=np.float32).reshape(len(texts), 4)
        return base + 1.0

    monkeypatch.setattr(embedder, "_post_batch", fake_post)
    first = embedder.embed_many(["alpha", "beta"])
    second = embedder.embed_many(["alpha", "beta"])
    embedder.close()

    assert calls == [2]
    assert set(first) == {"alpha", "beta"}
    for vector in second.values():
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)


class _FakeEmbedder:
    """Deterministic stand-in for ``OllamaEmbedder`` in unit tests."""

    def embed_many(self, texts, progress=None):
        vectors = {}
        for text in dict.fromkeys(str(text) for text in texts):
            magnitude = float(len(text) % 5 + 1)
            vector = np.array([magnitude, 1.0, 0.5, 2.0], dtype=np.float32)
            vectors[text] = vector / np.linalg.norm(vector)
        return vectors


def _metric_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run": "r",
                "arm": "adapter",
                "paper_id": "P1",
                "review": "Summary: good. Section 2 is clear.",
                "reference_review": "A reference review.",
                "output_tokens": 100,
                "seconds": 1.0,
                "hit_cap": False,
                "rouge_l": 0.1,
                "repeat5": 0.0,
            },
            {
                "run": "r",
                "arm": "adapter",
                "paper_id": "P2",
                "review": "Weaknesses: none. Table 1 is small.",
                "reference_review": "Another reference.",
                "output_tokens": 120,
                "seconds": 1.2,
                "hit_cap": True,
                "rouge_l": 0.2,
                "repeat5": 0.01,
            },
        ]
    )


def _metric_papers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "paper_id": ["P1", "P2"],
            "whole_paper": ["paper one " * 300, "paper two " * 300],
        }
    )


def test_build_metrics_without_embedder() -> None:
    metrics = build_metrics(_metric_frame(), _metric_papers())
    assert len(metrics) == 2
    assert metrics["emb_ref_cos"].isna().all()
    assert metrics["emb_paper_cos"].isna().all()
    assert metrics["structure_score"].tolist() == pytest.approx([0.6, 0.6])
    assert metrics["hit_cap"].tolist() == [False, True]
    assert metrics["location_any"].all()


def test_build_metrics_with_embedder() -> None:
    metrics = build_metrics(_metric_frame(), _metric_papers(), embedder=_FakeEmbedder())
    assert metrics["emb_ref_cos"].notna().all()
    assert metrics["emb_paper_cos"].notna().all()
    assert metrics["emb_paper_cos_top3"].notna().all()
    assert (metrics["emb_paper_cos"] >= metrics["emb_paper_cos_top3"]).all()
    assert metrics["emb_div"].notna().all()
    assert metrics["emb_div"].nunique() == 1


def test_build_metrics_rejects_unknown_paper() -> None:
    frame = _metric_frame()
    frame.loc[0, "paper_id"] = "missing"
    papers = pd.DataFrame({"paper_id": ["P1"], "whole_paper": ["x"]})
    with pytest.raises(ValueError, match="missing"):
        build_metrics(frame, papers)


@real_data
def test_build_metrics_on_real_reviews_without_embeddings() -> None:
    _, reviews = load_all()
    metrics = build_metrics(reviews, load_papers())
    assert len(metrics) == 960
    assert metrics["structure_score"].between(0.0, 1.0).all()
    assert metrics["length_fit"].between(0.0, 1.0).all()
    assert metrics["degenerate_loop"].dtype == bool


# ---------------------------------------------------------------------------
# Phase 1 analysis
# ---------------------------------------------------------------------------

from experiments.analysis.analyze_phase1 import (  # noqa: E402
    adapter_lift,
    aggregate,
    confound_map,
    guardrail_evidence,
    guardrail_table,
    paired_bootstrap,
    rank_candidates,
    recommendation,
    reference_copy_rate,
    reference_copy_sentence,
)


def _analysis_aggregates() -> pd.DataFrame:
    specs = [
        ("best_config", 1.00, 0.95, 0.95, 1.00, 0.10),
        ("second_config", 0.95, 0.90, 0.90, 0.95, 0.11),
        ("third_config", 0.90, 0.85, 0.85, 0.90, 0.12),
        ("fourth_config", 0.85, 0.80, 0.80, 0.85, 0.13),
        ("fifth_config", 0.80, 0.75, 0.75, 0.80, 0.14),
        ("sixth_config", 0.75, 0.70, 0.70, 0.75, 0.15),
        ("seventh_config", 0.70, 0.65, 0.65, 0.70, 0.16),
        ("a_bit_altered_prompt_60_papers", 0.60, 0.50, 0.50, 0.60, 0.17),
        ("rouge_leader", 0.40, 0.30, 0.30, 0.40, 0.30),
    ]
    rows = []
    for run, length_fit_value, ref, paper, structure, rouge in specs:
        rows.append(
            {
                "run": run,
                "arm": "adapter",
                "n": 10,
                "prompt_family": "synthetic",
                "words_mean": 400.0,
                "length_fit": length_fit_value,
                "emb_ref_cos": ref,
                "emb_paper_cos": paper,
                "structure_score": structure,
                "repeat5_mean": 0.01,
                "degenerate_rate": 0.0,
                "meta_rate": 0.0,
                "cap_rate": 0.0,
                "rouge_l": rouge,
            }
        )
    return pd.DataFrame(rows)


def _analysis_guardrails(runs) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run": list(runs),
            "eligible": [True] * len(list(runs)),
            "guardrail_failures": [""] * len(list(runs)),
        }
    )


def test_rank_candidates_forces_controls_into_shortlist() -> None:
    aggregates = _analysis_aggregates()
    guardrails = _analysis_guardrails(aggregates["run"])
    ranked = rank_candidates(
        aggregates,
        guardrails,
        forced_controls=("a_bit_altered_prompt_60_papers", "rouge_leader"),
        shortlist_size=6,
    )
    assert len(ranked) == 9
    assert ranked["rank"].tolist() == list(range(1, 10))
    assert ranked.iloc[0]["run"] == "best_config"
    shortlisted = ranked[ranked["shortlisted"]]
    assert len(shortlisted) == 6
    forced = set(ranked[ranked["forced_control"]]["run"])
    assert forced == {"a_bit_altered_prompt_60_papers", "rouge_leader"}
    assert forced <= set(shortlisted["run"])
    assert "seventh_config" not in set(shortlisted["run"])
    assert "rouge_leader" not in set(ranked.head(6)["run"])


def test_recommendation_clear_lead_vs_ambiguous() -> None:
    aggregates = _analysis_aggregates()
    guardrails = _analysis_guardrails(aggregates["run"])
    stop_now, reason = recommendation(rank_candidates(aggregates, guardrails))
    assert stop_now is True
    assert "best_config" in reason

    tied = aggregates.copy()
    for component in ("length_fit", "emb_ref_cos", "emb_paper_cos", "structure_score"):
        tied[component] = 0.5
    stop_now, reason = recommendation(
        rank_candidates(tied, _analysis_guardrails(tied["run"]))
    )
    assert stop_now is False
    assert "not decisive" in reason


def test_guardrail_table_flags_degenerate_and_caps() -> None:
    frame = pd.DataFrame(
        {
            "run": ["run_a"] * 4,
            "arm": ["adapter"] * 4,
            "paper_id": ["p1", "p2", "p3", "p4"],
            "degenerate_loop": [True, True, False, False],
            "meta_commentary": [False, False, False, False],
            "hit_cap": [True, False, False, False],
        }
    )
    table = guardrail_table(frame).iloc[0]
    assert table["eligible"] == False  # noqa: E712 - numpy bool
    assert "degenerate_loop on 2 papers" in table["guardrail_failures"]
    assert "cap_rate" in table["guardrail_failures"]


def test_aggregate_matches_hand_computed_means() -> None:
    frame = pd.DataFrame(
        {
            "run": ["r"] * 2,
            "arm": ["adapter"] * 2,
            "paper_id": ["p1", "p2"],
            "words": [100, 300],
            "output_tokens": [150, 350],
            "hit_cap": [False, True],
            "repeat5": [0.0, 0.2],
            "repeat10": [0.0, 0.4],
            "degenerate_loop": [False, True],
            "meta_commentary": [False, False],
            "location_any": [True, False],
            "structure_score": [1.0, 0.0],
            "length_fit": [0.8, 0.6],
            "emb_ref_cos": [0.5, 0.7],
            "emb_paper_cos": [0.6, 0.8],
            "emb_paper_cos_top3": [0.55, 0.75],
            "rouge_l": [0.1, 0.2],
            "emb_div": [0.3, 0.3],
            "seconds": [10.0, 30.0],
        }
    )
    row = aggregate(frame).iloc[0]
    assert row["n"] == 2
    assert row["words_mean"] == pytest.approx(200.0)
    assert row["words_median"] == pytest.approx(200.0)
    assert row["cap_rate"] == pytest.approx(0.5)
    assert row["length_fit"] == pytest.approx(0.7)
    assert row["emb_ref_cos"] == pytest.approx(0.6)


def test_paired_bootstrap_is_deterministic_and_directional() -> None:
    rows = []
    for index in range(5):
        paper = f"p{index}"
        rows.append({"run": "r", "arm": "base", "paper_id": paper, "words": 100.0})
        rows.append({"run": "r", "arm": "adapter", "paper_id": paper, "words": 110.0})
    frame = pd.DataFrame(rows)
    first = paired_bootstrap(frame, "r", "words", n_resamples=2000)
    second = paired_bootstrap(frame, "r", "words", n_resamples=2000)
    assert first == second
    assert first["mean_diff"] == pytest.approx(10.0)
    assert first["ci_low"] == pytest.approx(10.0)
    assert first["ci_high"] == pytest.approx(10.0)
    assert first["win_rate"] == 1.0
    assert paired_bootstrap(frame, "missing", "words") is None


def test_confound_map_labels_single_axis_and_package_effect() -> None:
    runs = pd.DataFrame(
        {
            "run": ["a", "b", "c"],
            "prompt_family": ["x", "x", "x"],
            "n_papers": [10, 10, 10],
            "temperature": [0.7, 0.8, 0.8],
            "top_p": [0.9, 0.9, 0.8],
            "top_k": [50, 50, 50],
            "repetition_penalty": [1.15, 1.15, 1.15],
            "max_paper_tokens": [17500, 17500, 17500],
            "max_new_tokens": [4096, 4096, 4096],
            "arms": ["adapter,base", "adapter,base", "adapter"],
        }
    )
    table = confound_map(runs).set_index(["run_a", "run_b"])
    assert table.loc[("a", "b"), "differing_axes"] == "temperature"
    assert table.loc[("a", "b"), "interpretation"] == "single_axis"
    assert table.loc[("b", "c"), "n_differing_axes"] == 2
    assert table.loc[("b", "c"), "interpretation"] == "package_effect"


def test_rank_candidates_derives_prompt_family_when_absent() -> None:
    aggregates = _analysis_aggregates().drop(columns=["prompt_family"])
    aggregates.loc[aggregates["run"] == "best_config", "run"] = "tweaked_prompt_high_temp"
    ranked = rank_candidates(aggregates, _analysis_guardrails(aggregates["run"]))
    families = dict(zip(ranked["run"], ranked["prompt_family"]))
    assert families["tweaked_prompt_high_temp"] == "tweaked"
    assert families["second_config"] == "unknown"


def test_paired_bootstrap_accepts_boolean_metrics() -> None:
    metrics = pd.DataFrame(
        {
            "run": ["r"] * 4,
            "paper_id": ["p1", "p1", "p2", "p2"],
            "arm": ["adapter", "base", "adapter", "base"],
            "hit_cap": [False, True, False, False],
        }
    )
    result = paired_bootstrap(metrics, "r", "hit_cap", n_resamples=200)
    assert result is not None
    assert result["n_papers"] == 2
    assert result["mean_diff"] == pytest.approx(-0.5)


def test_guardrail_table_applies_rate_rule_on_small_runs() -> None:
    rows = []
    for i in range(10):
        rows.append(
            {
                "run": "ten_paper_run",
                "arm": "adapter",
                "paper_id": f"p{i}",
                "degenerate_loop": False,
                "meta_commentary": i == 0,
                "hit_cap": False,
            }
        )
    table = guardrail_table(pd.DataFrame(rows))
    row = table.iloc[0]
    assert row["meta_papers"] == 1
    assert bool(row["eligible"]) is True
    assert bool(row["eligible_strict"]) is False
    assert "meta_rate" in row["strict_failures"]


def test_reference_copy_rate_detects_verbatim_reuse() -> None:
    words = [f"w{i}" for i in range(20)]
    copied = " ".join(words)
    assert reference_copy_rate(copied, copied) == pytest.approx(1.0)
    assert reference_copy_rate(copied, "completely different text here") == 0.0

    independent = " ".join(f"other{i}" for i in range(20))
    assert reference_copy_rate(independent, copied) == 0.0

    half = " ".join(words[:13] + [f"new{i}" for i in range(20)])
    assert 0.0 < reference_copy_rate(half, copied) < 1.0
    assert reference_copy_rate("", copied) == 0.0


def test_forced_control_inside_shortlist_reason_is_truthful() -> None:
    agg = _analysis_aggregates()
    mask = agg["run"] == "rouge_leader"
    for column in ("length_fit", "emb_ref_cos", "emb_paper_cos", "structure_score"):
        agg.loc[mask, column] = float(agg[column].max()) + 0.05
    rails = _analysis_guardrails(agg["run"])
    ranked = rank_candidates(
        agg, rails, forced_controls=("a_bit_altered_prompt_60_papers", "rouge_leader")
    )
    row = ranked[ranked["run"] == "rouge_leader"].iloc[0]
    assert bool(row["forced_control"]) is True
    assert bool(row["shortlisted"]) is True
    assert row["rank"] <= 6
    assert "outside" not in row["shortlist_reason"]


def test_adapter_lift_labels_helped_and_hurt() -> None:
    rows = []
    base = {
        "arm": "base",
        "n": 10,
        "words_mean": 1500.0,
        "length_fit": 0.20,
        "emb_ref_cos": 0.80,
        "emb_paper_cos": 0.80,
        "structure_score": 0.40,
        "repeat5_mean": 0.02,
        "cap_rate": 0.10,
    }
    better = {
        **base,
        "arm": "adapter",
        "words_mean": 400.0,
        "length_fit": 0.90,
        "emb_ref_cos": 0.85,
        "emb_paper_cos": 0.85,
        "structure_score": 0.70,
        "repeat5_mean": 0.01,
        "cap_rate": 0.0,
    }
    worse = {**better, "run": "worse_run", "length_fit": 0.10, "emb_ref_cos": 0.70, "emb_paper_cos": 0.70, "structure_score": 0.20}
    for run, adapter in (("better_run", better), ("worse_run", worse)):
        rows.append({**base, "run": run})
        rows.append({**adapter, "run": run, "arm": "adapter"})
    lift = adapter_lift(pd.DataFrame(rows))
    assert set(lift["run"]) == {"better_run", "worse_run"}
    by_run = lift.set_index("run")
    assert by_run.at["better_run", "verdict"] == "helped"
    assert by_run.at["better_run", "components_won"] == 4
    assert by_run.at["worse_run", "verdict"] == "hurt"
    assert by_run.at["better_run", "d_words_mean"] < 0


def test_add_all_reference_metrics_uses_every_human_review() -> None:
    class _VectorEmbedder:
        def embed_many(self, texts, progress=None):
            out = {}
            for text in dict.fromkeys(str(text) for text in texts):
                vector = np.array(
                    [1.0, float(len(text) % 7), float(sum(map(ord, text)) % 11)],
                    dtype=np.float32,
                )
                out[text] = vector / np.linalg.norm(vector)
            return out

    metrics = pd.DataFrame(
        {
            "paper_id": ["p1", "p1"],
            "review": ["a review text", "another review text"],
            "reference_review": ["ref one", "ref one"],
        }
    )
    references = {"p1": ["ref one", "ref two", "ref three"]}
    out = add_all_reference_metrics(metrics, references, _VectorEmbedder())
    assert list(out["n_human_reviews"]) == [3, 3]
    assert (out["emb_ref_cos_all_max"] >= out["emb_ref_cos_all_mean"]).all()
    assert out["emb_ref_cos_all_max"].notna().all()


def test_load_all_references_counts_multiple_human_reviews() -> None:
    from experiments.analysis.local_metrics import DEFAULT_EXPERIMENTS_DIR, PAPERS_FILENAME

    path = Path(DEFAULT_EXPERIMENTS_DIR) / PAPERS_FILENAME
    if not path.is_file():
        pytest.skip("external experiment data not present")
    references = load_all_references(path)
    assert len(references) == 172
    assert sum(len(bucket) for bucket in references.values()) == 280
    assert max(len(bucket) for bucket in references.values()) == 4


def test_guardrail_evidence_only_names_rate_rule_runs() -> None:
    metrics = pd.DataFrame(
        {
            "run": ["rate_only", "rate_only", "clean"],
            "arm": ["adapter"] * 3,
            "paper_id": ["p1", "p2", "p1"],
            "meta_commentary": [True, False, False],
            "degenerate_loop": [False, False, False],
            "meta_phrases": ["post-editing notes", "", ""],
            "reference_review": ["summary_of_the_paper: nothing", "", ""],
        }
    )
    guardrails = pd.DataFrame(
        {
            "run": ["rate_only", "clean"],
            "eligible": [True, True],
            "eligible_strict": [False, True],
            "meta_rate": [0.5, 0.0],
            "degenerate_rate": [0.0, 0.0],
        }
    )
    text = guardrail_evidence(metrics, guardrails)
    assert "rate_only" in text
    assert "p1" in text
    assert "single flagged review" in text
    assert "clean" not in text


def test_guardrail_evidence_reports_agreement_and_missing_rule() -> None:
    metrics = pd.DataFrame(columns=["run", "arm", "paper_id"])
    agreeing = pd.DataFrame(
        {"run": ["a"], "eligible": [True], "eligible_strict": [True]}
    )
    assert "agree on every run" in guardrail_evidence(metrics, agreeing)
    missing = pd.DataFrame({"run": ["a"], "eligible": [True]})
    assert "not available" in guardrail_evidence(metrics, missing)


def test_reference_copy_sentence_reports_max_adapter_value() -> None:
    aggregates = pd.DataFrame(
        {
            "run": ["strict_run", "sloppy_run", "base_run"],
            "arm": ["adapter", "adapter", "base"],
            "reference_copy12": [0.001, 0.004, 0.9],
        }
    )
    text = reference_copy_sentence(aggregates)
    assert "sloppy_run" in text
    assert "base" not in text


# ---------------------------------------------------------------------------
# Phase 2 analysis
# ---------------------------------------------------------------------------

from experiments.analysis.analyze_phase2 import (  # noqa: E402
    absolute_table,
    audit_agreement,
    bradley_terry,
    paired_bootstrap as phase2_paired_bootstrap,
    render_decision,
)


def _judge_records(specs):
    """specs: list of (config, arm, overall, flags)."""
    rows = []
    for config, arm, overall, flags in specs:
        for index in range(len(overall)):
            rows.append(
                {
                    "mode": "absolute",
                    "config": config,
                    "arm": arm,
                    "paper_id": f"p{index}",
                    "overall": float(overall[index]),
                    "factual": 5.0,
                    "grounding": 5.0,
                    "coverage": 5.0,
                    "actionability": 5.0,
                    "argument": 5.0,
                    "calibration": 5.0,
                    "length": 5.0,
                    "flags": flags[index] if isinstance(flags, list) else dict(flags),
                    "parse_ok": True,
                }
            )
    return pd.DataFrame(rows)


def test_absolute_table_flags_repeated_critical_flags() -> None:
    clean = {"invented_location": False, "generic_filler": False, "meta_commentary": False, "degenerate_repetition": False}
    dirty = dict(clean, invented_location=True)
    records = _judge_records(
        [
            ("good_run", "adapter", [50, 50, 50], clean),
            ("bad_run", "adapter", [10, 10, 10], [dirty, dirty, clean]),
        ]
    )
    table = absolute_table(records).set_index("config")
    assert table.loc["good_run", "overall"] == 50.0
    assert bool(table.loc["good_run", "eligible"]) is True
    assert bool(table.loc["bad_run", "eligible"]) is False
    assert "invented_location 2/3" in table.loc["bad_run", "critical_failures"][0]


def test_phase2_paired_bootstrap_direction_and_ci() -> None:
    base = {f"p{i}": 60.0 for i in range(5)}
    adapt = {f"p{i}": 30.0 for i in range(5)}
    records = _judge_records([("run_x", "adapter", list(adapt.values()), {}), ("run_x", "base", list(base.values()), {})])
    stats = phase2_paired_bootstrap(records, "run_x", "adapter", "run_x", "base")
    assert stats["n"] == 5
    assert stats["diff"] == pytest.approx(-30.0)
    assert stats["ci_low"] <= stats["diff"] <= stats["ci_high"]
    assert stats["win_rate"] == 0.0


def test_bradley_terry_ranks_transitive_winners() -> None:
    rows = [("A", "B", 1.0)] * 6 + [("A", "C", 1.0)] * 6 + [("B", "C", 1.0)] * 6
    strengths = bradley_terry(rows)
    assert list(strengths.index) == ["A", "B", "C"]
    assert strengths["A"] > strengths["B"] > strengths["C"]
    assert np.isfinite(strengths.to_numpy()).all()


def test_audit_agreement_counts_reversed_order_consistency() -> None:
    records = pd.DataFrame(
        [
            {
                "mode": "pairwise",
                "key": "k1",
                "paper_id": "p1",
                "config": "run_a",
                "arm": "adapter",
                "opponent": "run_b",
                "opponent_arm": "adapter",
                "left": "run_a [adapter]",
                "right": "run_b [adapter]",
                "winner_label": "run_a [adapter]",
                "winner_config": "run_a",
                "winner_arm": "adapter",
            },
            {
                "mode": "pairwise_audit",
                "key": "k2",
                "paired_key": "k1",
                "paper_id": "p1",
                "config": "run_a",
                "arm": "adapter",
                "opponent": "run_b",
                "opponent_arm": "adapter",
                "winner_config": "run_a",
                "winner_arm": "adapter",
            },
            {
                "mode": "pairwise_audit",
                "key": "k3",
                "paired_key": "k1",
                "paper_id": "p1",
                "config": "run_a",
                "arm": "adapter",
                "opponent": "run_b",
                "opponent_arm": "adapter",
                "winner_config": "run_b",
                "winner_arm": "adapter",
            },
        ]
    )
    records = records.set_index("key", drop=False)
    audit = audit_agreement(records)
    assert audit["checked"] == 2
    assert audit["agree"] == 1
    assert audit["rate"] == pytest.approx(0.5)


def test_render_decision_prefers_higher_overall_and_names_least_severe() -> None:
    table = pd.DataFrame(
        [
            {
                "config": "run_base",
                "arm": "base",
                "n": 10,
                "overall": 70.0,
                "eligible": False,
                "critical_failures": ["meta_commentary 2/10"],
                "severity": 1,
                "invented_location_count": 0,
                "meta_commentary_count": 2,
                "degenerate_repetition_count": 0,
                "generic_filler_count": 1,
                "factual": 6.0,
                "grounding": 7.0,
                "coverage": 7.0,
                "actionability": 7.0,
                "argument": 6.0,
                "calibration": 6.0,
                "length": 7.0,
            },
            {
                "config": "run_adapter",
                "arm": "adapter",
                "n": 10,
                "overall": 35.0,
                "eligible": False,
                "critical_failures": ["invented_location 3/10"],
                "severity": 1,
                "invented_location_count": 3,
                "meta_commentary_count": 0,
                "degenerate_repetition_count": 0,
                "generic_filler_count": 5,
                "factual": 4.0,
                "grounding": 3.0,
                "coverage": 3.0,
                "actionability": 3.0,
                "argument": 3.0,
                "calibration": 3.0,
                "length": 4.0,
            },
        ]
    )
    audit = {"checked": 0, "agree": 0, "ties": 0, "rate": float("nan"), "details": []}
    text = render_decision(table, pd.DataFrame(), pd.Series(dtype=float), audit, {}, {}, 2)
    assert "run_base" in text
    assert "70.0" in text
    assert "Fine-tuning did not help" in text
    assert "pre-registered guardrails" in text


# --------------------------------------------------------------------------
# Phase 3A: length curves, settings weights, token efficiency


from experiments.analysis.analyze_phase2 import (  # noqa: E402
    attach_token_efficiency,
    token_costs,
)
from experiments.analysis.length_and_settings import (  # noqa: E402
    axis_split,
    length_bands,
    length_correlations,
    length_verdict,
    matched_pairs,
    ridge_weights,
    settings_frame,
)


def _judged_frame(rows):
    """rows: list of (config, arm, words, overall)."""
    return pd.DataFrame(
        [
            {
                "config": config,
                "arm": arm,
                "paper_id": f"p{index}",
                "words": words,
                "overall": overall,
                "mode": "absolute",
            }
            for index, (config, arm, words, overall) in enumerate(rows)
        ]
    )


def _settings_runs(rows):
    """rows: list of (run, family, temperature, n_papers)."""
    base = {
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.15,
        "max_paper_tokens": 17500,
        "max_new_tokens": 6144,
    }
    return pd.DataFrame(
        [
            {
                "run": run,
                "prompt_family": family,
                "temperature": temperature,
                "n_papers": n_papers,
                **base,
            }
            for run, family, temperature, n_papers in rows
        ]
    )


def _aggregate_frame(rows):
    """rows: list of (run, composite_rank, n_papers)."""
    return pd.DataFrame(
        [
            {
                "run": run,
                "composite_rank": composite_rank,
                "n_papers": n_papers,
                "length_fit": 0.6,
                "emb_ref_cos": 0.85,
                "emb_paper_cos": 0.83,
                "structure_score": 0.55,
                "words_mean": 330.0,
                "repeat5_mean": 0.01,
                "degenerate_rate": 0.0,
                "rouge_l": 0.145,
            }
            for run, composite_rank, n_papers in rows
        ]
    )


def test_length_bands_groups_and_averages_by_arm():
    judged = _judged_frame(
        [
            ("run_a", "adapter", 150, 10.0),
            ("run_a", "adapter", 360, 40.0),
            ("run_a", "adapter", 380, 20.0),
            ("run_b", "base", 1500, 70.0),
        ]
    )
    bands = length_bands(judged)
    adapter = bands[bands["arm"] == "adapter"].set_index("band")
    assert adapter.loc["350-400", "n"] == 2
    assert adapter.loc["350-400", "mean_score"] == pytest.approx(30.0)
    assert adapter.loc["<200", "mean_score"] == pytest.approx(10.0)
    base = bands[bands["arm"] == "base"]
    assert list(base["band"]) == ["1000+"]


def test_length_correlations_reverse_sign_between_arms():
    judged = _judged_frame(
        [
            ("run_a", "adapter", 200, 20.0),
            ("run_a", "adapter", 250, 24.0),
            ("run_a", "adapter", 300, 28.0),
            ("run_a", "adapter", 350, 32.0),
            ("run_a", "adapter", 400, 36.0),
            ("run_b", "base", 1200, 78.0),
            ("run_b", "base", 1400, 70.0),
            ("run_b", "base", 1600, 62.0),
            ("run_b", "base", 1800, 54.0),
            ("run_b", "base", 2000, 46.0),
        ]
    )
    table = length_correlations(judged).set_index("group")
    assert table.loc["arm=adapter", "spearman"] > 0.9
    assert table.loc["arm=base", "spearman"] < -0.9


def test_length_verdict_names_a_band():
    judged = _judged_frame(
        [
            ("run_a", "adapter", 150, 10.0),
            ("run_a", "adapter", 360, 40.0),
            ("run_a", "adapter", 380, 42.0),
        ]
    )
    verdict, best = length_verdict(judged)
    assert "350-400" in verdict
    assert best == pytest.approx(41.0)


def test_matched_pairs_flags_bundled_axes_and_signed_delta():
    runs = _settings_runs(
        [
            ("fam_low", "fam", 0.4, 20),
            ("fam_high", "fam", 0.8, 20),
            ("bundled_low", "bundle", 0.4, 20),
            ("bundled_high", "bundle", 0.8, 40),
            ("single", "solo", 0.7, 20),
        ]
    )
    aggregates = _aggregate_frame(
        [
            ("fam_low", 8.0, 20),
            ("fam_high", 2.0, 20),
            ("bundled_low", 6.0, 20),
            ("bundled_high", 3.0, 40),
            ("single", 5.0, 20),
        ]
    )
    pairs = matched_pairs(aggregates, runs).set_index("family")
    assert pairs.loc["fam", "delta_composite"] == pytest.approx(-6.0)
    assert pairs.loc["fam", "other_axes_differ"] == ""
    assert "n_papers" in pairs.loc["bundle", "other_axes_differ"]
    assert "solo" not in pairs.index


def test_axis_split_reports_one_row_per_axis():
    runs = _settings_runs(
        [
            ("r1", "fam", 0.4, 10),
            ("r2", "fam", 0.5, 20),
            ("r3", "fam", 0.6, 40),
            ("r4", "fam", 0.8, 60),
            ("r5", "fam", 0.9, 100),
        ]
    )
    aggregates = _aggregate_frame([(f"r{index}", float(index), 10) for index in range(1, 6)])
    split = axis_split(aggregates, runs)
    assert "axis" in split.columns
    assert "temperature" in set(split["axis"])


def test_ridge_weights_reports_leave_one_out_score():
    runs = _settings_runs(
        [
            (f"r{index}", "fam", 0.4 + index * 0.1, 10 * (index + 1))
            for index in range(6)
        ]
    )
    aggregates = _aggregate_frame([(f"r{index}", float(index + 1), 10 * (index + 1)) for index in range(6)])
    weights = ridge_weights(aggregates, runs)
    assert set(["feature", "coefficient", "std_coefficient"]).issubset(weights.columns)
    assert weights.attrs["n_runs"] == 6
    assert isinstance(weights.attrs["loo_r2"], float)


def test_settings_frame_keeps_only_setting_columns():
    runs = _settings_runs([("r1", "fam", 0.8, 10)])
    frame = settings_frame(runs)
    assert "run" in frame.columns
    assert "temperature" in frame.columns
    assert "adapter_dir" not in frame.columns


def test_token_costs_uses_only_judged_papers():
    records = pd.DataFrame(
        [
            {"config": "run_a", "arm": "adapter", "paper_id": "p1", "mode": "absolute"},
            {"config": "run_a", "arm": "adapter", "paper_id": "p1", "mode": "pairwise"},
        ]
    )
    costs = pd.DataFrame(
        [
            {"config": "run_a", "arm": "adapter", "paper_id": "p1", "prompt_tokens": 10000, "output_tokens": 400, "words": 300},
            {"config": "run_a", "arm": "adapter", "paper_id": "p9", "prompt_tokens": 99000, "output_tokens": 9900, "words": 3000},
        ]
    )
    result = token_costs(records, costs=costs).set_index("config")
    assert result.loc["run_a", "output_tokens"] == pytest.approx(400.0)
    assert result.loc["run_a", "prompt_tokens"] == pytest.approx(10000.0)


def test_attach_token_efficiency_computes_both_ratios():
    table = pd.DataFrame([{"config": "run_a", "arm": "adapter", "overall": 40.0}])
    costs = pd.DataFrame(
        [{"config": "run_a", "arm": "adapter", "output_tokens": 400.0, "prompt_tokens": 12000.0, "words": 300.0}]
    )
    result = attach_token_efficiency(table, costs)
    assert result.loc[0, "points_per_1k_output"] == pytest.approx(100.0)
    assert result.loc[0, "points_per_1k_total"] == pytest.approx(40.0 / 12400.0 * 1000.0)


def test_attach_token_efficiency_degrades_without_costs():
    table = pd.DataFrame([{"config": "run_a", "arm": "adapter", "overall": 40.0}])
    result = attach_token_efficiency(table, pd.DataFrame())
    assert np.isnan(result.loc[0, "points_per_1k_output"])


# --- Phase 3B: final configuration pick ----------------------------------

from experiments.analysis.final_pick import (
    candidate_table,
    gate_table,
    pick,
    render_report,
)


def _final_local_rows():
    return pd.DataFrame(
        [
            {
                "run": "clean_high",
                "arm": "adapter",
                "n_papers": 20,
                "length_fit": 0.66,
                "emb_ref_cos": 0.85,
                "emb_paper_cos": 0.84,
                "structure_score": 0.61,
                "words_mean": 360.0,
                "repeat5_mean": 0.005,
                "degenerate_rate": 0.0,
                "meta_rate": 0.0,
                "cap_rate": 0.0,
                "composite_rank": 2.0,
            },
            {
                "run": "noisy_high",
                "arm": "adapter",
                "n_papers": 20,
                "length_fit": 0.64,
                "emb_ref_cos": 0.85,
                "emb_paper_cos": 0.83,
                "structure_score": 0.60,
                "words_mean": 380.0,
                "repeat5_mean": 0.08,
                "degenerate_rate": 0.25,
                "meta_rate": 0.0,
                "cap_rate": 0.0,
                "composite_rank": 3.0,
            },
            {
                "run": "tie_high",
                "arm": "adapter",
                "n_papers": 20,
                "length_fit": 0.62,
                "emb_ref_cos": 0.84,
                "emb_paper_cos": 0.82,
                "structure_score": 0.58,
                "words_mean": 420.0,
                "repeat5_mean": 0.006,
                "degenerate_rate": 0.0,
                "meta_rate": 0.0,
                "cap_rate": 0.0,
                "composite_rank": 4.0,
            },
        ]
    )


def _final_scores_rows():
    return pd.DataFrame(
        [
            {"config": "clean_high", "arm": "adapter", "n": 20, "overall": 40.0},
            {"config": "noisy_high", "arm": "adapter", "n": 20, "overall": 55.0},
            {"config": "tie_high", "arm": "adapter", "n": 20, "overall": 38.0},
        ]
    )


def test_gate_table_rejects_degenerate_run():
    gated = gate_table(_final_local_rows())
    by_run = gated.set_index("run")
    assert not bool(by_run.loc["noisy_high", "passes_gates"])
    assert "degenerate_rate" in by_run.loc["noisy_high", "gate_failures"]
    assert bool(by_run.loc["clean_high", "passes_gates"])


def test_pick_ignores_highest_score_that_fails_a_gate():
    table = candidate_table(_final_scores_rows(), _final_local_rows())
    table["points_per_1k_output"] = [88.0, 95.0, 70.0]
    winner, reason = pick(table, noise_band=1.0)
    assert winner == "clean_high"
    assert "passes every gate" in reason


def test_pick_breaks_noise_band_tie_on_token_efficiency():
    table = candidate_table(_final_scores_rows(), _final_local_rows())
    table.loc[table["config"] == "noisy_high", "overall"] = 39.5
    table["points_per_1k_output"] = [80.0, 60.0, 95.0]
    winner, reason = pick(table, noise_band=5.0)
    assert winner == "tie_high"
    assert "noise band" in reason


def test_pick_returns_none_when_every_config_fails():
    local = _final_local_rows().copy()
    local["meta_rate"] = 0.5
    table = candidate_table(_final_scores_rows(), local)
    winner, reason = pick(table)
    assert winner is None
    assert "hard gates" in reason


def test_candidate_table_keeps_only_adapter_rows():
    scores = pd.concat(
        [
            _final_scores_rows(),
            pd.DataFrame([{"config": "clean_high", "arm": "base", "n": 20, "overall": 72.0}]),
        ],
        ignore_index=True,
    )
    table = candidate_table(scores, _final_local_rows())
    assert set(table["arm"]) == {"adapter"}
    assert len(table) == 3


def test_render_report_without_winner_is_explicit():
    report = render_report(pd.DataFrame(), None, "nothing passed", {}, "")
    assert "No winner" in report


# --- Phase 3C: artifact verification -------------------------------------

from experiments.analysis import verify_results


def test_verify_flags_duplicate_judge_keys(tmp_path, monkeypatch):
    import json as _json

    log = tmp_path / "judge.jsonl"
    rows = [
        {"key": "a", "mode": "absolute", "overall": 50.0, "parse_ok": True},
        {"key": "a", "mode": "absolute", "overall": 55.0, "parse_ok": True},
        {"key": "b", "mode": "absolute", "overall": None, "parse_ok": False},
    ]
    log.write_text("\n".join(_json.dumps(row) for row in rows), encoding="utf-8")
    monkeypatch.setattr(verify_results, "JUDGE_LOG", log)
    joined = " ".join(verify_results._check_judge_log())
    assert "duplicate resume keys" in joined
    assert "failed to parse" in joined


def test_verify_flags_duplicate_score_rows(tmp_path, monkeypatch):
    scores = pd.DataFrame(
        [
            {"config": "a", "arm": "adapter", "overall": 30.0, "points_per_1k_output": 70.0},
            {"config": "a", "arm": "adapter", "overall": 31.0, "points_per_1k_output": 71.0},
            {"config": "a", "arm": "base", "overall": 60.0, "points_per_1k_output": 20.0},
        ]
    )
    path = tmp_path / "scores.csv"
    scores.to_csv(path, index=False)
    monkeypatch.setattr(verify_results, "SCORES_CSV", path)
    joined = " ".join(verify_results._check_scores())
    assert "duplicate rows" in joined


def test_verify_flags_self_comparison_in_pairwise(tmp_path, monkeypatch):
    pairwise = pd.DataFrame(
        [
            {"a_config": "a", "b_config": "a", "winner_config": "a"},
            {"a_config": "a", "b_config": "b", "winner_config": "b"},
        ]
    )
    path = tmp_path / "pairwise.csv"
    pairwise.to_csv(path, index=False)
    monkeypatch.setattr(verify_results, "PAIRWISE_CSV", path)
    joined = " ".join(verify_results._check_pairwise())
    assert "comparing a config with itself" in joined

def test_common_paper_scores_restricts_to_shared_papers():
    from experiments.analysis import analyze_phase2

    records = pd.DataFrame(
        [
            {"mode": "absolute", "config": "wide", "arm": "adapter", "paper_id": COMMON_10[0], "overall": 10.0},
            {"mode": "absolute", "config": "wide", "arm": "adapter", "paper_id": COMMON_10[1], "overall": 20.0},
            {"mode": "absolute", "config": "wide", "arm": "adapter", "paper_id": "extra-paper", "overall": 90.0},
            {"mode": "pairwise", "config": "wide", "arm": "adapter", "paper_id": COMMON_10[0], "overall": 1.0},
        ]
    )
    scores = analyze_phase2.common_paper_scores(records)
    row = scores[(scores["config"] == "wide")].iloc[0]
    assert row["n_common"] == 2
    assert row["overall_common"] == pytest.approx(15.0)


def test_attach_common_paper_scores_fills_nan_when_absent():
    from experiments.analysis import analyze_phase2

    table = pd.DataFrame(
        [{"config": "wide", "arm": "adapter", "overall": 40.0}]
    )
    attached = analyze_phase2.attach_common_paper_scores(table, pd.DataFrame())
    assert attached["overall_common"].isna().all()
    assert (attached["n_common"] == 0).all()


def test_render_report_mentions_quality_first_alternative():
    from experiments.analysis import final_pick

    table = _candidate_frame()
    text = final_pick.render_report(
        table,
        winner="cheap_runner",
        reason="tie broken on efficiency",
        settings={"prompt_chars": 8673},
        prompt_excerpt="",
        alternative={
            "config": "quality_leader",
            "score": 41.1,
            "words": 389,
            "prompt_chars": 1415,
        },
    )
    assert "Quality-first alternative" in text
    assert "quality_leader" in text
    assert "1415" in text
    assert "8673" in text
def test_render_report_mentions_quality_first_alternative():
    from experiments.analysis import final_pick

    table = final_pick.candidate_table(
        pd.DataFrame(_final_scores_rows()), pd.DataFrame(_final_local_rows())
    )
    text = final_pick.render_report(
        table,
        winner="cheap_runner",
        reason="tie broken on efficiency",
        settings={"prompt_chars": 8673},
        prompt_excerpt="",
        alternative={
            "config": "quality_leader",
            "score": 41.1,
            "words": 389,
            "prompt_chars": 1415,
        },
    )
    assert "Quality-first alternative" in text
    assert "quality_leader" in text
    assert "1415" in text
    assert "8673" in text


def test_quality_first_alternative():
    from experiments.analysis import final_pick

    table = final_pick.candidate_table(
        pd.DataFrame(_final_scores_rows()), pd.DataFrame(_final_local_rows())
    )
    assert not table.empty


def test_head_to_head_reports_both_sides(tmp_path):
    from experiments.analysis import final_pick

    path = tmp_path / "phase2_pairwise.csv"
    pd.DataFrame(
        [
            {"config_a": "pick [adapter]", "config_b": "rival [adapter]",
             "a_wins": 12, "b_wins": 8, "ties": 0, "n": 20},
            {"config_a": "other [adapter]", "config_b": "pick [adapter]",
             "a_wins": 3, "b_wins": 7, "ties": 0, "n": 10},
        ]
    ).to_csv(path, index=False)

    lines = final_pick.head_to_head("pick", "adapter", path=path)
    assert lines == [
        "- against rival (adapter): won 12, lost 8 of 20",
        "- against other (adapter): won 7, lost 3 of 10",
    ]


def test_head_to_head_is_empty_without_pairwise_file(tmp_path):
    from experiments.analysis import final_pick

    assert final_pick.head_to_head("pick", path=tmp_path / "absent.csv") == []


def test_higher_scoring_elsewhere_lists_only_strictly_better_runs():
    from experiments.analysis import final_pick

    table = pd.DataFrame(
        [
            {"config": "pick", "overall": 35.8, "n": 20},
            {"config": "better", "overall": 42.1, "n": 20},
            {"config": "worse", "overall": 29.0, "n": 10},
        ]
    )
    lines = final_pick.higher_scoring_elsewhere(table, "pick")
    assert len(lines) == 1
    assert "better" in lines[0] and "42.1" in lines[0]
    assert all("worse" not in line for line in lines)
    assert final_pick.higher_scoring_elsewhere(table, "absent") == []


def test_select_configs_judges_both_arms_of_an_explicit_list():
    from experiments.analysis import judge

    adapters, bases = judge.select_configs(
        "a_bit_altered_prompt_60_papers, very_big_prompt_high_temp"
    )
    assert adapters == ["a_bit_altered_prompt_60_papers", "very_big_prompt_high_temp"]
    assert bases == adapters


def test_select_configs_falls_back_to_the_screening_set(tmp_path):
    from experiments.analysis import judge

    ranking = tmp_path / "ranking.csv"
    ranking.write_text(
        "rank,run,arm\n1,b_run,adapter\n2,a_run,adapter\n3,c_run,adapter\n",
        encoding="utf-8",
    )
    adapters, bases = judge.select_configs("", False, ranking_path=ranking)
    assert adapters == ["b_run", "a_run", "c_run"]
    assert bases == ["b_run", "a_run"]

    every_adapter, _ = judge.select_configs("", True, ranking_path=ranking)
    assert every_adapter == ["b_run", "a_run", "c_run"]

def test_shared_paper_means_uses_only_common_papers():
    from experiments.analysis.final_pick import shared_paper_means

    records = pd.DataFrame(
        [
            {"mode": "absolute", "config": "wide", "arm": "adapter", "paper_id": "p1", "overall": 40.0},
            {"mode": "absolute", "config": "wide", "arm": "adapter", "paper_id": "p2", "overall": 30.0},
            {"mode": "absolute", "config": "wide", "arm": "adapter", "paper_id": "p3", "overall": 90.0},
            {"mode": "absolute", "config": "narrow", "arm": "adapter", "paper_id": "p1", "overall": 20.0},
            {"mode": "absolute", "config": "narrow", "arm": "adapter", "paper_id": "p2", "overall": 10.0},
            {"mode": "pairwise", "config": "wide", "arm": "adapter", "paper_id": "p1", "overall": None},
        ]
    )
    comparison = shared_paper_means(records, "wide", "narrow")
    assert comparison["n"] == 2
    assert comparison["mean_a"] == 35.0
    assert comparison["mean_b"] == 15.0
    assert comparison["delta"] == 20.0


def test_pick_prefers_paired_comparison_over_token_efficiency():
    from experiments.analysis import final_pick

    table = pd.DataFrame(
        [
            {
                "config": "cheap",
                "score": 40.0,
                "overall": 40.0,
                "n": 10,
                "points_per_1k_output": 95.0,
                "emb_paper_cos": 0.83,
                "length_fit": 0.60,
                "passes_gates": True,
                "gate_failures": "",
            },
            {
                "config": "better",
                "score": 37.0,
                "overall": 37.0,
                "n": 10,
                "points_per_1k_output": 50.0,
                "emb_paper_cos": 0.83,
                "length_fit": 0.60,
                "passes_gates": True,
                "gate_failures": "",
            },
        ]
    )
    records = []
    for index in range(30):
        records.append(
            {"mode": "absolute", "config": "cheap", "arm": "adapter", "paper_id": f"p{index}", "overall": 20.0}
        )
        records.append(
            {"mode": "absolute", "config": "better", "arm": "adapter", "paper_id": f"p{index}", "overall": 50.0}
        )
    winner, reason = final_pick.pick(table, records=pd.DataFrame(records))
    assert winner == "better"
    assert "30 papers both were judged on" in reason


def test_pick_falls_back_to_efficiency_when_paired_difference_is_noise():
    from experiments.analysis import final_pick

    table = pd.DataFrame(
        [
            {
                "config": "cheap",
                "score": 40.0,
                "overall": 40.0,
                "n": 10,
                "points_per_1k_output": 95.0,
                "emb_paper_cos": 0.83,
                "length_fit": 0.60,
                "passes_gates": True,
                "gate_failures": "",
            },
            {
                "config": "pricey",
                "score": 38.0,
                "overall": 38.0,
                "n": 10,
                "points_per_1k_output": 20.0,
                "emb_paper_cos": 0.83,
                "length_fit": 0.60,
                "passes_gates": True,
                "gate_failures": "",
            },
        ]
    )
    records = []
    for index in range(20):
        records.append(
            {"mode": "absolute", "config": "cheap", "arm": "adapter", "paper_id": f"p{index}", "overall": 40.0}
        )
        records.append(
            {"mode": "absolute", "config": "pricey", "arm": "adapter", "paper_id": f"p{index}", "overall": 42.0}
        )
    winner, reason = final_pick.pick(table, records=pd.DataFrame(records))
    assert winner == "cheap"
    assert "token efficiency" in reason


def test_gate_table_applies_judged_flag_rates():
    """A review that invents locations or role-plays a reviewer is unusable even if it scores."""
    from experiments.analysis import final_pick

    frame = pd.DataFrame(
        [
            {
                "config": "role_player",
                "arm": "adapter",
                "n": 20,
                "degenerate_rate": 0.0,
                "meta_rate": 0.0,
                "cap_rate": 0.0,
                "invented_location_count": 1,
                "meta_commentary_count": 9,
                "degenerate_repetition_count": 0,
            },
            {
                "config": "clean",
                "arm": "adapter",
                "n": 20,
                "degenerate_rate": 0.0,
                "meta_rate": 0.0,
                "cap_rate": 0.0,
                "invented_location_count": 2,
                "meta_commentary_count": 2,
                "degenerate_repetition_count": 1,
            },
        ]
    )

    gated = final_pick.gate_table(frame).set_index("config")

    assert not bool(gated.loc["role_player", "passes_gates"])
    assert "judge_meta=45%>25%" in gated.loc["role_player", "gate_failures"]
    assert bool(gated.loc["clean", "passes_gates"])
    assert gated.loc["clean", "gate_failures"] == ""
