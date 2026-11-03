"""Consistency checks over the generated Phase 1/2/3 artifacts.

This is the last gate before reporting. It does not recompute anything; it checks
that the files on disk agree with each other and with the archived runs.

Run with ``python -m experiments.analysis.verify_results``. Exit code 0 means
every check passed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import pandas as pd

from .load_runs import load_all, validate_all

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "experiments" / "results"

METRICS_CSV = RESULTS_DIR / "per_review_metrics.csv"
RANKING_CSV = RESULTS_DIR / "phase1_ranking.csv"
RANKING_BASE_CSV = RESULTS_DIR / "phase1_ranking_base.csv"
LIFT_CSV = RESULTS_DIR / "adapter_lift.csv"
CONFOUND_CSV = RESULTS_DIR / "confound_map.csv"
JUDGE_LOG = RESULTS_DIR / "judge_scores.jsonl"
SCORES_CSV = RESULTS_DIR / "phase2_scores.csv"
PAIRWISE_CSV = RESULTS_DIR / "phase2_pairwise.csv"
LENGTH_BANDS_CSV = RESULTS_DIR / "length_bands.csv"
SETTINGS_PAIRS_CSV = RESULTS_DIR / "settings_pairs.csv"
SETTINGS_WEIGHTS_CSV = RESULTS_DIR / "settings_weights.csv"
FINAL_MD = RESULTS_DIR / "FINAL_CONFIG.md"

REQUIRED_METRIC_COLUMNS = (
    "run",
    "arm",
    "paper_id",
    "review",
    "words",
    "output_tokens",
    "length_fit",
    "structure_score",
    "emb_ref_cos",
    "emb_paper_cos",
    "degenerate_loop",
    "meta_commentary",
    "hit_cap",
)
REQUIRED_EMBEDDING_COLUMNS = ("emb_ref_cos", "emb_paper_cos", "emb_paper_cos_top3")


def _check_files(paths: Sequence[Path]) -> List[str]:
    return [f"missing artifact: {path.name}" for path in paths if not path.exists()]


def _check_metrics(reviews_rows: int) -> List[str]:
    if not METRICS_CSV.exists():
        return []
    metrics = pd.read_csv(METRICS_CSV)
    errors: List[str] = []
    missing = [column for column in REQUIRED_METRIC_COLUMNS if column not in metrics.columns]
    if missing:
        errors.append(f"per_review_metrics.csv missing columns: {missing}")
        return errors
    if len(metrics) != reviews_rows:
        errors.append(
            f"per_review_metrics.csv has {len(metrics)} rows, archived runs have {reviews_rows}"
        )
    duplicates = metrics.duplicated(subset=["run", "arm", "paper_id"]).sum()
    if duplicates:
        errors.append(f"per_review_metrics.csv has {duplicates} duplicate (run, arm, paper_id) rows")
    for column in REQUIRED_EMBEDDING_COLUMNS:
        if column in metrics.columns:
            gaps = metrics[column].isna().sum()
            if gaps:
                errors.append(f"{column} has {gaps} missing values")
    empty_reviews = metrics["review"].isna().sum() + (metrics["review"].astype(str).str.strip() == "").sum()
    if empty_reviews:
        errors.append(f"{empty_reviews} empty review texts in per_review_metrics.csv")
    return errors


def _check_ranking() -> List[str]:
    if not RANKING_CSV.exists():
        return []
    ranking = pd.read_csv(RANKING_CSV)
    errors: List[str] = []
    if "arm" in ranking.columns and set(ranking["arm"].unique()) != {"adapter"}:
        errors.append(f"phase1_ranking.csv should be adapter-only, found {set(ranking['arm'])}")
    if ranking["rank"].duplicated().any():
        errors.append("phase1_ranking.csv has duplicate ranks")
    if ranking["composite_rank"].isna().any():
        errors.append("phase1_ranking.csv has missing composite ranks")
    if RANKING_BASE_CSV.exists():
        base = pd.read_csv(RANKING_BASE_CSV)
        if "arm" in base.columns and set(base["arm"].unique()) != {"base"}:
            errors.append(f"phase1_ranking_base.csv should be base-only, found {set(base['arm'])}")
        if len(base) >= len(ranking):
            errors.append("base leaderboard should have fewer runs than the adapter one")
    return errors


def _check_lift() -> List[str]:
    if not LIFT_CSV.exists():
        return []
    lift = pd.read_csv(LIFT_CSV)
    errors: List[str] = []
    if "verdict" in lift.columns:
        unexpected = set(lift["verdict"]) - {"helped", "hurt", "wash"}
        if unexpected:
            errors.append(f"adapter_lift.csv has unexpected verdicts: {unexpected}")
    if "components_won" in lift.columns and not lift["components_won"].between(0, 4).all():
        errors.append("adapter_lift.csv components_won outside 0-4")
    return errors


def _check_confounds() -> List[str]:
    if not CONFOUND_CSV.exists():
        return []
    confounds = pd.read_csv(CONFOUND_CSV)
    errors: List[str] = []
    if "interpretation" in confounds.columns:
        unexpected = set(confounds["interpretation"]) - {"identical", "single_axis", "package_effect"}
        if unexpected:
            errors.append(f"confound_map.csv has unexpected interpretations: {unexpected}")
    return errors


def _check_judge_log() -> List[str]:
    if not JUDGE_LOG.exists():
        return []
    records = []
    with JUDGE_LOG.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    errors: List[str] = []
    failed = [record for record in records if not record.get("parse_ok")]
    if failed:
        errors.append(f"{len(failed)} judge records failed to parse")
    keys = [record["key"] for record in records]
    if len(keys) != len(set(keys)):
        errors.append("judge_scores.jsonl has duplicate resume keys")
    absolute = [record for record in records if record.get("mode") == "absolute"]
    missing_score = [record for record in absolute if record.get("overall") is None]
    if missing_score:
        errors.append(f"{len(missing_score)} absolute judge records have no overall score")
    out_of_range = [
        record
        for record in absolute
        if record.get("overall") is not None and not 0 <= float(record["overall"]) <= 100
    ]
    if out_of_range:
        errors.append(f"{len(out_of_range)} judge records have overall outside 0-100")
    return errors


def _check_scores() -> List[str]:
    if not SCORES_CSV.exists():
        return []
    scores = pd.read_csv(SCORES_CSV)
    errors: List[str] = []
    if scores.empty:
        errors.append("phase2_scores.csv is empty")
        return errors
    key_columns = ["config", "arm"] if "arm" in scores.columns else ["config"]
    if scores.duplicated(subset=key_columns).any():
        duplicates = scores[scores.duplicated(subset=key_columns)][key_columns].to_dict("records")
        errors.append(f"phase2_scores.csv has duplicate rows: {duplicates}")
    if "points_per_1k_output" not in scores.columns:
        errors.append("phase2_scores.csv missing points_per_1k_output")
    elif scores["points_per_1k_output"].isna().all():
        errors.append("phase2_scores.csv points_per_1k_output is entirely missing")
    for column in ("overall_common", "n_common"):
        if column not in scores.columns:
            errors.append(f"phase2_scores.csv missing {column}")
    if "n_common" in scores.columns:
        counts = set(int(value) for value in scores["n_common"].dropna())
        if counts and counts != {10}:
            errors.append(
                f"phase2_scores.csv n_common should be 10 for every compared configuration, saw {sorted(counts)}"
            )
    if {"overall", "overall_common"} <= set(scores.columns):
        drift = (scores["overall_common"] - scores["overall"]).abs()
        if scores["overall"].notna().any() and drift.max(skipna=True) > 60:
            errors.append("phase2_scores.csv overall_common and overall disagree by more than 60 points")
    return errors


def _check_pairwise() -> List[str]:
    if not PAIRWISE_CSV.exists():
        return []
    pairwise = pd.read_csv(PAIRWISE_CSV)
    errors: List[str] = []
    if "a_config" in pairwise.columns and "b_config" in pairwise.columns:
        same = (pairwise["a_config"] == pairwise["b_config"]).sum()
        if same:
            errors.append(f"phase2_pairwise.csv has {same} rows comparing a config with itself")
    return errors


def _check_phase3() -> List[str]:
    errors: List[str] = []
    if LENGTH_BANDS_CSV.exists():
        bands = pd.read_csv(LENGTH_BANDS_CSV)
        if {"arm", "band"} - set(bands.columns):
            errors.append("length_bands.csv missing arm or band column")
    if SETTINGS_PAIRS_CSV.exists():
        pairs = pd.read_csv(SETTINGS_PAIRS_CSV)
        if "other_axes_differ" not in pairs.columns:
            errors.append("settings_pairs.csv missing other_axes_differ")
    if SETTINGS_WEIGHTS_CSV.exists():
        weights = pd.read_csv(SETTINGS_WEIGHTS_CSV)
        if "std_coefficient" not in weights.columns:
            errors.append("settings_weights.csv missing std_coefficient")
    if FINAL_MD.exists():
        text = FINAL_MD.read_text(encoding="utf-8")
        if "No winner" not in text and "**Run `" not in text:
            errors.append("FINAL_CONFIG.md does not name a run")
    return errors


def verify() -> List[str]:
    errors: List[str] = []
    errors.extend(
        _check_files(
            [
                METRICS_CSV,
                RANKING_CSV,
                RANKING_BASE_CSV,
                LIFT_CSV,
                CONFOUND_CSV,
                JUDGE_LOG,
                SCORES_CSV,
                PAIRWISE_CSV,
                LENGTH_BANDS_CSV,
                SETTINGS_PAIRS_CSV,
                SETTINGS_WEIGHTS_CSV,
                FINAL_MD,
            ]
        )
    )

    validation = validate_all()
    errors.extend(f"archive validation: {message}" for message in validation)

    try:
        _, reviews = load_all()
        reviews_rows = len(reviews)
    except Exception as error:  # pragma: no cover - only on corrupt data
        errors.append(f"could not load archived runs: {type(error).__name__}: {error}")
        reviews_rows = -1

    if reviews_rows > 0:
        errors.extend(_check_metrics(reviews_rows))
    errors.extend(_check_ranking())
    errors.extend(_check_lift())
    errors.extend(_check_confounds())
    errors.extend(_check_judge_log())
    errors.extend(_check_scores())
    errors.extend(_check_pairwise())
    errors.extend(_check_phase3())
    return errors


def main() -> int:
    errors = verify()
    if not errors:
        print("all artifact checks passed")
        return 0
    print(f"{len(errors)} problem(s) found:")
    for message in errors:
        print(f" - {message}")
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
