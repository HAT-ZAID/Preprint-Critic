"""Check the numbers quoted in README.md and HOW_THE_EXPERIMENT_RAN.md.

This is a documentation audit, not a test of the analysis. It reads the artifacts produced by the
Phase 1-3 pipeline and asserts the specific figures that the written reports claim, so a stale
number cannot survive a rerun. Run with ``python -m experiments.analysis.audit_docs``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
MAIN_REPO = Path(r"D:\Preprint Critic")


def _close(actual: float, expected: float, tolerance: float = 0.06) -> bool:
    return abs(float(actual) - float(expected)) <= tolerance


def check_scores(problems: List[str]) -> None:
    path = RESULTS_DIR / "phase2_scores.csv"
    if not path.exists():
        problems.append("missing phase2_scores.csv")
        return
    table = pd.read_csv(path)

    def row(config: str, arm: str) -> pd.Series:
        match = table[(table["config"] == config) & (table["arm"] == arm)]
        if match.empty:
            raise KeyError(f"{config}/{arm}")
        return match.iloc[0]

    claims: List[Tuple[str, str, str, float]] = [
        ("a_bit_altered_prompt_60_papers", "adapter", "overall", 38.63),
        ("very_big_prompt_high_temp", "adapter", "overall", 40.18),
        ("original_prompt_high_temp", "adapter", "overall", 42.15),
        ("new_prompt_with_constrains_60_papers", "adapter", "overall", 38.5),
        ("a_bit_altered_prompt_60_papers", "adapter", "overall_common", 41.1),
        ("new_prompt_with_constrains_60_papers", "adapter", "overall_common", 40.0),
        ("tweaked_prompt_mid_temp", "adapter", "overall_common", 37.3),
        ("very_big_prompt_high_temp", "adapter", "overall_common", 37.0),
        ("original_prompt_original_settings", "adapter", "overall_common", 36.2),
        ("original_prompt_high_temp", "adapter", "overall_common", 36.0),
        ("gpt_prompt_high_temp", "adapter", "overall_common", 35.5),
        ("relaxed_references_prompt_high_temp", "adapter", "overall_common", 33.9),
        ("tweaked_prompt_high_temp", "adapter", "overall_common", 33.2),
        ("original_prompt_low_temp", "adapter", "overall_common", 32.1),
        ("very_big_prompt_very_high_temp", "adapter", "overall_common", 29.0),
        ("original_prompt_high_temp", "base", "overall_common", 72.7),
        ("gpt_prompt_high_temp", "base", "overall_common", 55.1),
        ("a_bit_altered_prompt_60_papers", "base", "overall", 73.6),
        ("very_big_prompt_high_temp", "base", "overall", 71.35),
        ("very_big_prompt_high_temp", "adapter", "points_per_1k_output", 81.2),
        ("original_prompt_low_temp", "adapter", "points_per_1k_output", 94.3),
        ("original_prompt_high_temp", "adapter", "points_per_1k_output", 92.4),
        ("a_bit_altered_prompt_60_papers", "adapter", "points_per_1k_output", 76.2),
        ("original_prompt_high_temp", "base", "points_per_1k_output", 35.0),
        ("gpt_prompt_high_temp", "base", "points_per_1k_output", 17.7),
        ("original_prompt_high_temp", "base", "points_per_1k_total", 5.52),
        ("a_bit_altered_prompt_60_papers", "adapter", "points_per_1k_total", 3.32),
        ("very_big_prompt_high_temp", "adapter", "points_per_1k_total", 3.03),
    ]
    for config, arm, column, expected in claims:
        try:
            actual = row(config, arm)[column]
        except KeyError:
            problems.append(f"missing row {config}/{arm}")
            continue
        if column.endswith("points_per_1k_total"):
            ok = _close(actual, expected, 0.02)
        elif column.startswith("points_per_1k"):
            ok = _close(actual, expected, 0.2)
        else:
            ok = _close(actual, expected, 0.06)
        if not ok:
            problems.append(f"{config}/{arm} {column}: claimed {expected}, artifact {actual}")

    # The word counts quoted in the markdown tables.
    for config, expected_words in (
        ("a_bit_altered_prompt_60_papers", 389),
        ("very_big_prompt_high_temp", 374),
        ("tweaked_prompt_mid_temp", 634),
    ):
        actual = row(config, "adapter")["words"]
        if abs(float(actual) - expected_words) > 1.0:
            problems.append(f"{config} words: claimed {expected_words}, artifact {actual}")


def check_ranking_gates(problems: List[str]) -> None:
    path = RESULTS_DIR / "phase1_ranking.csv"
    if not path.exists():
        problems.append("missing phase1_ranking.csv")
        return
    ranking = pd.read_csv(path)
    expectations = {
        "tweaked_prompt_mid_temp": ("degenerate_rate", 0.20),
        "original_prompt_original_settings": ("degenerate_rate", 0.25),
        "gpt_prompt_high_temp": ("meta_rate", 0.10),
        "original_prompt_low_temp": ("degenerate_rate", 0.10),
    }
    for run, (column, expected) in expectations.items():
        match = ranking[ranking["run"] == run]
        if match.empty:
            problems.append(f"missing ranking row {run}")
            continue
        actual = match.iloc[0][column]
        if not _close(actual, expected, 0.005):
            problems.append(f"{run} {column}: claimed {expected}, artifact {actual}")

    # Coverage after the full-coverage expansion of the two finalists.
    scores = pd.read_csv(RESULTS_DIR / "phase2_scores.csv")
    adapters = scores[scores["arm"] == "adapter"]
    coverage = dict(zip(adapters["config"], adapters["n"].astype(int)))
    expected_coverage = {
        "a_bit_altered_prompt_60_papers": 60,
        "very_big_prompt_high_temp": 40,
    }
    for config, expected in expected_coverage.items():
        if coverage.get(config) != expected:
            problems.append(
                f"{config} judged papers: claimed {expected}, artifact {coverage.get(config)}"
            )
    twenty = adapters[adapters["n"] == 20]
    if len(twenty) != 6:
        problems.append(f"expected 6 adapters with 20 judged papers, found {len(twenty)}")
    if (scores["n_common"] != 10).any():
        problems.append("not every configuration has 10 shared judged papers")


def check_final_config(problems: List[str]) -> None:
    path = RESULTS_DIR / "FINAL_CONFIG.md"
    if not path.exists():
        problems.append("missing FINAL_CONFIG.md")
        return
    text = path.read_text(encoding="utf-8")
    for token in (
        "very_big_prompt_high_temp",
        "**0.8**",
        "**0.9**",
        "**40**",
        "**1.15**",
        "**16050**",
        "**4096**",
        "Quality-first alternative",
        "a_bit_altered_prompt_60_papers",
    ):
        if token not in text:
            problems.append(f"FINAL_CONFIG.md is missing {token!r}")


def check_settings_pairs(problems: List[str]) -> None:
    path = RESULTS_DIR / "settings_pairs.csv"
    if not path.exists():
        problems.append("missing settings_pairs.csv")
        return
    pairs = pd.read_csv(path)
    claimed = {"gpt": -7.50, "original": -4.75, "tweaked": -3.75, "relaxed": -2.00}
    flags = pairs["other_axes_differ"]
    clean = pairs[flags.isna() | (flags.astype(str).str.strip() == "")]
    for family, expected in claimed.items():
        match = clean[clean["family"] == family]
        if match.empty:
            problems.append(f"no clean temperature pair for {family}")
            continue
        actual = float(match.iloc[0]["delta_composite"])
        if not _close(actual, expected, 0.01):
            problems.append(f"{family} temperature delta: claimed {expected}, artifact {actual}")
    if len(clean) != 4:
        problems.append(f"expected 4 clean single-axis pairs, found {len(clean)}")


def check_docs(problems: List[str]) -> None:
    readme = MAIN_REPO / "README.md"
    explainer = MAIN_REPO / "HOW_THE_EXPERIMENT_RAN.md"
    for path in (readme, explainer):
        if not path.exists():
            problems.append(f"missing {path.name}")
            continue
        text = path.read_text(encoding="utf-8")
        for token in ("38.6", "40.2", "73.7"):
            if token not in text:
                problems.append(f"{path.name} does not mention {token}")
    if readme.exists() and "## Evaluation across 15 archived runs" not in readme.read_text(encoding="utf-8"):
        problems.append("README.md is missing the evaluation section")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    problems: List[str] = []
    check_scores(problems)
    check_ranking_gates(problems)
    check_final_config(problems)
    check_settings_pairs(problems)
    check_docs(problems)

    if problems:
        print(f"{len(problems)} documentation claim(s) do not match the artifacts:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    if not args.quiet:
        print("every documented number matches the artifacts")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
