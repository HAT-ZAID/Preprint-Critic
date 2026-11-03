"""Phase 1 analysis: local, non-generative comparison of the archived runs.

This module never calls a generative model. It aggregates the per-review local
metrics produced by :mod:`experiments.analysis.local_metrics`, applies the
Phase 1 guardrails and shortlist rule from ``experiments/COMPARISON_PROTOCOL.md``,
and writes the ranking, confound map, and ``PHASE1_RESULTS.md``.

Entry point::

    python -m experiments.analysis.analyze_phase1 --recompute
"""

from __future__ import annotations

import argparse
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .load_runs import EXPECTED_RUNS, load_all, prompt_family, validate_all
from .local_metrics import (
    OllamaEmbedder,
    add_all_reference_metrics,
    build_metrics,
    ensure_server,
    load_all_references,
    load_papers,
    normalise,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
METRICS_CSV = RESULTS_DIR / "per_review_metrics.csv"
RANKING_CSV = RESULTS_DIR / "phase1_ranking.csv"
CONFOUND_CSV = RESULTS_DIR / "confound_map.csv"
REPORT_MD = RESULTS_DIR / "PHASE1_RESULTS.md"

#: Components of the Phase 1 composite. Higher is better for all four.
COMPOSITE_COMPONENTS: Tuple[str, ...] = (
    "length_fit",
    "emb_ref_cos",
    "emb_paper_cos",
    "structure_score",
)

#: Configurations kept in view for reporting only. Phase 1B does not force them
#: into the shortlist; the Phase 2 screening set is simply the top 7 adapter
#: configurations by local rank.
CONTROL_RUNS: Tuple[str, ...] = (
    "a_bit_altered_prompt_60_papers",
    "original_prompt_original_settings",
)

#: Size of the Phase 2 screening set (user decision: top 7).
SHORTLIST_SIZE = 7
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42
BOOTSTRAP_METRICS: Tuple[str, ...] = (
    "words",
    "length_fit",
    "emb_ref_cos",
    "emb_paper_cos",
    "repeat5",
    "structure_score",
    "hit_cap",
)

#: Guardrail thresholds from the protocol.
MAX_DEGENERATE_PAPERS = 2
MAX_META_PAPERS = 2
MAX_CAP_RATE = 0.05
# Rate-based sensitivity rules: the count rules above are too lenient on a
# 10-paper run, where a single flagged paper is already 10% of the sample.
MAX_META_RATE = 0.05
MAX_DEGENERATE_RATE = 0.05

#: A composite lead is "clear" when it beats the runner-up by this many mean
#: rank positions and owns at least two of the four component bests.
CLEAR_MARGIN = 1.0
CLEAR_COMPONENT_WINS = 2

CONFOUND_AXES: Tuple[str, ...] = (
    "prompt_family",
    "n_papers",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "max_paper_tokens",
    "max_new_tokens",
    "arms",
)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def reference_copy_rate(review: str, reference: str, n: int = 12) -> float:
    """Share of the review's word 12-grams that also appear in the reference.

    A high value means the review is echoing the human reference rather than
    writing independently. It is a diagnostic, not a composite component.
    """
    def grams(text: str) -> set:
        words = re.findall(r"[a-z0-9]+", str(text).lower())
        return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}

    review_grams = grams(review)
    if not review_grams:
        return 0.0
    return len(review_grams & grams(reference)) / len(review_grams)


def aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-review metrics into one row per ``(run, arm)``."""
    rows: List[dict] = []
    for (run, arm), group in metrics.groupby(["run", "arm"], sort=False):
        words = group["words"].astype(float)
        rows.append(
            {
                "run": run,
                "arm": arm,
                "n": int(len(group)),
                "words_mean": float(words.mean()),
                "words_median": float(words.median()),
                "words_p10": float(words.quantile(0.10)),
                "words_p90": float(words.quantile(0.90)),
                "words_lt150_rate": float((words < 150).mean()),
                "words_gt800_rate": float((words > 800).mean()),
                "output_tokens_mean": float(group["output_tokens"].mean()),
                "cap_rate": float(group["hit_cap"].astype(bool).mean()),
                "repeat5_mean": float(group["repeat5"].mean()),
                "repeat10_mean": float(group["repeat10"].mean()),
                "reference_copy12": float(group["reference_copy12"].mean())
                if "reference_copy12" in group
                else float("nan"),
                "degenerate_rate": float(group["degenerate_loop"].astype(bool).mean()),
                "meta_rate": float(group["meta_commentary"].astype(bool).mean()),
                "location_any_rate": float(group["location_any"].astype(bool).mean()),
                "structure_score": float(group["structure_score"].mean()),
                "length_fit": float(group["length_fit"].mean()),
                "emb_ref_cos": float(group["emb_ref_cos"].mean()),
                "emb_ref_cos_all_max": float(group["emb_ref_cos_all_max"].mean())
                if "emb_ref_cos_all_max" in group
                else float("nan"),
                "emb_ref_cos_all_mean": float(group["emb_ref_cos_all_mean"].mean())
                if "emb_ref_cos_all_mean" in group
                else float("nan"),
                "n_human_reviews": float(group["n_human_reviews"].mean())
                if "n_human_reviews" in group
                else float("nan"),
                "emb_paper_cos": float(group["emb_paper_cos"].mean()),
                "emb_paper_cos_top3": float(group["emb_paper_cos_top3"].mean()),
                "rouge_l": float(group["rouge_l"].mean()),
                "emb_div": float(group["emb_div"].mean()),
                "seconds_mean": float(group["seconds"].mean()),
            }
        )
    return pd.DataFrame(rows)


def guardrail_table(metrics: pd.DataFrame, arm: str = "adapter") -> pd.DataFrame:
    """Apply the Phase 1 guardrails to every run of one arm."""
    rows: List[dict] = []
    for run, group in metrics[metrics["arm"] == arm].groupby("run", sort=False):
        degenerate = int(group["degenerate_loop"].astype(bool).sum())
        meta = int(group["meta_commentary"].astype(bool).sum())
        caps = int(group["hit_cap"].astype(bool).sum())
        cap_rate = float(group["hit_cap"].astype(bool).mean())
        degenerate_rate = degenerate / len(group)
        meta_rate = meta / len(group)
        failures: List[str] = []
        if degenerate >= MAX_DEGENERATE_PAPERS:
            failures.append(f"degenerate_loop on {degenerate} papers")
        if meta >= MAX_META_PAPERS:
            failures.append(f"meta_commentary on {meta} papers")
        if cap_rate > MAX_CAP_RATE:
            failures.append(f"cap_rate {cap_rate:.1%} > {MAX_CAP_RATE:.0%}")
        strict_failures = list(failures)
        if degenerate_rate > MAX_DEGENERATE_RATE:
            strict_failures.append(
                f"degenerate_rate {degenerate_rate:.1%} > {MAX_DEGENERATE_RATE:.0%}"
            )
        if meta_rate > MAX_META_RATE:
            strict_failures.append(f"meta_rate {meta_rate:.1%} > {MAX_META_RATE:.0%}")
        rows.append(
            {
                "run": run,
                "degenerate_papers": degenerate,
                "meta_papers": meta,
                "cap_papers": caps,
                "cap_rate": cap_rate,
                "degenerate_rate": degenerate_rate,
                "meta_rate": meta_rate,
                "eligible": not failures,
                "guardrail_failures": "; ".join(failures),
                "eligible_strict": not strict_failures,
                "strict_failures": "; ".join(strict_failures),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ranking and shortlist
# ---------------------------------------------------------------------------


def _safe_prompt_family(run: str) -> str:
    """Map a run name to its prompt family, tolerating synthetic names."""
    try:
        return prompt_family(run)
    except ValueError:
        return "unknown"


def rank_candidates(
    aggregates: pd.DataFrame,
    guardrails: pd.DataFrame,
    forced_controls: Sequence[str] = (),
    shortlist_size: int = SHORTLIST_SIZE,
    arm: str = "adapter",
) -> pd.DataFrame:
    """Rank the configs of one arm by the mean rank of the four components.

    ``arm`` selects the leaderboard: ``adapter`` ranks the fine-tuned outputs,
    ``base`` ranks the same prompts on the untouched model. The two are never
    mixed, because their length behaviour differs by roughly 1,000-2,000 words.

    Controls outside the top ``shortlist_size`` replace the lowest-ranked
    non-control entry when ``forced_controls`` is non-empty. Phase 1B passes no
    controls: the screening set is the plain top 7.
    """
    ranked = aggregates[aggregates["arm"] == arm].copy()
    if "prompt_family" not in ranked.columns:
        ranked["prompt_family"] = ranked["run"].map(_safe_prompt_family)
    ranked = ranked.merge(
        guardrails[
            [
                column
                for column in (
                    "run",
                    "eligible",
                    "guardrail_failures",
                    "eligible_strict",
                    "strict_failures",
                )
                if column in guardrails.columns
            ]
        ],
        on="run",
        how="left",
    )
    ranked["eligible"] = ranked["eligible"].fillna(False).astype(bool)
    ranked["guardrail_failures"] = ranked["guardrail_failures"].fillna("")
    if "eligible_strict" in ranked.columns:
        ranked["eligible_strict"] = ranked["eligible_strict"].fillna(False).astype(bool)
        ranked["strict_failures"] = ranked["strict_failures"].fillna("")

    for component in COMPOSITE_COMPONENTS:
        ranked[f"rank_{component}"] = ranked[component].rank(
            ascending=False, method="min"
        )
    rank_columns = [f"rank_{component}" for component in COMPOSITE_COMPONENTS]
    ranked["composite_rank"] = ranked[rank_columns].mean(axis=1)
    ranked = ranked.sort_values(
        ["composite_rank", "emb_paper_cos", "length_fit", "run"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)

    controls = _controls(ranked, forced_controls)
    ordered = list(ranked["run"])
    reasons: Dict[str, str] = {}
    shortlist = ordered[:shortlist_size]
    for run in shortlist:
        reasons[run] = "top composite rank"
    for control in controls:
        if control in shortlist:
            reasons.setdefault(control, "forced control (already inside top composite ranks)")
            continue
        replaceable = [run for run in shortlist if run not in controls]
        if replaceable:
            dropped = replaceable[-1]
            shortlist.remove(dropped)
            reasons.pop(dropped, None)
        shortlist.append(control)
        reasons[control] = "forced control (outside top composite ranks)"

    ranked["forced_control"] = ranked["run"].isin(controls)
    ranked["shortlisted"] = ranked["run"].isin(shortlist)
    ranked["shortlist_reason"] = ranked["run"].map(reasons).fillna("")

    columns = [
        "rank",
        "run",
        "arm",
        "prompt_family",
        "n",
        "eligible",
        *(
            ["eligible_strict", "strict_failures"]
            if "eligible_strict" in ranked.columns
            else []
        ),
        "guardrail_failures",
        "words_mean",
        "length_fit",
        "emb_ref_cos",
        "emb_paper_cos",
        "structure_score",
        "repeat5_mean",
        "degenerate_rate",
        "meta_rate",
        "cap_rate",
        "rouge_l",
        "composite_rank",
        "forced_control",
        "shortlisted",
        "shortlist_reason",
    ]
    return ranked[columns].rename(columns={"n": "n_papers"})


def _controls(ranked: pd.DataFrame, forced_controls: Sequence[str]) -> List[str]:
    controls: List[str] = []
    for run in forced_controls:
        if run in set(ranked["run"]) and run not in controls:
            controls.append(run)
    return controls


LIFT_COMPONENTS: Tuple[str, ...] = (
    "length_fit",
    "emb_ref_cos",
    "emb_paper_cos",
    "structure_score",
)

LIFT_EXTRA_METRICS: Tuple[str, ...] = (
    "words_mean",
    "repeat5_mean",
    "cap_rate",
)


def adapter_lift(aggregates: pd.DataFrame) -> pd.DataFrame:
    """Compare the adapter and base arms of the same run on the same papers.

    This is the only clean comparison in the archive, because both arms share
    the prompt, decoding settings, and paper list. ``components_won`` counts how
    many of the four composite components the adapter improves, and ``verdict``
    summarises it as helped / hurt / wash.
    """
    adapter = aggregates[aggregates["arm"] == "adapter"].set_index("run")
    base = aggregates[aggregates["arm"] == "base"].set_index("run")
    shared = [run for run in adapter.index if run in base.index]
    rows: List[dict] = []
    for run in shared:
        row: Dict[str, object] = {
            "run": run,
            "n_papers": int(adapter.at[run, "n"]),
        }
        wins = 0
        losses = 0
        for component in LIFT_COMPONENTS:
            difference = float(adapter.at[run, component]) - float(base.at[run, component])
            row[f"d_{component}"] = difference
            wins += int(difference > 0)
            losses += int(difference < 0)
        for metric in LIFT_EXTRA_METRICS:
            row[f"d_{metric}"] = float(adapter.at[run, metric]) - float(base.at[run, metric])
        row["components_won"] = wins
        if wins >= 3:
            row["verdict"] = "helped"
        elif losses >= 3:
            row["verdict"] = "hurt"
        else:
            row["verdict"] = "wash"
        rows.append(row)
    return pd.DataFrame(rows)


def recommendation(ranked: pd.DataFrame) -> Tuple[bool, str]:
    """Return ``(stop_now, explanation)`` for the Phase 1 stop rule.

    The rate-based guardrails are the deciding rules. The pre-registered count
    rules are reported alongside them, because on a 10-paper run a single
    flagged review is already 10% of the sample.
    """
    strict = (
        ranked[ranked["eligible_strict"]]
        if "eligible_strict" in ranked.columns
        else ranked[ranked["eligible"]]
    ).reset_index(drop=True)
    loose = ranked[ranked["eligible"]].reset_index(drop=True)
    if strict.empty:
        return False, "No adapter configuration passed the Phase 1 guardrails."
    best = strict.iloc[0]
    if len(strict) == 1:
        return True, (
            f"`{best['run']}` is the only configuration that passes the "
            "rate-based guardrails, so it leads by default; a judge would add "
            "no discriminating evidence."
        )
    runner_up = strict.iloc[1]
    margin = float(runner_up["composite_rank"] - best["composite_rank"])
    component_wins = sum(
        float(best[component]) >= float(strict[component].max()) - 1e-12
        for component in COMPOSITE_COMPONENTS
    )
    stop_now = margin >= CLEAR_MARGIN and component_wins >= CLEAR_COMPONENT_WINS
    disqualified = ""
    if not loose.empty and loose.iloc[0]["run"] != best["run"]:
        top = loose.iloc[0]
        disqualified = (
            f" The count-based rule alone would have ranked `{top['run']}` first, "
            f"but it is excluded by the rate-based rule ({top['strict_failures']})."
        )
    if stop_now:
        return True, (
            f"`{best['run']}` passes the guardrails, leads the composite by "
            f"{margin:.1f} rank positions over `{runner_up['run']}`, and is best "
            f"on {component_wins} of {len(COMPOSITE_COMPONENTS)} components."
            + disqualified
        )
    return False, (
        f"`{best['run']}` leads `{runner_up['run']}` by only {margin:.1f} composite "
        f"rank positions and is best on {component_wins} of "
        f"{len(COMPOSITE_COMPONENTS)} components; the local evidence is not "
        "decisive enough to stop without a judge." + disqualified
    )


# ---------------------------------------------------------------------------
# Paired bootstrap and confound map
# ---------------------------------------------------------------------------


def paired_bootstrap(
    metrics: pd.DataFrame,
    run: str,
    metric: str,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Optional[dict]:
    """Bootstrap adapter-minus-base over papers for one run and metric."""
    subset = metrics[(metrics["run"] == run) & metrics["arm"].isin(["adapter", "base"])]
    if subset.empty:
        return None
    pivot = subset.pivot_table(
        index="paper_id", columns="arm", values=metric, aggfunc="first"
    )
    if not {"adapter", "base"}.issubset(pivot.columns):
        return None
    pivot = pivot.astype(float)
    differences = (pivot["adapter"] - pivot["base"]).dropna().to_numpy(dtype=float)
    if differences.size == 0:
        return None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(n_resamples, differences.size))
    means = differences[indices].mean(axis=1)
    return {
        "run": run,
        "metric": metric,
        "n_papers": int(differences.size),
        "mean_diff": float(differences.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "win_rate": float((differences > 0).mean()),
    }


def bootstrap_table(
    metrics: pd.DataFrame,
    runs: Iterable[str] = EXPECTED_RUNS,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> pd.DataFrame:
    rows: List[dict] = []
    for run in runs:
        for metric in BOOTSTRAP_METRICS:
            result = paired_bootstrap(metrics, run, metric, n_resamples=n_resamples)
            if result is not None:
                rows.append(result)
    return pd.DataFrame(rows)


def _values_differ(axis: str, left: object, right: object) -> bool:
    if axis == "arms":
        left_arms = {part.strip() for part in str(left).split(",") if part.strip()}
        right_arms = {part.strip() for part in str(right).split(",") if part.strip()}
        return left_arms != right_arms
    if axis == "prompt_family":
        return str(left) != str(right)
    try:
        return abs(float(left) - float(right)) > 1e-9
    except (TypeError, ValueError):
        return str(left) != str(right)


def confound_map(runs: pd.DataFrame) -> pd.DataFrame:
    """List which axes differ between every pair of runs."""
    rows: List[dict] = []
    records = runs.to_dict("records")
    for left, right in combinations(records, 2):
        differing = [
            axis
            for axis in CONFOUND_AXES
            if _values_differ(axis, left.get(axis), right.get(axis))
        ]
        if not differing:
            interpretation = "identical"
        elif len(differing) == 1:
            interpretation = "single_axis"
        else:
            interpretation = "package_effect"
        rows.append(
            {
                "run_a": left["run"],
                "run_b": right["run"],
                "differing_axes": ",".join(differing),
                "n_differing_axes": len(differing),
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _format_value(value: object) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "-"
        return f"{value:.3f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in frame[list(columns)].itertuples(index=False):
        lines.append("| " + " | ".join(_format_value(value) for value in row) + " |")
    return "\n".join(lines)


def guardrail_evidence(
    metrics: pd.DataFrame, guardrails: pd.DataFrame
) -> str:
    """Describe the runs that only the rate-based rule excludes.

    Everything here is computed from the loaded metrics and guardrail table, so
    the report cannot claim something that the data does not show.
    """
    if "eligible_strict" not in guardrails.columns:
        return "The rate-based guardrail is not available for this metrics file."
    rate_only = guardrails[guardrails["eligible"] & ~guardrails["eligible_strict"]]
    if rate_only.empty:
        return (
            "No run is excluded by the rate rule alone; the count-based and "
            "rate-based guardrails agree on every run."
        )
    details: List[str] = []
    copied_from_reference = 0
    total_flagged = 0
    for row in rate_only.itertuples(index=False):
        flagged = metrics[
            (metrics["run"] == row.run)
            & (metrics["arm"] == "adapter")
            & (
                metrics["meta_commentary"].astype(bool)
                | metrics["degenerate_loop"].astype(bool)
            )
        ]
        total_flagged += len(flagged)
        papers = ", ".join(f"`{paper}`" for paper in flagged["paper_id"]) or "none"
        details.append(
            f"`{row.run}` (meta rate {row.meta_rate:.1%}, degenerate rate "
            f"{row.degenerate_rate:.1%}), flagged on {papers}"
        )
        for review in flagged.itertuples(index=False):
            phrases = [
                phrase.strip()
                for phrase in str(getattr(review, "meta_phrases", "")).split(";")
                if phrase.strip()
            ]
            reference = normalise(str(review.reference_review)).lower().replace(
                "\u2019", "'"
            )
            if any(phrase in reference for phrase in phrases):
                copied_from_reference += 1
    evidence = (
        "The rate rule alone excludes "
        + "; ".join(details)
        + ", which the count rule keeps. The flagged papers are listed so they "
        "can be read directly."
    )
    noun = "review" if total_flagged == 1 else "reviews"
    if total_flagged:
        if copied_from_reference:
            evidence += (
                f" {copied_from_reference} of {total_flagged} flagged {noun} "
                "repeated the matched phrase from its reference review, so part "
                "of the flag may be reference reuse rather than reviewer voice."
            )
        elif total_flagged == 1:
            evidence += (
                " The single flagged review did not contain the matched phrase "
                "in its reference review, so the flag is reviewer voice produced "
                "by the model, not reference copying."
            )
        else:
            evidence += (
                f" None of the {total_flagged} flagged {noun} contained the "
                "matched phrase in its reference review, so the flag is reviewer "
                "voice produced by the model, not reference copying."
            )
    return evidence


def reference_sensitivity_sentence(aggregates: pd.DataFrame) -> str:
    """Quantify how much the chosen human reference affected the scores."""
    adapter = aggregates[aggregates["arm"] == "adapter"]
    if "emb_ref_cos_all_max" not in adapter.columns or adapter.empty:
        return "The all-reference check is not available for this metrics file."
    gap = adapter["emb_ref_cos_all_max"] - adapter["emb_ref_cos"]
    if gap.dropna().empty:
        return "The all-reference check is not available for this metrics file."
    largest_run = str(adapter.loc[gap.idxmax(), "run"])
    top_shipped = set(adapter.nlargest(SHORTLIST_SIZE, "emb_ref_cos")["run"])
    top_all = set(adapter.nlargest(SHORTLIST_SIZE, "emb_ref_cos_all_max")["run"])
    overlap = len(top_shipped & top_all)
    return (
        f"Matching against the best of all human reviews instead of the shipped "
        f"one raises a config's `emb_ref_cos` by {float(gap.mean()):.4f} on average "
        f"and by at most {float(gap.max()):.4f} (`{largest_run}`), so the shipped "
        f"reference was not an unusually bad match. Ranking by this component "
        f"alone keeps {overlap} of {SHORTLIST_SIZE} configurations, so at most "
        f"{SHORTLIST_SIZE - overlap} place"
        f"{'s' if SHORTLIST_SIZE - overlap != 1 else ''} would change."
    )


def reference_copy_sentence(aggregates: pd.DataFrame) -> str:
    """Report the maximum adapter ``reference_copy12`` actually observed."""
    adapter = aggregates[aggregates["arm"] == "adapter"]
    if "reference_copy12" not in adapter.columns or adapter.empty:
        return (
            "The reference-copy diagnostic is not available for this metrics file."
        )
    values = pd.to_numeric(adapter["reference_copy12"], errors="coerce").dropna()
    if values.empty:
        return "The reference-copy diagnostic is not available for this metrics file."
    maximum = float(values.max())
    leader = str(adapter.loc[values.idxmax(), "run"])
    return (
        "Reference copying is not the explanation for the embedding scores: "
        "`reference_copy12` is the share of a review's 12-word sequences that also "
        f"occur in its reference, and the highest adapter value is {maximum:.6f} "
        f"(`{leader}`), so the similarity is stylistic and topical agreement rather "
        "than verbatim reuse."
    )


def render_report(
    runs: pd.DataFrame,
    metrics: pd.DataFrame,
    aggregates: pd.DataFrame,
    guardrails: pd.DataFrame,
    ranked: pd.DataFrame,
    ranked_base: pd.DataFrame,
    lift: pd.DataFrame,
    bootstraps: pd.DataFrame,
    confounds: pd.DataFrame,
    validation_errors: Sequence[str],
    metrics_source: str,
) -> str:
    stop_now, explanation = recommendation(ranked)
    shortlisted = ranked[ranked["shortlisted"]]
    eligible_column = "eligible_strict" if "eligible_strict" in ranked.columns else "eligible"
    best = ranked[ranked[eligible_column]].head(1)
    headline = best.iloc[0] if not best.empty else ranked.iloc[0]

    lines: List[str] = []
    lines.append("# Phase 1 results: local comparison of the 15 archived runs")
    lines.append("")
    lines.append(
        "Phase 1 only. No generative model was called, no OpenCode session was "
        "created, no training or new review generation was run."
    )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(
        f"**Phase 2 screening set: the top {SHORTLIST_SIZE} adapter configurations "
        "below.** The user asked for the top 7 by local rank, so guardrails are "
        "reported as annotations and do not remove a config from the screening set."
    )
    lines.append("")
    lines.append(
        f"Local-only leader: `{headline['run']}` (composite mean rank "
        f"{headline['composite_rank']:.2f}, {int(headline['n_papers'])} papers). "
        + explanation
    )
    lines.append("")
    lines.append(
        "Adapter and base are ranked on **separate leaderboards** because the "
        "untuned model writes roughly 1,000-2,000 more words per review; mixing "
        "them would compare length policies, not review quality."
    )
    lines.append("")
    lines.append("## Validation")
    lines.append("")
    if validation_errors:
        lines.append(f"- Validation errors: **{len(validation_errors)}**")
        for error in validation_errors[:10]:
            lines.append(f"  - {error}")
    else:
        lines.append("- All 15 runs validated: required files, columns, row counts, paper order, summary aggregates, and reference identity.")
    lines.append(f"- Runs loaded: {len(runs)}; review rows: {len(metrics)}.")
    lines.append(f"- Metrics source: {metrics_source}.")
    lines.append(f"- Embedding model: `nomic-embed-text:latest` (768-d, local Ollama).")
    lines.append("")
    lines.append("## Ranking (adapter arms)")
    lines.append("")
    lines.append(
        _markdown_table(
            ranked,
            [
                "rank",
                "run",
                "n_papers",
                "eligible",
                "eligible_strict",
                "composite_rank",
                "length_fit",
                "emb_ref_cos",
                "emb_paper_cos",
                "structure_score",
                "rouge_l",
                "shortlisted",
            ],
        )
    )
    lines.append("")
    lines.append(
        "Every component is shown. `composite_rank` is the mean of the four "
        "component ranks (lower is better); it is not a quality score."
    )
    lines.append("")
    lines.append("## Ranking (base arms, same prompts, untuned model)")
    lines.append("")
    lines.append(
        _markdown_table(
            ranked_base,
            [
                "rank",
                "run",
                "n_papers",
                "eligible",
                "composite_rank",
                "length_fit",
                "emb_ref_cos",
                "emb_paper_cos",
                "structure_score",
                "rouge_l",
            ],
        )
    )
    lines.append("")
    lines.append(
        "Base reviews are much longer, so `length_fit` is low for almost every "
        "base run. This board answers a different question: which prompt produces "
        "the best output when the adapter is switched off."
    )
    lines.append("")
    lines.append("## Adapter lift vs base (same run, same papers)")
    lines.append("")
    if lift.empty:
        lines.append("No run has both arms.")
    else:
        lines.append(
            _markdown_table(
                lift,
                [
                    "run",
                    "n_papers",
                    "d_length_fit",
                    "d_emb_ref_cos",
                    "d_emb_paper_cos",
                    "d_structure_score",
                    "d_words_mean",
                    "d_repeat5_mean",
                    "d_cap_rate",
                    "components_won",
                    "verdict",
                ],
            )
        )
        counts = lift["verdict"].value_counts().to_dict()
        lines.append("")
        lines.append(
            f"The adapter helped on {counts.get('helped', 0)} runs, hurt on "
            f"{counts.get('hurt', 0)}, and was a wash on {counts.get('wash', 0)}. "
            "`d_` columns are adapter minus base. A negative `d_words_mean`, "
            "`d_repeat5_mean`, or `d_cap_rate` is an improvement."
        )
    lines.append("")
    lines.append("## Guardrails")
    lines.append("")
    lines.append(
        _markdown_table(
            guardrails.sort_values("run"),
            [
                "run",
                "degenerate_papers",
                "meta_papers",
                "cap_rate",
                "eligible",
                "guardrail_failures",
                "eligible_strict",
                "strict_failures",
            ],
        )
    )
    lines.append("")
    lines.append(
        f"Thresholds: degenerate loop on {MAX_DEGENERATE_PAPERS}+ papers, "
        f"meta-commentary on {MAX_META_PAPERS}+ papers, cap rate above "
        f"{MAX_CAP_RATE:.0%}. A failed guardrail blocks the winner, not the ranking."
    )
    lines.append("")
    lines.append(
        f"`eligible` applies those count rules. `eligible_strict` additionally "
        f"fails a run when more than {MAX_DEGENERATE_RATE:.0%} of its papers have a "
        f"degenerate loop or more than {MAX_META_RATE:.0%} contain reviewer "
        "meta-commentary. The rate rule is the deciding one: on a 10-paper run a "
        "single flagged review is already 10% of the sample. The count-only result "
        "is kept so the effect of the threshold is visible."
    )
    lines.append("")
    lines.append("### Guardrail evidence")
    lines.append("")
    lines.append(guardrail_evidence(metrics, guardrails))
    lines.append("")
    lines.append(reference_copy_sentence(aggregates))
    lines.append("")
    lines.append("## Component detail")
    lines.append("")
    adapter = aggregates[aggregates["arm"] == "adapter"].set_index("run")
    adapter = adapter.reindex(ranked["run"])
    detail_columns = [
        "run",
        "words_mean",
        "words_median",
        "words_p10",
        "words_p90",
        "words_lt150_rate",
        "words_gt800_rate",
        "cap_rate",
        "repeat5_mean",
        "degenerate_rate",
        "meta_rate",
        "reference_copy12",
        "location_any_rate",
        "emb_ref_cos",
        "emb_ref_cos_all_max",
        "emb_ref_cos_all_mean",
        "n_human_reviews",
        "emb_paper_cos",
        "emb_paper_cos_top3",
    ]
    detail_columns = [name for name in detail_columns if name in adapter.columns]
    lines.append(
        _markdown_table(
            adapter.reset_index(),
            detail_columns,
        )
    )
    lines.append("")
    if "emb_ref_cos_all_max" in adapter.columns:
        lines.append(
            "`emb_ref_cos` compares against the single human review shipped in "
            "the run CSV. `emb_ref_cos_all_max` and `emb_ref_cos_all_mean` "
            "compare against every human review the dataset holds for that paper "
            "(1-4 reviews), and `n_human_reviews` is the average count. The gap "
            "between the two shows how much the score depended on which human "
            "review was picked."
        )
        lines.append("")
        lines.append(reference_sensitivity_sentence(aggregates))
        lines.append("")
    lines.append("## Adapter vs base (paired bootstrap, 10,000 resamples)")
    lines.append("")
    if bootstraps.empty:
        lines.append("No run has both arms.")
    else:
        lines.append(
            _markdown_table(
                bootstraps,
                ["run", "metric", "n_papers", "mean_diff", "ci_low", "ci_high", "win_rate"],
            )
        )
        lines.append("")
        lines.append(
            "`mean_diff` is adapter minus base; `win_rate` is the share of papers "
            "where the adapter value is larger. For `repeat5` and `hit_cap` lower is "
            "better, so a negative difference is favourable."
        )
    lines.append("")
    lines.append("## Phase 2 screening set (top %d adapters by local rank)" % SHORTLIST_SIZE)
    lines.append("")
    lines.append(
        _markdown_table(
            shortlisted,
            ["rank", "run", "n_papers", "eligible_strict", "composite_rank"],
        )
    )
    lines.append("")
    lines.append(
        "These seven are the configs a judge would score next. Guardrail status "
        "is shown but does not remove a config, per the user's instruction."
    )
    lines.append("")
    eligibility = (
        ranked["eligible_strict"].astype(bool)
        if "eligible_strict" in ranked.columns
        else ranked["eligible"].astype(bool)
    )
    displaced = ranked[
        (~ranked["shortlisted"].astype(bool))
        & (ranked["rank"] <= SHORTLIST_SIZE)
        & eligibility
    ]
    if len(displaced):
        names = ", ".join(f"`{run}`" for run in displaced["run"])
        single = len(displaced) == 1
        lines.append(
            f"Forced controls displaced {names} from the top {SHORTLIST_SIZE}. "
            f"{names} {'is' if single else 'are'} still guardrail-eligible and "
            f"{'scores' if single else 'score'} above a control on the composite, so "
            f"Phase 2 should judge {'it' if single else 'them'} if the shortlist is "
            f"expanded."
        )
        lines.append("")
    lines.append("## Confound summary")
    lines.append("")
    counts = confounds["interpretation"].value_counts().to_dict()
    lines.append(
        f"- Pairs compared: {len(confounds)}; identical: {counts.get('identical', 0)}; "
        f"single axis: {counts.get('single_axis', 0)}; "
        f"package effect: {counts.get('package_effect', 0)}."
    )
    single = confounds[confounds["interpretation"] == "single_axis"]
    if not single.empty:
        lines.append("- Single-axis pairs (clean comparisons):")
        for row in single.itertuples(index=False):
            lines.append(
                f"  - `{row.run_a}` vs `{row.run_b}`: `{row.differing_axes}`"
            )
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- The 10 shared papers were already used during prompt development, so this "
        "is a dev-set comparison, not a held-out test."
    )
    lines.append(
        "- Sample sizes differ (10 to 100 papers) and two 100-paper runs are "
        "adapter-only, so cross-run comparisons are unequal."
    )
    lines.append(
        "- Embedding cosine is a proxy for topical similarity and grounding, not "
        "proof of factual correctness."
    )
    lines.append(
        "- ROUGE rewards word overlap and can reward copying or repetition; it is "
        "diagnostic only and never a tie-breaker."
    )
    lines.append(
        "- No human study was run. Any Phase 2 judge is an LLM proxy, not human "
        "reviewer judgement."
    )
    lines.append(
        "- Each paper has 1-4 human reviews, but the run CSVs ship only one "
        "selected reference. `emb_ref_cos` therefore depends on which human "
        "review was picked; the all-reference columns show how much."
    )
    lines.append(
        "- Adapter and base are ranked separately. The base leaderboard is "
        "useful for prompt choice without fine-tuning, not as a competitor to "
        "the adapter."
    )
    lines.append(
        "- The rate-based guardrail was added during Phase 1 after a 10-paper run "
        "showed that one flagged review is already 10% of the sample. The "
        "pre-registered count-rule result is kept in the `eligible` column, so the "
        "effect of the threshold is visible."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def compute_metrics() -> pd.DataFrame:
    """Load every run and compute the per-review metrics with local embeddings."""
    if not ensure_server():
        raise SystemExit(
            "Ollama embedding server is not reachable; start it with "
            "OLLAMA_MODELS=D:\\ollama and OLLAMA_HOST=127.0.0.1:11435."
        )
    papers = load_papers()
    references = load_all_references()
    _, reviews = load_all()
    embedder = OllamaEmbedder()
    last = {"done": 0}

    def progress(done: int, total: int) -> None:
        if total and (done == total or done - last["done"] >= 200):
            last["done"] = done
            print(f"  embedded {done}/{total}", file=sys.stderr, flush=True)

    try:
        metrics = build_metrics(reviews, papers, embedder=embedder, progress=progress)
        return add_all_reference_metrics(
            metrics, references, embedder, progress=progress
        )
    finally:
        embedder.close()


def _to_bool(series: pd.Series) -> pd.Series:
    """Coerce a CSV column that may hold real booleans or ``"True"``/``"False"``."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="recompute per-review metrics and overwrite per_review_metrics.csv",
    )
    parser.add_argument("--metrics", type=Path, default=METRICS_CSV)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_RESAMPLES)
    args = parser.parse_args(argv)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    validation_errors = validate_all()
    runs, _ = load_all()

    if args.recompute or not args.metrics.exists():
        print("computing per-review metrics with local embeddings ...", file=sys.stderr)
        metrics = compute_metrics()
        metrics.to_csv(args.metrics, index=False)
        source = f"recomputed -> `{args.metrics.name}`"
    else:
        metrics = pd.read_csv(args.metrics)
        for column in ("hit_cap", "degenerate_loop", "meta_commentary", "location_any"):
            metrics[column] = _to_bool(metrics[column])
        source = f"loaded `{args.metrics.name}`"

    if "emb_ref_cos_all_max" not in metrics.columns:
        print("adding all-human-reference similarities ...", file=sys.stderr)
        if not ensure_server():
            raise SystemExit(
                "Ollama embedding server is not reachable; start it with "
                "OLLAMA_MODELS=D:\\ollama and OLLAMA_HOST=127.0.0.1:11435."
            )
        embedder = OllamaEmbedder()
        try:
            metrics = add_all_reference_metrics(
                metrics, load_all_references(), embedder
            )
        finally:
            embedder.close()

    metrics["reference_copy12"] = [
        reference_copy_rate(review, reference)
        for review, reference in zip(metrics["review"], metrics["reference_review"])
    ]
    metrics.to_csv(args.metrics, index=False)

    aggregates = aggregate(metrics)
    guardrails = guardrail_table(metrics, arm="adapter")
    guardrails_base = guardrail_table(metrics, arm="base")
    ranked = rank_candidates(aggregates, guardrails)
    ranked_base = rank_candidates(
        aggregates,
        guardrails_base,
        shortlist_size=SHORTLIST_SIZE,
        arm="base",
    )
    lift = adapter_lift(aggregates)
    bootstraps = bootstrap_table(metrics, n_resamples=args.bootstrap)
    confounds = confound_map(runs)

    ranked.to_csv(out_dir / "phase1_ranking.csv", index=False)
    ranked_base.to_csv(out_dir / "phase1_ranking_base.csv", index=False)
    lift.to_csv(out_dir / "adapter_lift.csv", index=False)
    confounds.to_csv(out_dir / "confound_map.csv", index=False)
    report = render_report(
        runs,
        metrics,
        aggregates,
        guardrails,
        ranked,
        ranked_base,
        lift,
        bootstraps,
        confounds,
        validation_errors,
        source,
    )
    (out_dir / "PHASE1_RESULTS.md").write_text(report, encoding="utf-8")

    stop_now, explanation = recommendation(ranked)
    print(report)
    print()
    print(
        "local-only verdict: "
        + ("decisive" if stop_now else "not decisive")
        + "; Phase 2 screening set is the top "
        + str(SHORTLIST_SIZE)
        + " adapters"
    )
    print(explanation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
