"""Choose the single adapter configuration for the last Kaggle run.

The ranking it produces is deliberately rule-based and declared here, before it looks
at the numbers, so the pick cannot be reverse-engineered from the result:

1. Hard gates, taken from the whole-run local metrics rather than the judged sample:
   degenerate loop rate <= 5%, meta-commentary rate <= 5%, cap rate <= 5%. A second gate
   uses the judge's own flags: invented locations, meta-commentary, or degenerate loops on
   more than 25% of the papers read make a review unusable, the same standard that
   disqualified `gpt_prompt_high_temp` in Phase 1.
2. Among the configurations that pass, rank by mean judge usefulness over the full set of
   papers that configuration was judged on (60 for `a_bit`, 40 for `very_big`, 20 or 10 for
   the rest).
3. Differences smaller than the measured noise band are treated as ties, and ties are
   broken by the widest paired comparison available between the tied configurations,
   then by generated-token efficiency, then by paper grounding, then by length fit.
4. Emit the exact decoding settings for the winning run so the Kaggle run reproduces it.

Run with ``python -m experiments.analysis.final_pick``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .load_runs import EXPERIMENTS_DIR, load_all

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
JUDGE_LOG = RESULTS_DIR / "judge_scores.jsonl"
RANKING_CSV = RESULTS_DIR / "phase1_ranking.csv"
SCORES_CSV = RESULTS_DIR / "phase2_scores.csv"
FINAL_MD = RESULTS_DIR / "FINAL_CONFIG.md"

CANDIDATE_ARM = "adapter"
MAX_DEGENERATE_RATE = 0.05
MAX_META_RATE = 0.05
MAX_CAP_RATE = 0.05
# Judged flags that make a review unusable for the authors, as a share of papers read.
MAX_JUDGE_FLAG_RATE = 0.25
# One paired standard error on the ten shared papers was ~4.3 points, so anything
# inside this band is not a real difference between configurations.
NOISE_BAND = 5.0
# The judged sample showed adapter reviews peaking in usefulness at 350-400 words.
TARGET_WORDS = (350, 400)

PROMPT_FILES = {
    "original": "original_prompt.txt",
    "gpt": "gpt_prompt.txt",
    "relaxed": "relaxed_references_prompt.txt",
    "tweaked": "tweaked_prompt.txt",
    "very_big": "very_big_prompt.txt",
    "a_bit": "a_bit_altered_prompt.txt",
    "new": "new_prompt.txt",
}


def load_records(path: Path = JUDGE_LOG) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if not path.exists():
        return pd.DataFrame()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def judged_scores(records: pd.DataFrame) -> pd.DataFrame:
    """Mean judge usefulness per (config, arm) on each paper-set size."""
    if records.empty:
        return pd.DataFrame(columns=["config", "arm", "n", "overall"])
    absolute = records[(records["mode"] == "absolute") & records["overall"].notna()].copy()
    if absolute.empty:
        return pd.DataFrame(columns=["config", "arm", "n", "overall"])
    absolute["overall"] = absolute["overall"].astype(float)
    grouped = (
        absolute.groupby(["config", "arm"])["overall"]
        .agg(n="count", overall="mean")
        .reset_index()
    )
    return grouped


def gate_table(local: pd.DataFrame) -> pd.DataFrame:
    """Apply the pre-registered hard gates to the aggregate local metrics.

    Two kinds of gate:
    - local rates over the whole generated run (degenerate loops, meta phrases, caps);
    - judged flag rates over the papers the judge actually read. A review that role-plays
      a human reviewer or invents section locations is unusable even if it scores well,
      which is the same standard that disqualified ``gpt_prompt_high_temp`` in Phase 1.
    """
    frame = local.copy()
    checks = {
        "degenerate_rate": MAX_DEGENERATE_RATE,
        "meta_rate": MAX_META_RATE,
        "cap_rate": MAX_CAP_RATE,
    }
    judged_checks = (
        ("invented_location_count", "judge_invented"),
        ("meta_commentary_count", "judge_meta"),
        ("degenerate_repetition_count", "judge_degenerate"),
    )
    failures: List[str] = []
    passed: List[bool] = []
    for _, row in frame.iterrows():
        reasons = []
        for column, limit in checks.items():
            if column not in frame.columns:
                continue
            value = row.get(column)
            if pd.notna(value) and float(value) > limit:
                reasons.append(f"{column}={float(value):.3f}>{limit}")
        judged = row.get("n")
        if pd.notna(judged) and float(judged) > 0:
            for column, label in judged_checks:
                if column not in frame.columns:
                    continue
                count = row.get(column)
                if pd.isna(count):
                    continue
                rate = float(count) / float(judged)
                if rate > MAX_JUDGE_FLAG_RATE:
                    reasons.append(f"{label}={rate:.0%}>{MAX_JUDGE_FLAG_RATE:.0%}")
        failures.append("; ".join(reasons))
        passed.append(not reasons)
    frame["gate_failures"] = failures
    frame["passes_gates"] = passed
    return frame


def candidate_table(scores: pd.DataFrame, local: pd.DataFrame) -> pd.DataFrame:
    """Join judged usefulness to local gates and efficiency, one row per config."""
    candidates = scores[(scores["arm"] == CANDIDATE_ARM)].copy()
    if candidates.empty:
        return candidates
    local_adapter = local[local.get("arm", CANDIDATE_ARM) == CANDIDATE_ARM].copy()
    keep = [
        "run",
        "n_papers",
        "length_fit",
        "emb_ref_cos",
        "emb_paper_cos",
        "structure_score",
        "words_mean",
        "repeat5_mean",
        "degenerate_rate",
        "meta_rate",
        "cap_rate",
        "composite_rank",
    ]
    local_adapter = local_adapter[[column for column in keep if column in local_adapter.columns]]
    merged = candidates.merge(
        local_adapter,
        left_on="config",
        right_on="run",
        how="left",
        suffixes=("", "_local"),
    ).drop(columns=["run"])
    merged = gate_table(merged)
    # Primary evidence is the full judged sample for each configuration, now that the two
    # finalists were judged on every paper they generated (60 and 40). The shared-ten mean
    # is kept as a diagnostic column because it is like-for-like across all fifteen runs.
    merged["score"] = merged["overall"]
    merged["score_basis"] = "all judged papers"
    return merged


def shared_paper_means(
    records: pd.DataFrame,
    config_a: str,
    config_b: str,
    arm: str = CANDIDATE_ARM,
) -> Optional[Dict[str, object]]:
    """Compare two configurations on the papers both were judged on.

    ``a_bit`` has sixty judged papers and ``very_big`` forty, and the forty are a
    subset of the sixty, so the largest paired comparison available is forty papers.
    """
    if records is None or records.empty:
        return None
    absolute = records[
        (records["mode"] == "absolute") & records["overall"].notna() & (records["arm"] == arm)
    ].copy()
    if absolute.empty:
        return None
    left = absolute[absolute["config"] == config_a]
    right = absolute[absolute["config"] == config_b]
    shared = set(left["paper_id"]) & set(right["paper_id"])
    if len(shared) < 2:
        return None
    left = left[left["paper_id"].isin(shared)]
    right = right[right["paper_id"].isin(shared)]
    mean_a = float(left["overall"].astype(float).mean())
    mean_b = float(right["overall"].astype(float).mean())
    return {
        "a": config_a,
        "b": config_b,
        "n": len(shared),
        "mean_a": mean_a,
        "mean_b": mean_b,
        "delta": mean_a - mean_b,
    }


def pick(
    table: pd.DataFrame,
    noise_band: float = NOISE_BAND,
    records: Optional[pd.DataFrame] = None,
) -> Tuple[Optional[str], str]:
    """Return (config, explanation) using the declared rule."""
    if table.empty:
        return None, "no judged adapter configurations"

    score_column = "score" if "score" in table.columns else "overall"
    eligible = table[table["passes_gates"]].copy()
    if eligible.empty:
        return None, "no adapter configuration passes the hard gates"

    eligible = eligible.sort_values(score_column, ascending=False).reset_index(drop=True)
    best = eligible.iloc[0]
    tied = eligible[eligible[score_column] >= float(best[score_column]) - noise_band]
    if len(tied) == 1:
        return str(best["config"]), (
            f"clearest mean usefulness ({float(best[score_column]):.1f} on {int(best['n'])} papers) "
            f"and passes every gate"
        )

    # A real head-to-head on the widest shared paper set outranks a proxy score.
    if len(tied) == 2:
        comparison = shared_paper_means(
            records, str(tied.iloc[0]["config"]), str(tied.iloc[1]["config"])
        )
        if comparison and abs(float(comparison["delta"])) >= noise_band:
            winner = comparison["a"] if float(comparison["delta"]) > 0 else comparison["b"]
            return str(winner), (
                f"{len(tied)} configurations fall inside the {noise_band:.0f}-point noise band, "
                f"so they were compared on the {int(comparison['n'])} papers both were judged on: "
                f"{float(comparison['mean_a']):.1f} versus {float(comparison['mean_b']):.1f}"
            )

    def efficiency(row: pd.Series) -> float:
        value = row.get("points_per_1k_output")
        return float(value) if pd.notna(value) else 0.0

    def grounding(row: pd.Series) -> float:
        value = row.get("emb_paper_cos")
        return float(value) if pd.notna(value) else 0.0

    def length_fit(row: pd.Series) -> float:
        value = row.get("length_fit")
        return float(value) if pd.notna(value) else 0.0

    ranked = tied.assign(
        _efficiency=tied.apply(efficiency, axis=1),
        _grounding=tied.apply(grounding, axis=1),
        _length_fit=tied.apply(length_fit, axis=1),
    ).sort_values(
        ["_efficiency", "_grounding", "_length_fit"],
        ascending=[False, False, False],
    )
    winner = ranked.iloc[0]
    return str(winner["config"]), (
        f"{len(tied)} configurations fall inside the {noise_band:.0f}-point noise band of the "
        f"best score, so the tie was broken on generated-token efficiency "
        f"({float(winner['_efficiency']):.1f} points per 1k output tokens)"
    )


def run_settings(run: str, runs: pd.DataFrame) -> Dict[str, object]:
    """Exact decoding settings and prompt file for a run name."""
    row = runs[runs["run"] == run]
    if row.empty:
        return {}
    record = row.iloc[0]
    family = str(record["prompt_family"])
    prompt_file = Path(EXPERIMENTS_DIR) / PROMPT_FILES.get(family, f"{family}_prompt.txt")
    prompt_chars = None
    if prompt_file.exists():
        prompt_chars = len(prompt_file.read_text(encoding="utf-8"))
    return {
        "run": run,
        "prompt_family": family,
        "prompt_file": str(prompt_file),
        "prompt_chars": prompt_chars,
        "temperature": float(record["temperature"]),
        "top_p": float(record["top_p"]),
        "top_k": int(record["top_k"]),
        "repetition_penalty": float(record["repetition_penalty"]),
        "no_repeat_ngram_size": int(record["no_repeat_ngram_size"]),
        "max_paper_tokens": int(record["max_paper_tokens"]),
        "max_new_tokens": int(record["max_new_tokens"]),
        "do_sample": bool(record["do_sample"]),
        "base_model": str(record["base_model"]),
        "adapter_dir": str(record["adapter_dir"]),
    }


def _fmt(value: object, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "—"
    return f"{number:.{digits}f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(str(item) for item in headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def head_to_head(winner: str, arm: str = CANDIDATE_ARM, path: Optional[Path] = None) -> List[str]:
    """Pairwise records involving the chosen configuration, as markdown bullets."""
    path = path or (RESULTS_DIR / "phase2_pairwise.csv")
    if not path.exists():
        return []
    table = pd.read_csv(path)
    label = f"{winner} [{arm}]"

    def name(value: object) -> str:
        text = str(value)
        return text.replace(" [", " (").rstrip("]") + ")" if text.endswith("]") else text

    lines: List[str] = []
    for row in table.itertuples(index=False):
        if str(row.config_a) == label:
            lines.append(
                f"- against {name(row.config_b)}: won {int(row.a_wins)}, "
                f"lost {int(row.b_wins)} of {int(row.n)}"
            )
        elif str(row.config_b) == label:
            lines.append(
                f"- against {name(row.config_a)}: won {int(row.b_wins)}, "
                f"lost {int(row.a_wins)} of {int(row.n)}"
            )
    return lines


def higher_scoring_elsewhere(table: pd.DataFrame, winner: str) -> List[str]:
    """Configurations that score above the winner on some judged sample."""
    if "overall" not in table.columns:
        return []
    chosen = table[table["config"] == winner]
    if chosen.empty:
        return []
    base_score = float(chosen.iloc[0]["overall"])
    lines: List[str] = []
    for row in table.sort_values("overall", ascending=False).itertuples(index=False):
        if str(row.config) == winner:
            continue
        if float(row.overall) <= base_score:
            continue
        lines.append(
            f"- `{row.config}` scores {float(row.overall):.1f} on {int(row.n)} judged papers "
            f"(the pick scores {base_score:.1f} on the same or fewer)"
        )
    return lines


def render_report(
    table: pd.DataFrame,
    winner: Optional[str],
    reason: str,
    settings: Dict[str, object],
    prompt_excerpt: str,
    alternative: Optional[Dict[str, object]] = None,
    head_to_head_lines: Optional[Sequence[str]] = None,
    higher_scores: Optional[Sequence[str]] = None,
) -> str:
    lines: List[str] = ["# Final configuration for the Kaggle run", ""]
    if winner is None:
        lines.append(f"No winner could be selected: {reason}.")
        return "\n".join(lines) + "\n"

    lines.append(f"**Run `{winner}` with the archived adapter, using the settings below.**")
    lines.append("")
    lines.append(f"Selection rule: {reason}.")
    lines.append("")
    lines.append("## Settings to reproduce")
    lines.append("")
    for key in (
        "prompt_file",
        "prompt_chars",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "no_repeat_ngram_size",
        "max_paper_tokens",
        "max_new_tokens",
    ):
        if key in settings:
            lines.append(f"- `{key}`: **{settings[key]}**")
    lines.append("")
    if prompt_excerpt:
        lines.append("Prompt opening as archived:")
        lines.append("")
        lines.append("```text")
        lines.append(prompt_excerpt.strip()[:600])
        lines.append("```")
        lines.append("")

    lines.append("## Candidate table")
    lines.append("")
    score_column = "score" if "score" in table.columns else "overall"
    basis = (
        str(table["score_basis"].iloc[0])
        if "score_basis" in table.columns and len(table)
        else "all judged papers"
    )
    lines.append(f"Judge score basis: **{basis}**.")
    lines.append("")
    ordered = table.sort_values(score_column, ascending=False)
    rows = [
        [
            row.get("config", ""),
            _fmt(row.get(score_column), 1),
            int(row.get("n", 0) or 0),
            _fmt(row.get("points_per_1k_output"), 1),
            _fmt(row.get("emb_paper_cos"), 3),
            _fmt(row.get("length_fit"), 3),
            _fmt(row.get("words_mean"), 0),
            "pass" if row.get("passes_gates") else f"fail ({row.get('gate_failures')})",
        ]
        for _, row in ordered.iterrows()
    ]
    lines.append(
        _table(
            ["config", "judge overall", "papers", "pts / 1k out", "paper cosine", "length fit", "words", "gates"],
            rows,
        )
    )
    lines.append("")
    if alternative:
        lines.append("## Quality-first alternative")
        lines.append("")
        lines.append(
            f"If judged usefulness is weighted above cost, the pick is "
            f"`{alternative['config']}` at {_fmt(alternative['score'], 1)} judged points "
            f"({alternative['words']} words on average, prompt "
            f"{alternative['prompt_chars']} characters against "
            f"{settings.get('prompt_chars')} for `{winner}`). The two are inside the "
            f"{NOISE_BAND:.0f}-point noise band, so this is a trade-off, not a missed winner."
        )
        lines.append("")
    if head_to_head_lines:
        lines.append("## Head-to-head record")
        lines.append("")
        lines.extend(head_to_head_lines)
        lines.append("")
    if higher_scores:
        lines.append("## Evidence that disagrees")
        lines.append("")
        lines.append(
            "These configurations scored higher than the pick on a judged sample. Treat the "
            "differences as unresolved: the paired standard error on ten papers is about 4 points, "
            "so these are ties broken by the declared rule rather than wins."
        )
        lines.append("")
        lines.extend(higher_scores)
        lines.append("")
    lines.append("## What this pick does not claim")
    lines.append("")
    lines.append(
        "- The judge is an LLM proxy, not a human review study. It is a consistent, "
        "rubric-fixed judge, but its absolute scores are not a peer-review standard."
    )
    lines.append(
        "- Configurations inside the noise band were separated by a paired comparison on "
        "the widest shared paper set, or by token efficiency when that comparison was also "
        "inside the band. Neither is a demonstrated quality difference."
    )
    lines.append(
        "- The fine-tuned adapter is still weaker than the untuned base model on judged "
        "usefulness. This pick is the best *adapter* configuration, which is what the "
        "Kaggle run needs; it is not a claim that fine-tuning beat the base model."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FINAL_MD)
    parser.add_argument("--scores", type=Path, default=SCORES_CSV)
    args = parser.parse_args(argv)

    local = pd.read_csv(RANKING_CSV)
    runs, _ = load_all()
    records = load_records()
    scores = pd.read_csv(args.scores) if args.scores.exists() else judged_scores(records)
    table = candidate_table(scores, local)
    winner, reason = pick(table, records=records)

    prompt_excerpt = ""
    settings: Dict[str, object] = {}
    alternative: Optional[Dict[str, object]] = None
    if winner is not None:
        settings = run_settings(winner, runs)
        prompt_path = Path(str(settings.get("prompt_file", "")))
        if prompt_path.exists():
            prompt_excerpt = prompt_path.read_text(encoding="utf-8")

        scored = table.sort_values("score", ascending=False)
        best_quality = scored.iloc[0]
        if str(best_quality["config"]) != winner:
            alt_settings = run_settings(str(best_quality["config"]), runs)
            alternative = {
                "config": str(best_quality["config"]),
                "score": float(best_quality["score"]),
                "words": int(round(float(best_quality["words_mean"]))) if pd.notna(best_quality.get("words_mean")) else "—",
                "prompt_chars": alt_settings.get("prompt_chars", "—"),
            }

    report = render_report(
        table,
        winner,
        reason,
        settings,
        prompt_excerpt,
        alternative,
        head_to_head(winner) if winner else None,
        higher_scoring_elsewhere(table, winner) if winner else None,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"winner: {winner} ({reason})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
