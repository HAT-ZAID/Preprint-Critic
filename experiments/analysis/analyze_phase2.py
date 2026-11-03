"""Phase 2 analysis: judge scores, paired bootstrap, Bradley-Terry, decision.

Reads ``experiments/results/judge_scores.jsonl`` (written by
``experiments.analysis.judge``) and writes:

- ``experiments/results/phase2_scores.csv``
- ``experiments/results/phase2_pairwise.csv``
- ``experiments/results/decision.md``

No model is called here.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .load_runs import COMMON_10

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
JUDGE_LOG = RESULTS_DIR / "judge_scores.jsonl"
METRICS_CSV = RESULTS_DIR / "per_review_metrics.csv"
SCORES_CSV = RESULTS_DIR / "phase2_scores.csv"
PAIRWISE_CSV = RESULTS_DIR / "phase2_pairwise.csv"
DECISION_MD = RESULTS_DIR / "decision.md"

SUBFIELDS = ("factual", "grounding", "coverage", "actionability", "argument", "calibration", "length")
CRITICAL_FLAGS = ("invented_location", "meta_commentary", "degenerate_repetition")
CRITICAL_COUNT = 2
CAP_RATE_LIMIT = 0.05
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42


# --------------------------------------------------------------------------
# loading


def load_records(path: Path = JUDGE_LOG) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not record.get("parse_ok"):
                continue
            rows.append(record)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"no valid judge records in {path}")
    return frame


def label(row) -> str:
    return f"{row['config']} [{row['arm']}]"


def absolute_table(records: pd.DataFrame) -> pd.DataFrame:
    absolute = records[records["mode"] == "absolute"].copy()
    if absolute.empty:
        raise ValueError("no absolute judge records")

    rows: List[Dict[str, object]] = []
    for (config, arm), group in absolute.groupby(["config", "arm"]):
        row: Dict[str, object] = {
            "config": config,
            "arm": arm,
            "n": len(group),
            "overall": group["overall"].mean(),
            "overall_median": group["overall"].median(),
            "overall_sd": group["overall"].std(ddof=1) if len(group) > 1 else 0.0,
        }
        for field in SUBFIELDS:
            row[field] = group[field].astype(float).mean()
        for flag in ("invented_location", "generic_filler", "meta_commentary", "degenerate_repetition"):
            row[f"{flag}_count"] = int(sum(1 for value in group["flags"] if value.get(flag)))
        rows.append(row)

    table = pd.DataFrame(rows)
    table["critical_failures"] = table.apply(_critical_failures, axis=1)
    table["eligible"] = table["critical_failures"].apply(lambda names: len(names) == 0)
    table = table.sort_values(["eligible", "overall"], ascending=[False, False]).reset_index(drop=True)
    return table


def _critical_failures(row) -> List[str]:
    failures = []
    for flag in CRITICAL_FLAGS:
        count = int(row.get(f"{flag}_count", 0))
        if count >= CRITICAL_COUNT:
            failures.append(f"{flag} {count}/{int(row['n'])}")
    return failures


def cap_rates(path: Path = METRICS_CSV) -> Dict[Tuple[str, str], float]:
    if not path.exists():
        return {}
    metrics = pd.read_csv(path, usecols=["run", "arm", "hit_cap"])
    metrics["hit_cap"] = metrics["hit_cap"].astype(str).str.lower().isin({"true", "1"})
    rates = metrics.groupby(["run", "arm"])["hit_cap"].mean()
    return {tuple(key): float(value) for key, value in rates.items()}


# --------------------------------------------------------------------------
# statistics


def load_cost_frame(root: Optional[Path] = None) -> pd.DataFrame:
    """Per-review token and word costs, taken from the run CSVs.

    ``per_review_metrics.csv`` does not carry ``prompt_tokens``, so the run CSVs are the
    authoritative source for token counts. Only the numeric columns are kept.
    """
    from .load_runs import load_all

    _, reviews = load_all() if root is None else load_all(root)
    columns = ["run", "arm", "paper_id", "prompt_tokens", "output_tokens", "words"]
    return reviews[columns].rename(columns={"run": "config"})


def token_costs(records: pd.DataFrame, costs: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Mean token cost per judged (config, arm), restricted to the papers that were judged.

    The restriction matters: judged configs do not all cover the same paper count, so a
    cost taken over the whole run would be measured on a different paper sample than the
    judge score it is divided into.
    """
    columns = ["config", "arm", "output_tokens", "prompt_tokens", "words"]
    absolute = records[records["mode"] == "absolute"]
    if absolute.empty:
        return pd.DataFrame(columns=columns)

    if costs is None:
        try:
            costs = load_cost_frame()
        except Exception:
            return pd.DataFrame(columns=columns)
    if costs.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for (config, arm), group in absolute.groupby(["config", "arm"]):
        papers = set(group["paper_id"].astype(str))
        subset = costs[
            (costs["config"] == config)
            & (costs["arm"] == arm)
            & (costs["paper_id"].astype(str).isin(papers))
        ]
        if subset.empty:
            continue
        rows.append(
            {
                "config": config,
                "arm": arm,
                "output_tokens": float(subset["output_tokens"].mean()),
                "prompt_tokens": float(subset["prompt_tokens"].mean()),
                "words": float(subset["words"].mean()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def attach_token_efficiency(table: pd.DataFrame, costs: pd.DataFrame) -> pd.DataFrame:
    """Add cost columns and the two efficiency ratios to the absolute-score table."""
    columns = ["output_tokens", "prompt_tokens", "words", "points_per_1k_output", "points_per_1k_total"]
    if costs.empty:
        for column in columns:
            table[column] = float("nan")
        return table
    table = table.merge(costs, on=["config", "arm"], how="left")
    table["points_per_1k_output"] = table["overall"] / table["output_tokens"] * 1000.0
    table["points_per_1k_total"] = table["overall"] / (table["output_tokens"] + table["prompt_tokens"]) * 1000.0
    return table


def common_paper_scores(
    records: pd.DataFrame, paper_ids: Sequence[str] = COMMON_10
) -> pd.DataFrame:
    """Mean usefulness on the papers every run shares, so configs compare like with like.

    After the coverage expansion some configurations are judged on twenty papers and
    others on ten. Averaging those directly would compare different paper sets, so the
    shared-ten mean is reported next to the full mean.
    """
    columns = ["config", "arm", "overall_common", "n_common"]
    if records.empty:
        return pd.DataFrame(columns=columns)
    absolute = records[(records["mode"] == "absolute") & records["overall"].notna()].copy()
    absolute = absolute[absolute["paper_id"].isin(list(paper_ids))]
    if absolute.empty:
        return pd.DataFrame(columns=columns)
    absolute["overall"] = absolute["overall"].astype(float)
    return (
        absolute.groupby(["config", "arm"])["overall"]
        .agg(overall_common="mean", n_common="count")
        .reset_index()
    )


def attach_common_paper_scores(table: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    """Merge the shared-paper mean onto the absolute table."""
    scores = common_paper_scores(records)
    if scores.empty:
        table["overall_common"] = float("nan")
        table["n_common"] = 0
        return table
    return table.merge(scores, on=["config", "arm"], how="left")


def paired_values(absolute: pd.DataFrame, config_a: str, arm_a: str, config_b: str, arm_b: str):
    left = absolute[(absolute["config"] == config_a) & (absolute["arm"] == arm_a)].set_index("paper_id")["overall"]
    right = absolute[(absolute["config"] == config_b) & (absolute["arm"] == arm_b)].set_index("paper_id")["overall"]
    joined = pd.concat([left.rename("a"), right.rename("b")], axis=1).dropna()
    return joined


def bootstrap_diff(values: np.ndarray, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED):
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[draws].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(low), float(high)


def paired_bootstrap(absolute: pd.DataFrame, config_a: str, arm_a: str, config_b: str, arm_b: str) -> Dict[str, float]:
    joined = paired_values(absolute, config_a, arm_a, config_b, arm_b)
    diff = (joined["a"] - joined["b"]).to_numpy(dtype=float)
    mean, low, high = bootstrap_diff(diff)
    return {
        "n": len(diff),
        "diff": mean,
        "ci_low": low,
        "ci_high": high,
        "win_rate": float((diff > 0).mean()) if len(diff) else float("nan"),
    }


def bradley_terry(rows: Iterable[Tuple[str, str, float]]) -> pd.Series:
    """MM algorithm. Each row is (winner, loser, weight); ties pass two rows of 0.5."""
    wins: Counter = Counter()
    meetings: Counter = Counter()
    items: List[str] = []
    for winner, loser, weight in rows:
        if winner == loser:
            continue
        for item in (winner, loser):
            if item not in items:
                items.append(item)
        wins[winner] += weight
        meetings[(winner, loser)] += weight
    if not items:
        return pd.Series(dtype=float)

    strength = {item: 1.0 for item in items}
    prior = 0.5  # mild regularisation so an undefeated item cannot run away
    for _ in range(1000):
        updated = {}
        for item in items:
            denominator = prior / (strength[item] + 1.0)
            for (winner, loser), weight in meetings.items():
                if item == winner:
                    opponent = loser
                elif item == loser:
                    opponent = winner
                else:
                    continue
                denominator += weight / (strength[item] + strength[opponent])
            updated[item] = (wins[item] + prior) / denominator if denominator > 0 else strength[item]
        mean = math.exp(sum(math.log(value) for value in updated.values() if value > 0) / len(updated))
        updated = {item: value / mean for item, value in updated.items()}
        delta = max(abs(updated[item] - strength[item]) for item in items)
        strength = updated
        if delta < 1e-9:
            break
    return pd.Series(strength).sort_values(ascending=False)


def pairwise_frame(records: pd.DataFrame) -> pd.DataFrame:
    pairs = records[records["mode"] == "pairwise"].copy()
    pairs["left"] = pairs["config"] + " [" + pairs["arm"] + "]"
    pairs["right"] = pairs["opponent"] + " [" + pairs["opponent_arm"] + "]"
    pairs["winner_label"] = [
        "tie" if row.winner_arm == "tie" else f"{row.winner_config} [{row.winner_arm}]"
        for row in pairs.itertuples(index=False)
    ]
    return pairs


def win_matrix(pairs: pd.DataFrame) -> pd.DataFrame:
    tally: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0, 0])
    for row in pairs.itertuples(index=False):
        key = (row.left, row.right)
        if row.winner_label == "tie":
            tally[key][2] += 1
        elif row.winner_label == row.left:
            tally[key][0] += 1
        else:
            tally[key][1] += 1
    rows = [
        {"config_a": left, "config_b": right, "a_wins": w, "b_wins": l, "ties": t, "n": w + l + t}
        for (left, right), (w, l, t) in tally.items()
    ]
    return pd.DataFrame(rows).sort_values(["config_a", "config_b"]).reset_index(drop=True)


def audit_agreement(records: pd.DataFrame) -> Dict[str, object]:
    pairs = records[records["mode"] == "pairwise"].set_index("key")
    audits = records[records["mode"] == "pairwise_audit"]
    checked = agree = ties = 0
    details = []
    for row in audits.itertuples(index=False):
        original = pairs.loc[row.paired_key] if row.paired_key in pairs.index else None
        if original is None:
            continue
        if "tie" in (original["winner_arm"], row.winner_arm):
            ties += 1
            checked += 1
            continue
        same = (original["winner_config"], original["winner_arm"]) == (row.winner_config, row.winner_arm)
        agree += int(same)
        checked += 1
        details.append(
            {
                "paper_id": row.paper_id,
                "pair": f"{row.config} [{row.arm}] vs {row.opponent} [{row.opponent_arm}]",
                "original": f"{original['winner_config']} [{original['winner_arm']}]",
                "audit": f"{row.winner_config} [{row.winner_arm}]",
                "agree": same,
            }
        )
    rate = agree / checked if checked else float("nan")
    return {"checked": checked, "agree": agree, "ties": ties, "rate": rate, "details": details}


# --------------------------------------------------------------------------
# report


def _fmt(value, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{value:.{digits}f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def render_decision(
    table: pd.DataFrame,
    win: pd.DataFrame,
    strengths: pd.Series,
    audit: Dict[str, object],
    bootstraps: Dict[str, Dict[str, float]],
    caps: Dict[Tuple[str, str], float],
    calls: int,
) -> str:
    adapters = table[table["arm"] == "adapter"].sort_values("overall", ascending=False)
    bases = table[table["arm"] == "base"].sort_values("overall", ascending=False)
    best_adapter = adapters.iloc[0]
    best_base = bases.iloc[0] if not bases.empty else None
    overall_best = table.sort_values("overall", ascending=False).iloc[0]
    least_severe = table.sort_values(["severity", "overall"], ascending=[True, False]).iloc[0]

    lines: List[str] = []
    lines.append("# Phase 2 decision: LLM judge on the top-7 adapter screening set\n")
    lines.append(
        f"Judge model `deepseek-v4.1-flash` at the default variant, stateless one-shot calls, "
        f"{calls} scored calls. No human study was run; the judge is a proxy, not ground truth.\n"
    )

    lines.append("## Verdict\n")
    lines.append(
        f"- **Highest judged usefulness: `{overall_best['config']}` [{overall_best['arm']}] at "
        f"{_fmt(overall_best['overall'])}/100.**"
    )
    if best_base is not None and overall_best["arm"] == "base":
        gap = overall_best["overall"] - best_adapter["overall"]
        lines.append(
            f"- **Fine-tuning did not help on this judge.** The best adapter, `{best_adapter['config']}`, "
            f"scores {_fmt(best_adapter['overall'])}, which is {_fmt(gap)} points below the best base arm, "
            f"`{best_base['config']}`, at {_fmt(best_base['overall'])}."
        )
    else:
        lines.append(
            f"- Best adapter `{best_adapter['config']}` at {_fmt(best_adapter['overall'])}; "
            f"best base `{best_base['config']}` at {_fmt(best_base['overall'])}."
        )
    lines.append(
        f"- **No configuration passes the pre-registered guardrails.** The least severe is "
        f"`{least_severe['config']}` [{least_severe['arm']}] with severity {int(least_severe['severity'])} "
        f"({'; '.join(least_severe['critical_failures']) if least_severe['critical_failures'] else 'no flagged rule'}); "
        "the rule is applied as written rather than relaxed after the fact."
    )
    lines.append("")


    lines.append("## Absolute rubric\n")
    has_common = "overall_common" in table.columns and table["overall_common"].notna().any()
    header = ["config", "arm", "n", "overall"]
    if has_common:
        header += ["n common", "overall (10 shared)"]
    header += [*SUBFIELDS, "eligible", "failures"]
    rows = []
    for row in table.itertuples(index=False):
        cells = [row.config, row.arm, row.n, _fmt(row.overall)]
        if has_common:
            cells += [
                getattr(row, "n_common", "–"),
                _fmt(getattr(row, "overall_common", float("nan"))),
            ]
        cells += [
            *[_fmt(getattr(row, field)) for field in SUBFIELDS],
            "yes" if row.eligible else "no",
            "; ".join(row.critical_failures) if row.critical_failures else "–",
        ]
        rows.append(cells)
    lines.append(_table(header, rows))
    if has_common:
        lines.append("")
        lines.append(
            "`overall (10 shared)` restricts each configuration to the ten papers that every run "
            "generated, so configurations judged on twenty papers can be compared like with like. "
            "The plain `overall` column uses every judged paper for that configuration."
        )
    lines.append("")

    lines.append("## Quality per token\n")
    if (
        "points_per_1k_output" in table.columns
        and table["points_per_1k_output"].notna().any()
    ):
        token_rows = []
        for row in table.sort_values("points_per_1k_output", ascending=False).itertuples(index=False):
            token_rows.append(
                [
                    row.config,
                    row.arm,
                    _fmt(row.overall),
                    _fmt(row.words, 0),
                    _fmt(row.output_tokens, 0),
                    _fmt(row.prompt_tokens, 0),
                    _fmt(row.points_per_1k_output, 1),
                    _fmt(row.points_per_1k_total, 2),
                ]
            )
        lines.append(
            _table(
                [
                    "config",
                    "arm",
                    "overall",
                    "words",
                    "output tokens",
                    "prompt tokens",
                    "pts / 1k output",
                    "pts / 1k total",
                ],
                token_rows,
            )
        )
        lines.append("")
        best_efficient = table.sort_values("points_per_1k_output", ascending=False).iloc[0]
        best_total = table.sort_values("points_per_1k_total", ascending=False).iloc[0]
        lines.append(
            f"- Most efficient on generated tokens: `{best_efficient['config']}` [{best_efficient['arm']}] at "
            f"{_fmt(best_efficient['points_per_1k_output'])} usefulness points per 1,000 output tokens."
        )
        lines.append(
            f"- Most efficient on all tokens: `{best_total['config']}` [{best_total['arm']}] at "
            f"{_fmt(best_total['points_per_1k_total'], 2)} points per 1,000 total tokens."
        )
        lines.append(
            "- These two answers disagree on purpose. The paper prompt is roughly 17,500 tokens and is "
            "identical for both arms, so it dominates any total-token count. The adapter saves output "
            "tokens only; it does not save the paper reading. Per generated token the adapter is the "
            "cheaper reviewer, per end-to-end token the longer base review still returns more usefulness "
            "per token spent."
        )
        lines.append(
            "- The score is a 0-100 rubric total, not a linear quantity of value, so a ratio of score to "
            "tokens is a coarse efficiency index, not a budget forecast."
        )
    else:
        lines.append("Per-review token columns were unavailable, so efficiency ratios were not computed.")
    lines.append("")

    lines.append("## Flag counts\n")
    flag_rows = []
    for row in table.itertuples(index=False):
        flag_rows.append(
            [
                row.config,
                row.arm,
                row.invented_location_count,
                row.meta_commentary_count,
                row.degenerate_repetition_count,
                row.generic_filler_count,
                _fmt(caps.get((row.config, row.arm), float("nan")), 3),
            ]
        )
    lines.append(_table(["config", "arm", "invented", "meta", "degenerate", "filler", "cap_rate"], flag_rows))
    lines.append("")

    if not win.empty:
        lines.append("## Pairwise\n")
        lines.append(
            _table(
                ["A", "B", "A wins", "B wins", "ties"],
                [
                    [r.config_a, r.config_b, r.a_wins, r.b_wins, r.ties]
                    for r in win.itertuples(index=False)
                ],
            )
        )
        lines.append("")
        lines.append("## Bradley-Terry strength\n")
        lines.append(
            _table(
                ["rank", "config", "strength"],
                [[index + 1, name, _fmt(value, 2)] for index, (name, value) in enumerate(strengths.items())],
            )
        )
        lines.append("")

    lines.append("## Paired bootstrap (adapter vs base, 10,000 resamples)\n")
    if bootstraps:
        lines.append(
            _table(
                ["comparison", "n", "mean diff", "95% CI", "adapter win rate"],
                [
                    [
                        name,
                        stats["n"],
                        _fmt(stats["diff"]),
                        f"[{_fmt(stats['ci_low'])}, {_fmt(stats['ci_high'])}]",
                        f"{stats['win_rate']*100:.0f}%",
                    ]
                    for name, stats in bootstraps.items()
                ],
            )
        )
    else:
        lines.append("No adapter/base pair survived in the judged set.")
    lines.append("")

    lines.append("## Order-swap audit\n")
    lines.append(
        f"Repeated {audit['checked']} pairwise calls with A/B reversed: "
        f"**{audit['agree']}/{audit['checked']} agreed** "
        f"({audit['rate']*100:.0f}%), {audit['ties']} tie(s)."
    )
    if audit["rate"] == audit["rate"] and audit["rate"] < 0.8:
        lines.append("Agreement is below the 80% threshold, so pairwise results are reported as order-sensitive.")
    lines.append("")

    lines.append("## Limitations\n")
    counts = sorted(set(int(value) for value in table.get("n", pd.Series(dtype=int)).dropna()))
    paper_text = "Ten papers" if not counts or max(counts) <= 10 else f"{min(counts)}–{max(counts)} papers"
    lines.append(
        f"- {paper_text}, all used earlier for prompt selection; they are dev papers, not a holdout."
    )
    if len(counts) > 1:
        lines.append(
            f"- Judged paper counts differ across configurations ({', '.join(str(value) for value in counts)}), "
            "because only the shorter runs share the full set. The `overall (10 shared)` column is the "
            "like-for-like comparison; the plain `overall` column is not comparable across differing counts."
        )
    lines.append("- One LLM judge, one prompt version, no repeated sampling of the same review, so judge variance is unmeasured.")
    lines.append("- The absolute rubric weights coverage and grounding, which structurally favours longer reviews.")
    lines.append("- Base arms are longer (roughly 1,000–2,000 more words) and slower at generation time; the efficiency table above is the only token accounting, and it is an index rather than a cost model.")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# cli


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 judge analysis")
    parser.add_argument("--log", type=Path, default=JUDGE_LOG)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    records = load_records(args.log)
    table = absolute_table(records)
    caps = cap_rates()
    table["cap_rate"] = [caps.get((row.config, row.arm), float("nan")) for row in table.itertuples(index=False)]
    table["severity"] = [
        len(row.critical_failures)
        + (1 if (row.cap_rate == row.cap_rate and row.cap_rate > CAP_RATE_LIMIT) else 0)
        for row in table.itertuples(index=False)
    ]
    table["severity"] = table["severity"].astype(int)
    table = attach_token_efficiency(table, token_costs(records))
    table = attach_common_paper_scores(table, records)
    table.to_csv(args.out_dir / SCORES_CSV.name, index=False)

    pairs = pairwise_frame(records)
    win = win_matrix(pairs)
    win.to_csv(args.out_dir / PAIRWISE_CSV.name, index=False)

    strength_rows: List[Tuple[str, str, float]] = []
    for row in pairs.itertuples(index=False):
        if row.winner_label == "tie":
            strength_rows.append((row.left, row.right, 0.5))
            strength_rows.append((row.right, row.left, 0.5))
        elif row.winner_label == row.left:
            strength_rows.append((row.left, row.right, 1.0))
        else:
            strength_rows.append((row.right, row.left, 1.0))
    strengths = bradley_terry(strength_rows)

    audit = audit_agreement(records)

    absolute = records[records["mode"] == "absolute"]
    bootstraps: Dict[str, Dict[str, float]] = {}
    for config in sorted(set(absolute["config"])):
        has_adapter = ((absolute["config"] == config) & (absolute["arm"] == "adapter")).any()
        has_base = ((absolute["config"] == config) & (absolute["arm"] == "base")).any()
        if has_adapter and has_base:
            bootstraps[f"{config}: adapter vs base"] = paired_bootstrap(absolute, config, "adapter", config, "base")

    report = render_decision(table, win, strengths, audit, bootstraps, caps, len(records))
    (args.out_dir / DECISION_MD.name).write_text(report, encoding="utf-8")

    print(table[["config", "arm", "n", "overall", "eligible"]].to_string(index=False))
    print()
    print(strengths.to_string())
    print()
    print(f"audit agreement {audit['agree']}/{audit['checked']} ({audit['rate']*100:.0f}%)")
    for name, stats in bootstraps.items():
        print(
            f"{name}: {stats['diff']:+.1f} [{stats['ci_low']:+.1f}, {stats['ci_high']:+.1f}] "
            f"win {stats['win_rate']*100:.0f}% n={stats['n']}"
        )
    print(f"wrote {args.out_dir / SCORES_CSV.name}, {args.out_dir / PAIRWISE_CSV.name}, {args.out_dir / DECISION_MD.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
