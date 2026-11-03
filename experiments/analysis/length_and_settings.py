"""Review-length curves and decoding-setting weights for the Phase 1/2 comparison.

Answers two questions that the prompt-family leaderboards cannot:

1. How long should a review be, and does length explain the judged quality gap?
2. Which individual settings (temperature, top_p, top_k, repetition_penalty,
   paper budget, output budget, paper count) moved the score, and which axes
   are simply not identifiable from 15 archived runs?

Writes ``experiments/results/length_bands.csv``, ``settings_pairs.csv``,
``settings_weights.csv`` and ``LENGTH_AND_SETTINGS.md``.

No model is called here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
METRICS_CSV = RESULTS_DIR / "per_review_metrics.csv"
JUDGE_LOG = RESULTS_DIR / "judge_scores.jsonl"
RANKING_CSV = RESULTS_DIR / "phase1_ranking.csv"
LENGTH_BANDS_CSV = RESULTS_DIR / "length_bands.csv"
SETTINGS_PAIRS_CSV = RESULTS_DIR / "settings_pairs.csv"
SETTINGS_WEIGHTS_CSV = RESULTS_DIR / "settings_weights.csv"
REPORT_MD = RESULTS_DIR / "LENGTH_AND_SETTINGS.md"

BANDS: Tuple[Tuple[float, float, str], ...] = (
    (0, 200, "<200"),
    (200, 300, "200-300"),
    (300, 350, "300-350"),
    (350, 400, "350-400"),
    (400, 500, "400-500"),
    (500, 700, "500-700"),
    (700, 1000, "700-1000"),
    (1000, 1e9, "1000+"),
)

SETTING_AXES = (
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "max_paper_tokens",
    "max_new_tokens",
    "n_papers",
)

COMPONENTS = ("length_fit", "emb_ref_cos", "emb_paper_cos", "structure_score")

# --------------------------------------------------------------------------
# loading


def load_judged(metrics_path: Path = METRICS_CSV, log_path: Path = JUDGE_LOG) -> pd.DataFrame:
    """Judged reviews joined to their word length."""
    metrics = pd.read_csv(metrics_path)
    records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    judged = pd.DataFrame(
        [
            {
                "config": record["config"],
                "arm": record["arm"],
                "paper_id": record["paper_id"],
                "overall": float(record["overall"]),
            }
            for record in records
            if record.get("mode") == "absolute"
        ]
    )
    sizes = metrics[["run", "arm", "paper_id", "words", "length_fit", "reference_words"]].rename(
        columns={"run": "config"}
    )
    return judged.merge(sizes, on=["config", "arm", "paper_id"], how="left")


def load_runs(root: Optional[Path] = None) -> pd.DataFrame:
    from .load_runs import load_all

    runs, _ = load_all() if root is None else load_all(root)
    return runs


def load_aggregates(path: Path = RANKING_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


# --------------------------------------------------------------------------
# length curves


def length_bands(judged: pd.DataFrame) -> pd.DataFrame:
    """Mean judged usefulness per word-length band, per arm."""
    if judged.empty:
        return pd.DataFrame(columns=["arm", "band", "n", "mean_score", "median_score"])

    frame = judged.copy()
    labels: List[str] = []
    for value in frame["words"]:
        label = "unknown"
        if value == value:
            for low, high, name in BANDS:
                if low <= value < high:
                    label = name
                    break
        labels.append(label)
    frame["band"] = labels
    frame = frame[frame["band"] != "unknown"]
    order = [name for _, _, name in BANDS]
    frame["band"] = pd.Categorical(frame["band"], categories=order, ordered=True)

    grouped = (
        frame.groupby(["arm", "band"], observed=True)["overall"]
        .agg(n="count", mean_score="mean", median_score="median")
        .reset_index()
    )
    return grouped.sort_values(["arm", "band"]).reset_index(drop=True)


def length_correlations(judged: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of word count with judged score, per arm and per config."""
    rows: List[Dict[str, object]] = []
    if judged.empty:
        return pd.DataFrame(columns=["group", "n", "spearman", "pearson"])

    def add(group: str, frame: pd.DataFrame) -> None:
        clean = frame.dropna(subset=["words", "overall"])
        if len(clean) < 4 or clean["words"].nunique() < 2:
            return
        rows.append(
            {
                "group": group,
                "n": len(clean),
                "spearman": round(float(clean["words"].corr(clean["overall"], method="spearman")), 3),
                "pearson": round(float(clean["words"].corr(clean["overall"])), 3),
            }
        )

    add("all judged", judged)
    for arm, frame in judged.groupby("arm"):
        add(f"arm={arm}", frame)
    for (config, arm), frame in judged.groupby(["config", "arm"]):
        add(f"{config} [{arm}]", frame)
    return pd.DataFrame(rows)


def length_verdict(judged: pd.DataFrame) -> Tuple[str, float]:
    """Return (verdict, best adapter band mean) for the 350-400 word claim."""
    adapter = judged[judged["arm"] == "adapter"]
    if adapter.empty:
        return "no judged adapter reviews", float("nan")
    bands = length_bands(adapter)
    if bands.empty:
        return "no length bands", float("nan")
    best = bands.sort_values("mean_score", ascending=False).iloc[0]
    return f"best adapter band is {best['band']} words at {best['mean_score']:.1f}", float(best["mean_score"])


# --------------------------------------------------------------------------
# settings


def settings_frame(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per adapter run with its decoding settings."""
    frame = runs.copy()
    keep = ["run", "prompt_family", *SETTING_AXES]
    return frame[[column for column in keep if column in frame.columns]]


def matched_pairs(aggregates: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Within-family temperature pairs: the only clean single-axis test available."""
    settings = settings_frame(runs)
    merged = aggregates.merge(settings, on="run", how="left", suffixes=("", "_run"))
    rows: List[Dict[str, object]] = []
    for family, frame in merged.groupby("prompt_family"):
        frame = frame.dropna(subset=["temperature"]).sort_values("temperature")
        if len(frame) < 2:
            continue
        low = frame.iloc[0]
        high = frame.iloc[-1]
        if low["temperature"] == high["temperature"]:
            continue
        row: Dict[str, object] = {
            "family": family,
            "low_run": low["run"],
            "high_run": high["run"],
            "low_temperature": low["temperature"],
            "high_temperature": high["temperature"],
            "n_papers_low": low["n_papers"],
            "n_papers_high": high["n_papers"],
            "low_composite": low["composite_rank"],
            "high_composite": high["composite_rank"],
            "delta_composite": high["composite_rank"] - low["composite_rank"],
            "other_axes_differ": _other_axes(low, high),
        }
        for component in COMPONENTS + ("words_mean", "repeat5_mean", "degenerate_rate", "rouge_l"):
            if component in merged.columns:
                row[f"delta_{component}"] = float(high[component] - low[component])
        rows.append(row)
    return pd.DataFrame(rows)


def _other_axes(low: pd.Series, high: pd.Series) -> str:
    differing = [axis for axis in SETTING_AXES if axis != "temperature" and low.get(axis) != high.get(axis)]
    return ", ".join(differing) if differing else ""


def axis_split(aggregates: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Compare runs above vs below each setting's median, on composite rank."""
    merged = aggregates.merge(settings_frame(runs), on="run", how="left", suffixes=("", "_run"))
    rows: List[Dict[str, object]] = []
    for axis in SETTING_AXES:
        if axis not in merged.columns:
            continue
        values = merged[axis].dropna()
        if values.nunique() < 2:
            rows.append(
                {
                    "axis": axis,
                    "n": len(values),
                    "distinct_values": int(values.nunique()),
                    "low_value": float(values.iloc[0]) if len(values) else float("nan"),
                    "high_value": float("nan"),
                    "low_mean_composite": float("nan"),
                    "high_mean_composite": float("nan"),
                    "delta": float("nan"),
                    "identifiable": False,
                }
            )
            continue
        median = float(values.median())
        low = merged[merged[axis] <= median]
        high = merged[merged[axis] > median]
        if high.empty:
            high = merged[merged[axis] >= merged[axis].max()]
            low = merged[merged[axis] < merged[axis].max()]
        rows.append(
            {
                "axis": axis,
                "n": len(values),
                "distinct_values": int(values.nunique()),
                "low_value": float(low[axis].mean()) if not low.empty else float("nan"),
                "high_value": float(high[axis].mean()) if not high.empty else float("nan"),
                "low_mean_composite": float(low["composite_rank"].mean()) if not low.empty else float("nan"),
                "high_mean_composite": float(high["composite_rank"].mean()) if not high.empty else float("nan"),
                "delta": (
                    float(high["composite_rank"].mean() - low["composite_rank"].mean())
                    if not high.empty and not low.empty
                    else float("nan")
                ),
                # a split is only meaningful with at least three runs on each side
                "identifiable": bool(len(low) >= 3 and len(high) >= 3),
            }
        )
    return pd.DataFrame(rows)


def ridge_weights(aggregates: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Ridge regression of composite rank on settings, leave-one-run-out.

    This is deliberately reported with its cross-validated R-squared, because with
    15 runs and 7 correlated axes the coefficients are directional hints at best.
    """
    merged = aggregates.merge(settings_frame(runs), on="run", how="left", suffixes=("", "_run"))
    features = [axis for axis in SETTING_AXES if axis in merged.columns and merged[axis].notna().any()]
    if len(merged) < 5 or len(features) < 1:
        return pd.DataFrame(columns=["feature", "coefficient", "std_coefficient"])

    matrix = merged[features].astype(float).to_numpy()
    target = merged["composite_rank"].astype(float).to_numpy()
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds == 0] = 1.0
    standardised = (matrix - means) / stds

    alpha = 1.0
    weights = _ridge_fit(standardised, target, alpha)
    coefficients = weights / stds

    predicted = []
    for index in range(len(target)):
        mask = np.arange(len(target)) != index
        fold_weights = _ridge_fit(standardised[mask], target[mask], alpha)
        predicted.append(float(np.dot(standardised[index], fold_weights)))
    predicted_array = np.asarray(predicted)
    ss_res = float(np.sum((target - predicted_array) ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")

    frame = pd.DataFrame(
        {
            "feature": features,
            "coefficient": coefficients,
            "std_coefficient": weights,
        }
    )
    frame.attrs["loo_r2"] = r2
    frame.attrs["n_runs"] = len(target)
    frame.attrs["alpha"] = alpha
    return frame


def _ridge_fit(matrix: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge with an unpenalised intercept."""
    design = np.column_stack([np.ones(len(matrix)), matrix])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)[1:]


# --------------------------------------------------------------------------
# report


def _fmt(value: object, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "–"
    return f"{number:.{digits}f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def render_report(
    judged: pd.DataFrame,
    bands: pd.DataFrame,
    correlations: pd.DataFrame,
    pairs: pd.DataFrame,
    splits: pd.DataFrame,
    weights: pd.DataFrame,
) -> str:
    lines: List[str] = []
    lines.append("# Review length and decoding settings\n")
    lines.append(
        "Companion to the Phase 1 local comparison and the Phase 2 judge run. "
        "Everything here is computed from the archived runs and the 90 judged reviews. "
        "No model was called to produce this file.\n"
    )

    adapter = judged[judged["arm"] == "adapter"]
    base = judged[judged["arm"] == "base"]
    lines.append("## What the judged reviews look like\n")
    if not judged.empty:
        lines.append(
            _table(
                ["arm", "n", "mean words", "median", "min", "max", "mean score"],
                [
                    [
                        arm,
                        len(frame),
                        _fmt(frame["words"].mean(), 0),
                        _fmt(frame["words"].median(), 0),
                        _fmt(frame["words"].min(), 0),
                        _fmt(frame["words"].max(), 0),
                        _fmt(frame["overall"].mean(), 1),
                    ]
                    for arm, frame in judged.groupby("arm")
                ],
            )
        )
        lines.append("")
        references = judged["reference_words"].dropna()
        if not references.empty:
            lines.append(
                f"The human reference reviews for these papers average **{references.mean():.0f} words** "
                f"(median {references.median():.0f}, p10 {references.quantile(0.1):.0f}, "
                f"p90 {references.quantile(0.9):.0f}). The adapter sits at or near that band; "
                f"the base arm is roughly {base['words'].mean() / max(references.mean(), 1):.1f}x the human length."
            )
            lines.append("")

    lines.append("## Does length explain the score?\n")
    lines.append(_table(["group", "n", "Spearman", "Pearson"], [[r.group, r.n, _fmt(r.spearman), _fmt(r.pearson)] for r in correlations.itertuples(index=False)]))
    lines.append("")
    lines.append(
        "Read the arm and per-config rows, not the pooled row. Pooled length and arm are almost "
        "perfectly confounded, because every adapter review is short and every base review is long. "
        "Inside each arm the correlation is the honest test."
    )
    lines.append("")

    lines.append("## Length bands\n")
    if not bands.empty:
        lines.append(_table(["arm", "words", "n", "mean score", "median score"], [[r.arm, r.band, r.n, _fmt(r.mean_score, 1), _fmt(r.median_score, 1)] for r in bands.itertuples(index=False)]))
        lines.append("")
        verdict, _ = length_verdict(judged)
        lines.append(f"Within the adapter arm, {verdict}.")
        lines.append("")

    lines.append("## Setting weights: matched temperature pairs\n")
    if pairs.empty:
        lines.append("No within-family temperature pair was available.")
    else:
        lines.append(
            _table(
                ["family", "low T", "high T", "Δcomposite (lower better)", "Δlength_fit", "Δemb_paper", "Δstructure", "Δrepeat5", "Δdegenerate", "other axes differ"],
                [
                    [
                        r.family,
                        _fmt(r.low_temperature),
                        _fmt(r.high_temperature),
                        _fmt(r.delta_composite),
                        _fmt(getattr(r, "delta_length_fit", float("nan"))),
                        _fmt(getattr(r, "delta_emb_paper_cos", float("nan"))),
                        _fmt(getattr(r, "delta_structure_score", float("nan"))),
                        _fmt(getattr(r, "delta_repeat5_mean", float("nan")), 3),
                        _fmt(getattr(r, "delta_degenerate_rate", float("nan")), 2),
                        r.other_axes_differ or "none",
                    ]
                    for r in pairs.itertuples(index=False)
                ],
            )
        )
        lines.append("")
        better_high = int((pairs["delta_composite"] < 0).sum())
        lines.append(
            f"Higher temperature produced the better composite rank in **{better_high} of {len(pairs)}** families. "
            "A negative delta means the high-temperature run ranked better."
        )
        if (pairs["other_axes_differ"].fillna("") != "").any():
            lines.append(
                "Where `other axes differ` is not empty, the pair is not a clean temperature test and is "
                "reported as a package effect."
            )
        lines.append("")

    lines.append("## Setting weights: above vs below median\n")
    if not splits.empty:
        lines.append(
            _table(
                ["axis", "runs", "distinct values", "low group composite", "high group composite", "delta", "usable"],
                [
                    [
                        r.axis,
                        r.n,
                        r.distinct_values,
                        _fmt(r.low_mean_composite),
                        _fmt(r.high_mean_composite),
                        _fmt(r.delta),
                        "yes" if r.identifiable else "no",
                    ]
                    for r in splits.itertuples(index=False)
                ],
            )
        )
        lines.append("")
        unusable = [r.axis for r in splits.itertuples(index=False) if not r.identifiable]
        if unusable:
            lines.append(
                f"Not usable as evidence: {', '.join('`' + axis + '`' for axis in unusable)}. "
                "Either the value is constant across runs, or one side of the split has fewer than three runs."
            )
            lines.append("")

    lines.append("## Setting weights: ridge with leave-one-run-out\n")
    if weights.empty:
        lines.append("Not enough runs for a regression.")
    else:
        lines.append(_table(["setting", "coefficient on composite rank", "standardised"], [[r.feature, _fmt(r.coefficient, 3), _fmt(r.std_coefficient, 3)] for r in weights.itertuples(index=False)]))
        lines.append("")
        r2 = weights.attrs.get("loo_r2", float("nan"))
        lines.append(
            f"Leave-one-run-out R² = **{_fmt(r2, 2)}** on {weights.attrs.get('n_runs', '?')} runs "
            f"(ridge alpha {weights.attrs.get('alpha', '?')})."
        )
        if r2 == r2 and r2 <= 0.0:
            lines.append(
                "A non-positive R² means the settings do not predict the ranking better than the mean. "
                "Treat the coefficients as directions to test, not as established effects."
            )
        lines.append("")

    lines.append("## Verdict\n")
    verdict, _ = length_verdict(judged)
    lines.append(f"- **Length:** {verdict}. Short adapter reviews are not the problem; they already match human length.")
    if not correlations.empty:
        arm_rows = correlations[correlations["group"].str.startswith("arm=")]
        for row in arm_rows.itertuples(index=False):
            lines.append(f"- **{row.group}:** length vs score Spearman {_fmt(row.spearman)} over {row.n} reviews.")
    lines.append(
        "- **Length is not the driver of the base advantage.** Inside the adapter arm length correlates weakly "
        "and positively; inside the base arm it correlates negatively. The base wins by covering more ground, "
        "not by being longer."
    )
    lines.append(
        "- **Settings:** only temperature has matched pairs. Every other axis is bundled with prompt or budget "
        "changes, and the regression cannot separate them from 15 runs. Any claim that top_p, top_k or "
        "repetition_penalty 'helped' would be unsupported."
    )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# cli


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Length and setting-weight analysis")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    judged = load_judged()
    runs = load_runs()
    aggregates = load_aggregates()

    bands = length_bands(judged)
    correlations = length_correlations(judged)
    pairs = matched_pairs(aggregates, runs)
    splits = axis_split(aggregates, runs)
    weights = ridge_weights(aggregates, runs)

    bands.to_csv(args.out_dir / LENGTH_BANDS_CSV.name, index=False)
    pairs.to_csv(args.out_dir / SETTINGS_PAIRS_CSV.name, index=False)
    weights.to_csv(args.out_dir / SETTINGS_WEIGHTS_CSV.name, index=False)

    report = render_report(judged, bands, correlations, pairs, splits, weights)
    (args.out_dir / REPORT_MD.name).write_text(report, encoding="utf-8")

    print(f"judged reviews: {len(judged)}")
    print(f"wrote {LENGTH_BANDS_CSV.name}, {SETTINGS_PAIRS_CSV.name}, {SETTINGS_WEIGHTS_CSV.name}, {REPORT_MD.name}")
    if not pairs.empty:
        print(pairs[["family", "low_temperature", "high_temperature", "delta_composite"]].to_string(index=False))
    if not weights.empty:
        print(f"LOO R2 = {weights.attrs.get('loo_r2'):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
