"""Load and validate the archived Preprint Critic review experiments.

Phase 1 of the comparison reads only the 15 archived ``*.zip`` runs that live in
``preprint_critic_review_experiments/``.  Those zips are external, untracked
data: nothing here copies them into the repository, and nothing here generates,
trains, or calls a language model.

The canonical machines-readable contract is ``experiments/COMPARISON_PROTOCOL.md``.
"""

from __future__ import annotations

import csv
import io
import json
import os
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

DEFAULT_EXPERIMENTS_DIR = r"D:\Preprint Critic\preprint_critic_review_experiments"
EXPERIMENTS_DIR = Path(
    os.environ.get("PREPRINT_EXPERIMENTS_DIR", DEFAULT_EXPERIMENTS_DIR)
)

REQUIRED_FILES: Tuple[str, ...] = (
    "reviews.csv",
    "summary_by_arm.csv",
    "reviews.jsonl",
    "run_config.json",
)

REQUIRED_REVIEW_COLUMNS: Tuple[str, ...] = (
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
)

EXPECTED_RUNS: Tuple[str, ...] = (
    "a_bit_altered_prompt_60_papers",
    "a_bit_altered_prompt_conservative_settings",
    "gpt_prompt_high_temp",
    "gpt_prompt_low_temp",
    "new_prompt_with_conservative_settings",
    "new_prompt_with_constrains_60_papers",
    "original_prompt_high_temp",
    "original_prompt_low_temp",
    "original_prompt_original_settings",
    "relaxed_references_prompt_high_temp",
    "relaxed_references_prompt_low_temp",
    "tweaked_prompt_high_temp",
    "tweaked_prompt_mid_temp",
    "very_big_prompt_high_temp",
    "very_big_prompt_very_high_temp",
)

#: Run-name prefix -> prompt family.  Longest prefixes are checked first so that
#: ``new_prompt_*`` is not mistaken for anything else.
_PROMPT_PREFIXES: Tuple[Tuple[str, str], ...] = (
    ("a_bit_altered_prompt", "a_bit"),
    ("relaxed_references_prompt", "relaxed"),
    ("very_big_prompt", "very_big"),
    ("tweaked_prompt", "tweaked"),
    ("original_prompt", "original"),
    ("new_prompt", "new"),
    ("gpt_prompt", "gpt"),
)

#: The ten papers every run shares, in the order they appear in the runs.
COMMON_10: Tuple[str, ...] = (
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

#: The two 100-paper runs were executed with the adapter arm only.
_ADAPTER_ONLY_RUNS = frozenset(
    {
        "a_bit_altered_prompt_conservative_settings",
        "new_prompt_with_conservative_settings",
    }
)

_FLOAT_TOLERANCE = 1e-3


def prompt_family(run: str) -> str:
    """Return the prompt family implied by a run name."""
    for prefix, family in _PROMPT_PREFIXES:
        if run.startswith(prefix):
            return family
    raise ValueError(f"cannot derive a prompt family from run name {run!r}")


def expected_arms(run: str) -> Tuple[str, ...]:
    """Return the arms a run must contain, sorted for deterministic output."""
    if run in _ADAPTER_ONLY_RUNS:
        return ("adapter",)
    return ("adapter", "base")


def _normalise_text(text: str) -> str:
    return " ".join(str(text).split())


def _parse_bool(value: object) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"cannot parse boolean value {value!r}")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


@dataclass
class _RunData:
    """Everything read from one run archive, already row-validated."""

    run: str
    config: dict
    rows: List[dict]
    summary: Dict[str, dict] = field(default_factory=dict)
    paper_ids: List[str] = field(default_factory=list)
    arms: Tuple[str, ...] = ()

    @property
    def n_papers(self) -> int:
        return int(self.config.get("n_papers", len(self.paper_ids)))


def _read_csv_members(archive: zipfile.ZipFile, name: str) -> List[dict]:
    with archive.open(name) as handle:
        text = io.TextIOWrapper(handle, encoding="utf-8-sig", newline="")
        return list(csv.DictReader(text))


def _collect_run(root: Path, run: str, errors: List[str]) -> "_RunData | None":
    """Read and row-validate one run.  Returns ``None`` when it cannot be read."""
    path = root / f"{run}.zip"
    if not path.is_file():
        errors.append(f"{run}: missing archive {path}")
        return None

    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:  # pragma: no cover - defensive
        errors.append(f"{run}: not a readable zip ({exc})")
        return None

    with archive:
        names = set(archive.namelist())
        missing = [name for name in REQUIRED_FILES if name not in names]
        if missing:
            errors.append(f"{run}: archive is missing required files {missing}")
            return None

        try:
            config = json.loads(archive.read("run_config.json").decode("utf-8-sig"))
            rows = _read_csv_members(archive, "reviews.csv")
            summary_rows = _read_csv_members(archive, "summary_by_arm.csv")
        except (ValueError, KeyError, UnicodeDecodeError) as exc:
            errors.append(f"{run}: cannot parse archive members ({exc})")
            return None

    if not rows:
        errors.append(f"{run}: reviews.csv has no rows")
        return None

    missing_columns = [
        column for column in REQUIRED_REVIEW_COLUMNS if column not in rows[0]
    ]
    if missing_columns:
        errors.append(f"{run}: reviews.csv is missing columns {missing_columns}")
        return None

    run_errors: List[str] = []

    paper_ids = [str(value) for value in config.get("paper_ids", [])]
    if not paper_ids:
        run_errors.append(f"{run}: run_config.json has no paper_ids")
    n_papers = int(config.get("n_papers", len(paper_ids)))
    if n_papers != len(paper_ids):
        run_errors.append(
            f"{run}: n_papers={n_papers} but paper_ids has {len(paper_ids)} entries"
        )

    row_paper_order: List[str] = []
    seen_pairs = Counter()
    empty_reviews = 0
    for row in rows:
        paper_id = str(row["paper_id"])
        if paper_id not in row_paper_order:
            row_paper_order.append(paper_id)
        seen_pairs[(paper_id, str(row["arm"]))] += 1
        if not str(row["review"]).strip() or not str(row["reference_review"]).strip():
            empty_reviews += 1

    if empty_reviews:
        run_errors.append(f"{run}: {empty_reviews} rows have an empty review or reference")

    duplicate_pairs = [pair for pair, count in seen_pairs.items() if count > 1]
    if duplicate_pairs:
        run_errors.append(
            f"{run}: duplicate (paper_id, arm) rows: {sorted(duplicate_pairs)}"
        )

    if row_paper_order != paper_ids:
        run_errors.append(
            f"{run}: row paper order does not match run_config.paper_ids"
        )

    row_arms = {str(row["arm"]) for row in rows}
    unexpected_arms = sorted(row_arms - {"base", "adapter"})
    if unexpected_arms:
        run_errors.append(f"{run}: unexpected arms {unexpected_arms}")
    if row_arms != set(expected_arms(run)):
        run_errors.append(
            f"{run}: arms {sorted(row_arms)} != expected {list(expected_arms(run))}"
        )

    expected_rows = n_papers * len(row_arms)
    if len(rows) != expected_rows:
        run_errors.append(
            f"{run}: row count {len(rows)} != n_papers*arms {expected_rows}"
        )

    summary: Dict[str, dict] = {}
    for row in summary_rows:
        summary[str(row.get("arm", ""))] = row

    for arm in sorted(row_arms):
        arm_rows = [row for row in rows if str(row["arm"]) == arm]
        summary_row = summary.get(arm)
        if summary_row is None:
            run_errors.append(f"{run}: summary_by_arm.csv has no row for arm {arm!r}")
            continue
        try:
            reported_n = int(float(summary_row["n"]))
        except (KeyError, TypeError, ValueError):
            run_errors.append(f"{run}: summary column 'n' for {arm} is missing or invalid")
            reported_n = None
        if reported_n is not None and reported_n != len(arm_rows):
            run_errors.append(
                f"{run}: summary n for {arm} is {summary_row['n']}, expected {len(arm_rows)}"
            )
        try:
            recomputed = {
                "mean_words": _mean([float(row["words"]) for row in arm_rows]),
                "mean_output_tokens": _mean(
                    [float(row["output_tokens"]) for row in arm_rows]
                ),
                "mean_rouge_l": _mean([float(row["rouge_l"]) for row in arm_rows]),
                "mean_repeat5": _mean([float(row["repeat5"]) for row in arm_rows]),
                "cap_reached": float(
                    sum(1 for row in arm_rows if _parse_bool(row["hit_cap"]))
                ),
            }
        except (KeyError, TypeError, ValueError) as error:
            run_errors.append(
                f"{run}: cannot recompute summary for {arm}: {type(error).__name__}: {error}"
            )
            continue
        for key, value in recomputed.items():
            try:
                reported = float(summary_row[key])
            except (KeyError, TypeError, ValueError):
                run_errors.append(f"{run}: summary column {key!r} is missing or invalid")
                continue
            if abs(value - reported) > _FLOAT_TOLERANCE:
                run_errors.append(
                    f"{run}: summary {key} for {arm} is {reported}, recomputed {value:.6f}"
                )

    errors.extend(run_errors)
    if run_errors:
        return None

    return _RunData(
        run=run,
        config=config,
        rows=rows,
        summary=summary,
        paper_ids=paper_ids,
        arms=tuple(sorted(row_arms)),
    )


def _canonical_paper_order(collected: Dict[str, _RunData], errors: List[str]) -> List[str]:
    """Return the longest run's paper order and check every run is its prefix."""
    longest = max(len(data.paper_ids) for data in collected.values())
    candidates = [data for data in collected.values() if len(data.paper_ids) == longest]
    canonical = list(candidates[0].paper_ids)
    for other in candidates[1:]:
        if other.paper_ids != canonical:
            errors.append(
                f"longest runs disagree on paper order: "
                f"{candidates[0].run} vs {other.run}"
            )
    return canonical


def validate_all(
    root: Path | str = EXPERIMENTS_DIR,
    expected_runs: Sequence[str] = EXPECTED_RUNS,
) -> List[str]:
    """Return every validation problem with the runs under ``root``.

    An empty list means the archives are structurally sound and internally
    consistent.  ``expected_runs`` is explicit so tests can validate a small
    synthetic fixture without touching the real experiment directory.
    """
    root = Path(root)
    errors: List[str] = []
    collected: Dict[str, _RunData] = {}

    if not root.is_dir():
        return [f"experiments directory does not exist: {root}"]

    for run in expected_runs:
        data = _collect_run(root, run, errors)
        if data is not None:
            collected[run] = data

    if not collected:
        return errors

    canonical = _canonical_paper_order(collected, errors)
    for run, data in collected.items():
        if data.paper_ids != canonical[: len(data.paper_ids)]:
            errors.append(f"{run}: paper order is not a prefix of the canonical order")

    references: Dict[str, Tuple[str, str]] = {}
    for run, data in collected.items():
        for row in data.rows:
            paper_id = str(row["paper_id"])
            reference = _normalise_text(row["reference_review"])
            previous = references.get(paper_id)
            if previous is None:
                references[paper_id] = (run, reference)
            elif previous[1] != reference:
                errors.append(
                    f"paper {paper_id}: reference review differs between "
                    f"{previous[0]} and {run}"
                )

    return errors


def _float_column(rows: Iterable[dict], key: str) -> List[float]:
    return [float(row[key]) for row in rows]


def _int_column(rows: Iterable[dict], key: str) -> List[int]:
    return [int(float(row[key])) for row in rows]


def _review_frame(data: _RunData) -> pd.DataFrame:
    rows = data.rows
    order = {paper_id: index for index, paper_id in enumerate(data.paper_ids)}
    return pd.DataFrame(
        {
            "run": data.run,
            "paper_id": [str(row["paper_id"]) for row in rows],
            "paper_order": [order[str(row["paper_id"])] for row in rows],
            "arm": [str(row["arm"]) for row in rows],
            "review": [str(row["review"]) for row in rows],
            "reference_review": [str(row["reference_review"]) for row in rows],
            "prompt_tokens": _int_column(rows, "prompt_tokens"),
            "output_tokens": _int_column(rows, "output_tokens"),
            "seconds": _float_column(rows, "seconds"),
            "words": _int_column(rows, "words"),
            "rouge_l": _float_column(rows, "rouge_l"),
            "repeat5": _float_column(rows, "repeat5"),
            "hit_cap": [_parse_bool(row["hit_cap"]) for row in rows],
            "seed": _int_column(rows, "seed"),
            "max_new_tokens": _int_column(rows, "max_new_tokens"),
            "temperature": _float_column(rows, "temperature"),
            "top_p": _float_column(rows, "top_p"),
            "top_k": _int_column(rows, "top_k"),
            "repetition_penalty": _float_column(rows, "repetition_penalty"),
            "no_repeat_ngram_size": _int_column(rows, "no_repeat_ngram_size"),
        }
    )


def _run_frame(data: _RunData) -> dict:
    config = data.config
    return {
        "run": data.run,
        "prompt_family": prompt_family(data.run),
        "n_papers": data.n_papers,
        "arms": ",".join(data.arms),
        "paper_ids": ",".join(data.paper_ids),
        "max_paper_tokens": int(config.get("max_paper_tokens", 0)),
        "max_new_tokens": int(config.get("max_new_tokens", 0)),
        "temperature": float(config.get("temperature", float("nan"))),
        "top_p": float(config.get("top_p", float("nan"))),
        "top_k": int(config.get("top_k", 0)),
        "repetition_penalty": float(config.get("repetition_penalty", float("nan"))),
        "no_repeat_ngram_size": int(config.get("no_repeat_ngram_size", 0)),
        "do_sample": bool(config.get("do_sample", False)),
        "base_model": str(config.get("base_model", "")),
        "adapter_dir": str(config.get("adapter_dir", "")),
    }


def load_all(
    root: Path | str = EXPERIMENTS_DIR,
    expected_runs: Sequence[str] = EXPECTED_RUNS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load every run, raising ``ValueError`` if validation fails.

    Returns ``(runs_df, reviews_df)``.  ``runs_df`` has one row per run;
    ``reviews_df`` has one row per ``(run, paper_id, arm)``.
    """
    errors = validate_all(root, expected_runs)
    if errors:
        raise ValueError(
            "experiment archives failed validation:\n" + "\n".join(errors)
        )

    root = Path(root)
    collected = [_collect_run(root, run, []) for run in expected_runs]
    present = [data for data in collected if data is not None]
    runs_df = pd.DataFrame([_run_frame(data) for data in present])
    reviews_df = pd.concat(
        [_review_frame(data) for data in present], ignore_index=True
    )
    return runs_df, reviews_df
