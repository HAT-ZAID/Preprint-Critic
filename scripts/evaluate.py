#!/usr/bin/env python
"""Offline evaluation of Preprint Critic V3a review outputs.

Computes per-arm surface statistics, a self-contained ROUGE-L, template-key
coverage, citation grounding and 5-gram repetition for the ``base``, ``fine``
and ``reference`` review arms, then writes a deterministic JSON report.

Usage (from the repository root)::

    python scripts/evaluate.py
    python scripts/evaluate.py --comparison path --papers path \
        --tokenizer path --out path

Only the standard library, ``tokenizers`` and the files under ``comparison/``
are required.  No network access is performed.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_COMPARISON = REPO_ROOT / "comparison" / "comparison_v3a.json"
DEFAULT_PAPERS = REPO_ROOT / "comparison" / "extracted_paper_texts.json"
DEFAULT_TOKENIZER = REPO_ROOT / "rc_project_adapter_V3" / "tokenizer.json"
DEFAULT_OUT = REPO_ROOT / "comparison" / "metrics_v3a.json"

ARMS = ("base", "fine", "reference")

TEMPLATE_KEYS = ("summary_of_the_paper", "strengths", "weaknesses", "questions")

# Optional leading Markdown decoration: blockquote markers, headings, bullets,
# emphasis and indentation.  ``\s`` is deliberately avoided so that a key can
# never be matched by crossing a line break.
_LINE_LEAD = r"[ \t\u00a0>#*\-]*"

_TEMPLATE_PATTERNS = {
    key: re.compile(
        rf"^{_LINE_LEAD}{re.escape(key)}(?![A-Za-z0-9_])[ \t]*(?:\*\*)?[ \t]*:?",
        re.IGNORECASE | re.MULTILINE,
    )
    for key in TEMPLATE_KEYS
}

_CITATION_RE = re.compile(
    r"\b(?:Section|Sec\.?|Table|Tab\.?|Figure|Fig\.?)\s*([0-9]+[A-Za-z]?)",
    re.IGNORECASE,
)

# Surface keyword -> canonical citation type.  Normalising "Fig. 3" to
# "Figure 3" keeps the emitted strings stable and lets grounding survive
# abbreviation differences between the review and the paper.
_CITATION_TYPES = {
    "section": "Section",
    "sec": "Section",
    "table": "Table",
    "tab": "Table",
    "figure": "Figure",
    "fig": "Figure",
}

_WORD_RE = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def word_tokens(text: str) -> List[str]:
    """Lowercased alphanumeric tokens used by the ROUGE-L implementation."""
    return _WORD_RE.findall(text.lower())


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence of two token sequences."""
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token in a:
        current = [0]
        for index, other in enumerate(b, 1):
            if token == other:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[index - 1]))
        previous = current
    return previous[-1]


def rouge_l(candidate: str, reference: str, beta: float = 1.2) -> float:
    """ROUGE-L F-measure (beta=1.2) over lowercased ``[a-z0-9]+`` tokens.

    Returns ``0.0`` when either side has no tokens, when the LCS is empty, or
    when the reference is empty.  Implemented directly from the LCS so that no
    ``rouge-score`` dependency is needed.
    """
    candidate_tokens = word_tokens(candidate)
    reference_tokens = word_tokens(reference)
    if not candidate_tokens or not reference_tokens:
        return 0.0

    lcs = lcs_length(candidate_tokens, reference_tokens)
    if lcs == 0:
        return 0.0

    precision = lcs / len(candidate_tokens)
    recall = lcs / len(reference_tokens)
    beta_sq = beta * beta
    return (1.0 + beta_sq) * precision * recall / (recall + beta_sq * precision)


def ngram_repeat_rate(text: str, n: int = 5) -> float:
    """``1 - distinct_n_grams / total_n_grams`` over lowercased words."""
    words = text.lower().split()
    if len(words) < n:
        return 0.0
    total = len(words) - n + 1
    distinct = len({tuple(words[i : i + n]) for i in range(total)})
    return 1.0 - distinct / total


def template_keys_present(text: str) -> List[str]:
    """Template keys that appear at the start of a line, in canonical order.

    A key may be preceded by ``#``, ``*``, ``-``, ``>`` or spaces, and may be
    followed by an optional ``:`` or ``**`` (for example
    ``**summary_of_the_paper:**`` or ``## Strengths``).
    """
    return [key for key in TEMPLATE_KEYS if _TEMPLATE_PATTERNS[key].search(text)]


def has_all_four_template_keys(text: str) -> bool:
    """True when all four template keys are present at line starts."""
    return len(template_keys_present(text)) == len(TEMPLATE_KEYS)


def citations(text: str) -> List[str]:
    """All citation mentions, normalised (``"Table 2"``, ``"Figure 3"``)."""
    found: List[str] = []
    for match in _CITATION_RE.finditer(text):
        prefix = match.group(0)[: match.start(1) - match.start(0)]
        keyword = prefix.strip().rstrip(".").strip().lower()
        canonical = _CITATION_TYPES.get(keyword, keyword.title())
        found.append(f"{canonical} {match.group(1)}")
    return found


def grounded_citations(found: Iterable[str], paper_text: str) -> List[str]:
    """Citations whose normalised lowercase form occurs in the paper text."""
    haystack = paper_text.lower()
    return [citation for citation in found if citation.lower() in haystack]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _round(value: float) -> float:
    return round(float(value), 6)


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else _round(numerator / denominator)


def _display_path(path: Path) -> str:
    """Stable, repo-relative path when possible so the report is portable."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _count_real_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def evaluate_sample(row: Dict, paper_text: str, tokenizer) -> Dict:
    reference_review = row["reference_review"]
    record: Dict[str, Dict] = {}

    for arm in ARMS:
        text = row[f"{arm}_review"]
        found = citations(text)
        grounded = grounded_citations(found, paper_text)
        keys = template_keys_present(text)
        record[arm] = {
            "words": len(text.split()),
            "chars": len(text),
            "real_output_tokens": _count_real_tokens(tokenizer, text),
            "recorded_output_tokens": row.get(f"{arm}_output_tokens"),
            "hit_cap": row.get(f"{arm}_hit_cap"),
            "rouge_l": _round(rouge_l(text, reference_review)),
            "template_keys_present": keys,
            "has_all_four": len(keys) == len(TEMPLATE_KEYS),
            "citations": found,
            "citation_count": len(found),
            "grounded_citations": grounded,
            "grounded_citation_count": len(grounded),
            "citation_grounding_rate": _rate(len(grounded), len(found)),
            "ngram_repeat_rate": _round(ngram_repeat_rate(text)),
        }

    return {"paper_id": row["paper_id"], **record}


def aggregate_arm(samples: Sequence[Dict], arm: str) -> Dict:
    arm_records = [sample[arm] for sample in samples]
    words = [record["words"] for record in arm_records]
    rouge_scores = [record["rouge_l"] for record in arm_records]
    repeat_rates = [record["ngram_repeat_rate"] for record in arm_records]
    real_tokens = [record["real_output_tokens"] for record in arm_records]
    recorded = [record["recorded_output_tokens"] for record in arm_records]
    citation_total = sum(record["citation_count"] for record in arm_records)
    grounded_total = sum(record["grounded_citation_count"] for record in arm_records)

    return {
        "n": len(arm_records),
        "mean_words": _round(statistics.mean(words)) if words else 0.0,
        "median_words": _round(statistics.median(words)) if words else 0.0,
        "mean_rouge_l": _round(statistics.mean(rouge_scores)) if rouge_scores else 0.0,
        "min_rouge_l": _round(min(rouge_scores)) if rouge_scores else 0.0,
        "max_rouge_l": _round(max(rouge_scores)) if rouge_scores else 0.0,
        "template_count": sum(1 for record in arm_records if record["template_keys_present"]),
        "has_all_four_count": sum(1 for record in arm_records if record["has_all_four"]),
        "summary_of_the_paper_count": sum(
            1 for record in arm_records if "summary_of_the_paper" in record["template_keys_present"]
        ),
        "citation_total": citation_total,
        "grounded_citation_total": grounded_total,
        "citation_grounding_rate": _rate(grounded_total, citation_total),
        "mean_ngram_repeat_rate": _round(statistics.mean(repeat_rates)) if repeat_rates else 0.0,
        "mean_real_output_tokens": _round(statistics.mean(real_tokens)) if real_tokens else 0.0,
        "real_output_tokens": real_tokens,
        "recorded_output_tokens": recorded if any(v is not None for v in recorded) else None,
        "hit_cap_count": (
            sum(1 for record in arm_records if record["hit_cap"])
            if arm in ("base", "fine")
            else None
        ),
    }


def head_to_head(samples: Sequence[Dict]) -> Dict[str, int]:
    fine_wins = base_wins = ties = 0
    for sample in samples:
        fine_score = sample["fine"]["rouge_l"]
        base_score = sample["base"]["rouge_l"]
        if fine_score > base_score:
            fine_wins += 1
        elif base_score > fine_score:
            base_wins += 1
        else:
            ties += 1
    return {"fine_wins": fine_wins, "base_wins": base_wins, "ties": ties}


DEFINITIONS = {
    "rouge_l": (
        "ROUGE-L F-measure with beta=1.2 from a token-level LCS. Tokens are "
        "lowercased matches of [a-z0-9]+; P=LCS/len(candidate), "
        "R=LCS/len(reference), F=(1+beta^2)*P*R/(R+beta^2*P). 0.0 when the "
        "reference (or candidate) is empty or the LCS is 0."
    ),
    "template_keys": (
        "Subset of [summary_of_the_paper, strengths, weaknesses, questions] "
        "found at the start of a line, case-insensitive, allowing leading "
        "'#', '*', '-', '>' or spaces and an optional ':' or '**'. "
        "has_all_four is true when all four keys are present. template_count "
        "counts reviews with at least one key; has_all_four_count counts "
        "reviews with all four."
    ),
    "citation_grounding": (
        "All matches of \\b(?:Section|Sec\\.?|Table|Tab\\.?|Figure|Fig\\.?)"
        "\\s*([0-9]+[A-Za-z]?), normalised to 'Section N', 'Table N' or "
        "'Figure N'. A citation is grounded when its lowercased normalised "
        "form is a substring of the lowercased paper text. Grounding rate is "
        "grounded/total, 0.0 when there are no citations."
    ),
    "ngram_repeat_rate": (
        "n=5 over lowercased whitespace words: "
        "1 - distinct_5grams/total_5grams, 0.0 when fewer than 5 words."
    ),
}

NOTES = (
    "Recorded output token counts (base_output_tokens / fine_output_tokens) "
    "come from the generation run and use padded batch width, i.e. every "
    "sample in a batch shares the longest sequence's length, so they are "
    "batch-level constants and NOT per-sample output lengths. real_output_tokens "
    "(tokenizer.encode(text, add_special_tokens=False)) is the per-sample "
    "measure. The reference arm has no recorded generation count. "
    "The cross-check figure of 0/10, 6/10, 5/10 for the template metric equals "
    "reviews whose first line declares summary_of_the_paper, reported here as "
    "summary_of_the_paper_count; the stricter four-key metric is "
    "has_all_four_count."
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def load_tokenizer(path: Path):
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "The 'tokenizers' package is required to count real output tokens."
        ) from exc
    return Tokenizer.from_file(str(path))


def build_report(comparison_path: Path, papers_path: Path, tokenizer_path: Path) -> Dict:
    rows = json.loads(comparison_path.read_text(encoding="utf-8"))
    papers = json.loads(papers_path.read_text(encoding="utf-8"))
    tokenizer = load_tokenizer(tokenizer_path)

    missing = [row["paper_id"] for row in rows if row["paper_id"] not in papers]
    if missing:
        raise SystemExit(f"Paper text missing for: {', '.join(missing)}")

    samples = [
        evaluate_sample(row, papers[row["paper_id"]], tokenizer) for row in rows
    ]

    return {
        "script": "scripts/evaluate.py",
        "inputs": {
            "comparison": _display_path(comparison_path),
            "papers": _display_path(papers_path),
            "tokenizer": _display_path(tokenizer_path),
        },
        "n_samples": len(samples),
        "definitions": DEFINITIONS,
        "arms": {arm: aggregate_arm(samples, arm) for arm in ARMS},
        "head_to_head": head_to_head(samples),
        "per_sample": samples,
        "notes": NOTES,
    }


def print_table(report: Dict) -> None:
    header = (
        f"{'arm':<10}{'mean_words':>12}{'mean_rouge_l':>15}"
        f"{'tmpl(any)':>12}{'4-key':>8}{'grounded_cit':>15}{'mean_repeat':>14}"
    )
    print(header)
    print("-" * len(header))
    total = report["n_samples"]
    for arm in ARMS:
        values = report["arms"][arm]
        grounded = (
            f"{values['grounded_citation_total']}/{values['citation_total']}"
            if values["citation_total"]
            else "0/0"
        )
        print(
            f"{arm:<10}{values['mean_words']:>12.1f}{values['mean_rouge_l']:>15.4f}"
            f"{values['template_count']:>8}/{total:<3}"
            f"{values['has_all_four_count']:>4}/{total:<3}"
            f"{grounded:>15}{values['mean_ngram_repeat_rate']:>14.4f}"
        )
    print()
    h2h = report["head_to_head"]
    print(
        "head-to-head (ROUGE-L, fine vs base): "
        f"fine_wins={h2h['fine_wins']} base_wins={h2h['base_wins']} ties={h2h['ties']}"
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON,
                        help="comparison JSON with the 10 review rows")
    parser.add_argument("--papers", type=Path, default=DEFAULT_PAPERS,
                        help="paper_id -> full paper text JSON")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER,
                        help="tokenizer.json used for the generation run")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="destination for the metrics JSON")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = build_report(args.comparison, args.papers, args.tokenizer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print_table(report)
    print(f"wrote {_display_path(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
