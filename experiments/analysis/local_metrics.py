"""Local, non-generative metrics for the Preprint Critic review comparison.

Everything here runs offline: lexical statistics, structural checks, and
embeddings from a local Ollama model.  No review is generated, no judge is
called, and no OpenCode session or quota is used.

The metric definitions are fixed in ``experiments/COMPARISON_PROTOCOL.md``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # requests is available in this environment, but keep import errors clear
    import requests
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("the 'requests' package is required for local embeddings") from exc

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENTS_DIR = Path(
    os.environ.get(
        "PREPRINT_EXPERIMENTS_DIR", r"D:\Preprint Critic\preprint_critic_review_experiments"
    )
)
PAPERS_FILENAME = "test_dataset_untokenized.jsonl"
CACHE_PATH = REPO_ROOT / "experiments" / "cache" / "embeddings.sqlite"

EMBED_MODEL = os.environ.get("PREPRINT_EMBED_MODEL", "nomic-embed-text:latest")
EMBED_HOST = os.environ.get("PREPRINT_EMBED_HOST", "http://127.0.0.1:11435")
OLLAMA_EXECUTABLE = os.environ.get("OLLAMA_EXE", r"D:\Ollama\ollama.exe")
OLLAMA_MODELS_DIR = os.environ.get("OLLAMA_MODELS", r"D:\ollama")

#: Character chunking for paper grounding.  2,000 characters with a 200
#: character overlap covers roughly the first 48,000 characters of a paper,
#: which is close to the 17,500-token paper budget the runs themselves used.
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
MAX_CHUNKS = 24

META_PHRASES: Tuple[str, ...] = (
    "post-editing notes",
    "post editing notes",
    "i haven't read",
    "i have not read",
    "as an ai",
    "initial summary",
    "i cannot read",
    "the provided text",
    "based on the excerpt",
)

_SUMMARY_RE = re.compile(r"\bsummar(?:y|ies|ise|ize|ising|izing)\b")
_STRENGTH_RE = re.compile(r"\bstrengths?\b")
_WEAKNESS_RE = re.compile(r"\bweakness(?:es)?\b|\bweak points?\b")
_VERDICT_RE = re.compile(
    r"\bverdict\b|\brecommend(?:ation|ations|ed|s)?\b|\bdecision\b"
)
_SECTION_RE = re.compile(
    r"(?:\b(?:section|sec\.)|§)\s*(\d+(?:\.\d+)*)", re.IGNORECASE
)
_TABLE_RE = re.compile(r"\btable\s*(\d+)", re.IGNORECASE)
_FIGURE_RE = re.compile(r"\b(?:figure|fig\.)\s*(\d+)", re.IGNORECASE)
_EQUATION_RE = re.compile(r"\b(?:equation|eq\.)\s*\(?(\d+)\)?", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Lexical helpers
# ---------------------------------------------------------------------------


def normalise(text: str) -> str:
    """Collapse all whitespace runs to single spaces."""
    return " ".join(str(text).split())


def count_words(text: str) -> int:
    """Count whitespace-delimited words after normalisation."""
    cleaned = normalise(text)
    return len(cleaned.split()) if cleaned else 0


def _ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def repeat_stats(text: str, n: int = 5) -> float:
    """Fraction of word n-gram positions that belong to a repeated n-gram."""
    tokens = normalise(text).split()
    grams = _ngrams(tokens, n)
    if not grams:
        return 0.0
    repeated = len(grams) - len(set(grams))
    return repeated / len(grams)


def loop_stats(text: str, n: int = 10) -> Tuple[float, int, bool]:
    """Return ``(repeat_rate, max_count, degenerate)`` for word n-grams.

    ``degenerate`` is deliberately conservative: a verbatim 10-gram must repeat
    at least three times, or half of all 10-gram positions must be repeats in a
    review of at least 50 words.  Ordinary repeated technical terms do not
    trigger it.
    """
    tokens = normalise(text).split()
    grams = _ngrams(tokens, n)
    if not grams:
        return 0.0, 0, False
    counts = Counter(grams)
    max_count = max(counts.values())
    repeat_rate = (len(grams) - len(counts)) / len(grams)
    degenerate = max_count >= 3 or (repeat_rate >= 0.5 and len(tokens) >= 50)
    return repeat_rate, max_count, degenerate


def structure_flags(text: str) -> Dict[str, object]:
    """Detect review structure, meta-commentary, and location references."""
    raw = str(text)
    lowered = raw.lower().replace("\u2019", "'")
    meta_phrases = [
        phrase
        for phrase in META_PHRASES
        if re.search(rf"\b{re.escape(phrase)}\b", lowered)
    ]
    section_refs = len(_SECTION_RE.findall(raw))
    table_refs = len(_TABLE_RE.findall(raw))
    figure_refs = len(_FIGURE_RE.findall(raw))
    equation_refs = len(_EQUATION_RE.findall(raw))
    location_refs = section_refs + table_refs + figure_refs + equation_refs
    return {
        "summary_opening": bool(_SUMMARY_RE.search(lowered[:400])),
        "has_strengths": bool(_STRENGTH_RE.search(lowered)),
        "has_weaknesses": bool(_WEAKNESS_RE.search(lowered)),
        "has_verdict": bool(_VERDICT_RE.search(lowered)),
        "meta_commentary": bool(meta_phrases),
        "meta_phrases": "; ".join(meta_phrases),
        "location_refs": location_refs,
        "section_refs": section_refs,
        "table_refs": table_refs,
        "figure_refs": figure_refs,
        "equation_refs": equation_refs,
        "location_any": location_refs > 0,
    }


def structure_score(row) -> float:
    """Weighted structure score fixed by the protocol."""
    def flag(name: str) -> float:
        value = row[name]
        if isinstance(value, str):
            return 1.0 if value.strip().lower() in {"true", "1", "yes"} else 0.0
        return 1.0 if bool(value) else 0.0

    return (
        0.2 * flag("summary_opening")
        + 0.2 * flag("has_strengths")
        + 0.2 * flag("has_weaknesses")
        + 0.4 * flag("location_any")
    )


def length_fit(words: float, reference_words: float) -> float:
    """Symmetric length agreement in ``[0, 1]``; 1.0 is a perfect match."""
    if words <= 0 or reference_words <= 0:
        return 0.0
    return math.exp(
        -abs(math.log((words + 1.0) / (reference_words + 1.0)))
    )


# ---------------------------------------------------------------------------
# Paper loading and chunking
# ---------------------------------------------------------------------------


def load_papers(
    path: Path | str = DEFAULT_EXPERIMENTS_DIR / PAPERS_FILENAME,
) -> pd.DataFrame:
    """Load the untokenised test papers, one row per ``paper_id``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"paper dataset not found: {path}")
    papers: Dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            paper_id = str(record["paper_id"])
            papers.setdefault(paper_id, record["whole_paper"])
    return pd.DataFrame(
        {"paper_id": list(papers.keys()), "whole_paper": list(papers.values())}
    )


def load_all_references(
    path: Path | str = DEFAULT_EXPERIMENTS_DIR / PAPERS_FILENAME,
) -> Dict[str, List[str]]:
    """Load every human review for every paper.

    A paper can have more than one human review in the dataset (one to four in
    the test split). The archived run CSVs only carry a single selected
    reference, so this helper exposes the rest for the all-reference
    similarity columns.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"paper dataset not found: {path}")
    references: Dict[str, List[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            paper_id = str(record["paper_id"])
            text = normalise(str(record.get("review_content", "")))
            if not text:
                continue
            bucket = references.setdefault(paper_id, [])
            if text not in bucket:
                bucket.append(text)
    return references


def chunk_text(
    text: str,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    max_chunks: int = MAX_CHUNKS,
) -> List[str]:
    """Split text into overlapping character chunks, capped at ``max_chunks``."""
    cleaned = normalise(text)
    if not cleaned:
        return []
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be in [0, size)")
    chunks: List[str] = []
    start = 0
    while start < len(cleaned) and len(chunks) < max_chunks:
        chunks.append(cleaned[start : start + size])
        if start + size >= len(cleaned):
            break
        start += size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Local Ollama embeddings
# ---------------------------------------------------------------------------


def _cache_key(model: str, text: str) -> str:
    digest = hashlib.sha256()
    digest.update(model.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(str(text).encode("utf-8"))
    return digest.hexdigest()


class _EmbeddingCache:
    """SQLite cache keyed by ``sha256(model, text)``."""

    def __init__(self, path: Path | str = CACHE_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(str(path))
        with self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS embeddings ("
                "key TEXT PRIMARY KEY, model TEXT NOT NULL, "
                "dim INTEGER NOT NULL, vector BLOB NOT NULL)"
            )

    def get(self, key: str) -> "np.ndarray | None":
        row = self._conn.execute(
            "SELECT dim, vector FROM embeddings WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        dim, blob = row
        vector = np.frombuffer(blob, dtype=np.float32)
        if vector.size != dim:
            return None
        return vector.copy()

    def put(self, key: str, model: str, vector: np.ndarray) -> None:
        array = np.ascontiguousarray(vector, dtype=np.float32)
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO embeddings (key, model, dim, vector) "
                "VALUES (?, ?, ?, ?)",
                (key, model, int(array.size), array.tobytes()),
            )

    def close(self) -> None:
        self._conn.close()


class OllamaEmbedder:
    """Batch embedding client for a local Ollama server, with a disk cache."""

    def __init__(
        self,
        model: str = EMBED_MODEL,
        host: str = EMBED_HOST,
        cache_path: Path | str = CACHE_PATH,
        batch_size: int = 16,
        timeout: int = 600,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.url = f"{self.host}/api/embed"
        self.batch_size = batch_size
        self.timeout = timeout
        self._cache = _EmbeddingCache(cache_path)

    def _post_batch(self, texts: Sequence[str]) -> np.ndarray:
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "input": list(texts),
                "truncate": True,
                "options": {"num_ctx": 8192},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        embeddings = payload.get("embeddings")
        if not embeddings or len(embeddings) != len(texts):
            raise RuntimeError(
                f"Ollama returned {len(embeddings or [])} embeddings for "
                f"{len(texts)} inputs"
            )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_many(
        self,
        texts: Iterable[str],
        progress=None,
    ) -> Dict[str, np.ndarray]:
        """Return L2-normalised embeddings keyed by the original text."""
        unique = list(dict.fromkeys(str(text) for text in texts))
        vectors: Dict[str, np.ndarray] = {}
        pending: List[Tuple[str, str]] = []
        for text in unique:
            key = _cache_key(self.model, text)
            cached = self._cache.get(key)
            if cached is None:
                pending.append((key, text))
            else:
                vectors[text] = cached

        total = len(pending)
        done = 0
        for start in range(0, total, self.batch_size):
            batch = pending[start : start + self.batch_size]
            array = self._post_batch([text for _, text in batch])
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            array = (array / norms).astype(np.float32)
            for (key, text), vector in zip(batch, array):
                self._cache.put(key, self.model, vector)
                vectors[text] = vector
            done += len(batch)
            if progress is not None:
                progress(done, total)
        return vectors

    def close(self) -> None:
        self._cache.close()


def ensure_server(
    host: str = EMBED_HOST,
    executable: str = OLLAMA_EXECUTABLE,
    models_dir: str = OLLAMA_MODELS_DIR,
    wait_seconds: int = 90,
) -> bool:
    """Start a dedicated local Ollama server if it is not already running."""
    health = f"{host.rstrip('/')}/api/tags"
    try:
        return requests.get(health, timeout=3).ok
    except requests.RequestException:
        pass

    executable_path = Path(executable)
    if not executable_path.is_file():
        return False

    environment = os.environ.copy()
    environment["OLLAMA_MODELS"] = models_dir
    environment["OLLAMA_HOST"] = host.replace("http://", "").replace("https://", "")
    flags = 0
    if sys.platform == "win32":  # keep the server alive after this process exits
        flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
    subprocess.Popen(
        [str(executable_path), "serve"],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
    )
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        try:
            if requests.get(health, timeout=2).ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(1.0)
    return False


# ---------------------------------------------------------------------------
# Per-review metrics
# ---------------------------------------------------------------------------


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    numerator = float(np.dot(left, right))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return float("nan")
    return numerator / denominator


def _paper_grounding(
    review_vector: np.ndarray, chunk_vectors: Sequence[np.ndarray]
) -> Tuple[float, float]:
    if not chunk_vectors:
        return float("nan"), float("nan")
    similarities = sorted(
        (_cosine(review_vector, chunk) for chunk in chunk_vectors), reverse=True
    )
    top_one = similarities[0]
    top_three = float(np.mean(similarities[: min(3, len(similarities))]))
    return top_one, top_three


def _diversity(
    vectors: Sequence[np.ndarray], paper_ids: Sequence[str], sample_limit: int = 100
) -> float:
    if len(vectors) < 2:
        return float("nan")
    candidates = [
        (i, j)
        for i in range(len(vectors))
        for j in range(i + 1, len(vectors))
        if paper_ids[i] != paper_ids[j]
    ]
    if not candidates:
        return float("nan")
    rng = random.Random(42)
    if len(candidates) > sample_limit:
        candidates = rng.sample(candidates, sample_limit)
    return float(np.mean([_cosine(vectors[i], vectors[j]) for i, j in candidates]))


def build_metrics(
    reviews_df: pd.DataFrame,
    papers_df: pd.DataFrame,
    embedder: "OllamaEmbedder | None" = None,
    progress=None,
) -> pd.DataFrame:
    """Compute the Phase 1 per-review metric table.

    If ``embedder`` is ``None`` the embedding columns are ``NaN``; this keeps the
    lexical path testable without a running model.
    """
    papers = dict(zip(papers_df["paper_id"].astype(str), papers_df["whole_paper"]))
    records = reviews_df.to_dict("records")

    missing = sorted({str(record["paper_id"]) for record in records} - set(papers))
    if missing:
        raise ValueError(f"papers missing from the dataset: {missing[:5]}")

    rows: List[dict] = []
    for record in records:
        review = str(record["review"])
        reference = str(record["reference_review"])
        words = count_words(review)
        reference_words = count_words(reference)
        repeat10, max_count, degenerate = loop_stats(review, 10)
        flags = structure_flags(review)
        row = {
            "run": str(record["run"]),
            "arm": str(record["arm"]),
            "paper_id": str(record["paper_id"]),
            "review": review,
            "reference_review": reference,
            "words": words,
            "reference_words": reference_words,
            "output_tokens": float(record["output_tokens"]),
            "seconds": float(record["seconds"]),
            "hit_cap": bool(record["hit_cap"]),
            "rouge_l": float(record["rouge_l"]),
            "repeat5": float(record.get("repeat5", repeat_stats(review, 5))),
            "repeat10": repeat10,
            "max_10gram_count": max_count,
            "degenerate_loop": degenerate,
            **flags,
            "length_fit": length_fit(words, reference_words),
            "emb_ref_cos": float("nan"),
            "emb_paper_cos": float("nan"),
            "emb_paper_cos_top3": float("nan"),
            "emb_div": float("nan"),
        }
        rows.append(row)

    metrics = pd.DataFrame(rows)
    metrics["structure_score"] = metrics.apply(structure_score, axis=1)

    if embedder is None:
        return metrics

    review_vectors = embedder.embed_many(metrics["review"], progress=progress)
    reference_vectors = embedder.embed_many(
        metrics["reference_review"], progress=progress
    )
    paper_ids = sorted(set(metrics["paper_id"]))
    chunk_texts: Dict[str, List[str]] = {
        paper_id: chunk_text(papers[paper_id]) for paper_id in paper_ids
    }
    flat_chunks = [chunk for chunks in chunk_texts.values() for chunk in chunks]
    chunk_vectors = embedder.embed_many(flat_chunks, progress=progress)

    reference_cos = []
    paper_cos = []
    paper_cos_top3 = []
    for row in metrics.itertuples(index=False):
        review_vector = review_vectors[row.review]
        reference_cos.append(_cosine(review_vector, reference_vectors[row.reference_review]))
        top_one, top_three = _paper_grounding(
            review_vector,
            [chunk_vectors[chunk] for chunk in chunk_texts[row.paper_id]],
        )
        paper_cos.append(top_one)
        paper_cos_top3.append(top_three)
    metrics["emb_ref_cos"] = reference_cos
    metrics["emb_paper_cos"] = paper_cos
    metrics["emb_paper_cos_top3"] = paper_cos_top3

    for (run, arm), group in metrics.groupby(["run", "arm"], sort=False):
        indices = list(group.index)
        diversity = _diversity(
            [review_vectors[metrics.at[index, "review"]] for index in indices],
            [metrics.at[index, "paper_id"] for index in indices],
        )
        metrics.loc[indices, "emb_div"] = diversity

    return metrics


def add_all_reference_metrics(
    metrics: pd.DataFrame,
    references: Dict[str, List[str]],
    embedder: "OllamaEmbedder",
    progress=None,
) -> pd.DataFrame:
    """Add similarity against *every* human review of the same paper.

    ``emb_ref_cos_all_max`` is the closest human review, ``emb_ref_cos_all_mean``
    the average over all of them, and ``n_human_reviews`` says how many exist.
    A paper with several human reviews can be matched well by accident, so both
    the best and the average are reported rather than a single number.
    """
    if metrics.empty:
        return metrics

    texts = sorted({text for bucket in references.values() for text in bucket})
    if not texts:
        metrics["emb_ref_cos_all_max"] = float("nan")
        metrics["emb_ref_cos_all_mean"] = float("nan")
        metrics["n_human_reviews"] = 0
        return metrics

    reference_vectors = embedder.embed_many(texts, progress=progress)
    review_vectors = embedder.embed_many(metrics["review"], progress=progress)

    best: List[float] = []
    average: List[float] = []
    counts: List[int] = []
    for row in metrics.itertuples(index=False):
        bucket = references.get(str(row.paper_id)) or [str(row.reference_review)]
        vectors = [
            reference_vectors[text]
            for text in bucket
            if text in reference_vectors
        ]
        if not vectors:
            best.append(float("nan"))
            average.append(float("nan"))
            counts.append(len(bucket))
            continue
        review_vector = review_vectors[row.review]
        similarities = [_cosine(review_vector, vector) for vector in vectors]
        best.append(float(max(similarities)))
        average.append(float(np.mean(similarities)))
        counts.append(len(bucket))

    metrics["emb_ref_cos_all_max"] = best
    metrics["emb_ref_cos_all_mean"] = average
    metrics["n_human_reviews"] = counts
    return metrics
