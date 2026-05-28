"""
pipeline/alignment_critic.py
============================
RLHF-style alignment critic for synthetic instruction/input/output datasets.

Responsibilities
----------------
1. Accept records structured as ``{"instruction", "input", "output", "_meta"}``
   as produced by ``SyntheticDataGenerator.generate_batch()``.
2. Score each record on four independent rubric dimensions, each normalised
   to the closed interval [0.0, 1.0].
3. Compute an arithmetic mean across the four dimensions as the aggregate
   alignment score, then stamp each record "PASSED" or "FAILED".
4. Chunk the evaluated records into ``passed`` / ``rejected`` groups and
   serialise the full analytical tree into ``output/reviewed/``.

Rubric dimensions
-----------------
- ``length_score``    : Word-count of the output field stays within healthy
                        bounds; rewards substantive but not runaway responses.
- ``diversity_score`` : Type-token ratio across lowercased word tokens in the
                        output; penalises generation loops and repetitive text.
- ``format_score``    : Structural presence check — detects correctly-terminated
                        sentences via terminal punctuation markers.
- ``coherence_score`` : Phrase-repetition penalty; the most-frequent token's
                        share is measured and deducted progressively.

Aggregate score
---------------
    aggregate = (length_score + diversity_score + format_score + coherence_score) / 4

Design invariants (Python 3.12)
--------------------------------
- No database connections or file lookups beyond ``output/`` JSON matrices.
- Pure Python stdlib heuristics — zero ML / GPU dependencies.
- All inputs are plain ``dict`` objects; no custom data-class constraints.
- ``evaluate_record`` is side-effect-free; only ``review_dataset_batch``
  touches the filesystem.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import settings

# ──────────────────────────────────────────────────────────────
# Directory wiring  (resolved once at import time)
# ──────────────────────────────────────────────────────────────
_OUTPUT_ROOT: Path = Path(settings.OUTPUT_DIR)
_GENERATED_DIR: Path = _OUTPUT_ROOT / "generated"
_REVIEWED_DIR: Path = _OUTPUT_ROOT / "reviewed"

# Ensure the reviewed directory exists before any writes.
_REVIEWED_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Length-score calibration constants
# ──────────────────────────────────────────────────────────────
_MIN_WORDS: int = 40    # Below this floor the response is too thin.
_MAX_WORDS: int = 600   # Above this ceiling verbosity is penalised.


class AlignmentCritic:
    """
    Evaluates and filters synthetic instruction-tuning datasets.

    Parameters
    ----------
    pass_threshold : float
        Aggregate score at or above which a record is classified as
        ``"PASSED"``; records below are ``"FAILED"`` (anomalies).
        Must lie in the open interval (0.0, 1.0].  Default: ``0.55``.

    Raises
    ------
    ValueError
        If ``pass_threshold`` is outside the valid range.
    """

    def __init__(self, pass_threshold: float = 0.55) -> None:
        if not (0.0 < pass_threshold <= 1.0):
            raise ValueError(
                f"pass_threshold must be in (0.0, 1.0], got {pass_threshold!r}."
            )
        self.pass_threshold: float = pass_threshold

    # ══════════════════════════════════════════════════════════
    # Public — single-record evaluation
    # ══════════════════════════════════════════════════════════

    def evaluate_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """
        Score a single synthetic record and stamp it PASSED or FAILED.

        The method is **side-effect-free** — it never writes to disk.
        Call ``review_dataset_batch`` for bulk persistence.

        Parameters
        ----------
        record : dict[str, Any]
            A record produced by ``SyntheticDataGenerator.generate_batch``.
            Expected to carry at least an ``"output"`` key; all other keys
            are preserved verbatim in the returned dict.

        Returns
        -------
        dict[str, Any]
            A shallow copy of ``record`` extended with:

            - ``scores``         : dict of the four dimension scores plus
                                   ``aggregate_score``.
            - ``aggregate_score``: arithmetic mean of the four dimensions.
            - ``status``         : ``"PASSED"`` or ``"FAILED"``.
        """
        output_text: str = record.get("output", "")

        length_score: float    = self._length_score(output_text)
        diversity_score: float = self._diversity_score(output_text)
        format_score: float    = self._format_score(output_text)
        coherence_score: float = self._coherence_score(output_text)

        aggregate_score: float = round(
            (length_score + diversity_score + format_score + coherence_score) / 4.0,
            4,
        )

        status: str = "PASSED" if aggregate_score >= self.pass_threshold else "FAILED"

        return {
            **record,
            "scores": {
                "length_score":    length_score,
                "diversity_score": diversity_score,
                "format_score":    format_score,
                "coherence_score": coherence_score,
                "aggregate_score": aggregate_score,
            },
            "aggregate_score": aggregate_score,
            "status": status,
        }

    # ══════════════════════════════════════════════════════════
    # Public — bulk batch review
    # ══════════════════════════════════════════════════════════

    def review_dataset_batch(self, file_path: str) -> dict[str, Any]:
        """
        Read a generated batch JSON file, evaluate every record, and
        persist the full analytical tree to ``output/reviewed/``.

        Parameters
        ----------
        file_path : str
            Absolute or relative path to a JSON file inside
            ``output/generated/`` produced by ``SyntheticDataGenerator``.
            The file must contain a JSON array of record dicts.

        Returns
        -------
        dict[str, Any]
            The complete analytical tree written to disk, containing:

            - ``source_file``    : Resolved path of the input batch.
            - ``reviewed_at``    : UTC ISO-8601 timestamp.
            - ``pass_threshold`` : Threshold applied during this run.
            - ``total``          : Total records processed.
            - ``passed_count``   : Number of PASSED records.
            - ``rejected_count`` : Number of FAILED records.
            - ``pass_rate``      : ``passed_count / total`` (0–1).
            - ``passed``         : List of evaluated records that PASSED.
            - ``rejected``       : List of evaluated records that FAILED.

        Raises
        ------
        FileNotFoundError
            If ``file_path`` does not exist.
        ValueError
            If the file does not contain a JSON array.
        """
        source: Path = Path(file_path).resolve()
        # Enforce that the path stays within the output directory boundaries to prevent path traversal
        if not source.is_relative_to(Path(settings.OUTPUT_DIR).resolve()):
            raise ValueError(
                f"[critic] Path traversal detected: {source} is outside output directory."
            )

        if not source.exists():
            raise FileNotFoundError(
                f"[critic] Batch file not found: {source}"
            )

        print(f"[critic] Reading batch -> {source}")

        raw: list[dict[str, Any]] = json.loads(source.read_text(encoding="utf-8"))

        if not isinstance(raw, list):
            raise ValueError(
                f"[critic] Expected a JSON array in {source}, "
                f"got {type(raw).__name__}."
            )

        print(f"[critic] Evaluating {len(raw)} records ...")

        # ── Evaluate every record ─────────────────────────────
        evaluated: list[dict[str, Any]] = [
            self.evaluate_record(rec) for rec in raw
        ]

        # ── Partition into structural groups ──────────────────
        passed:   list[dict[str, Any]] = [r for r in evaluated if r["status"] == "PASSED"]
        rejected: list[dict[str, Any]] = [r for r in evaluated if r["status"] == "FAILED"]

        total: int = len(evaluated)
        pass_rate: float = round(len(passed) / max(total, 1), 4)

        # ── Build analytical tree ─────────────────────────────
        analytical_tree: dict[str, Any] = {
            "source_file":    str(source),
            "reviewed_at":    datetime.now(tz=timezone.utc).isoformat(),
            "pass_threshold": self.pass_threshold,
            "total":          total,
            "passed_count":   len(passed),
            "rejected_count": len(rejected),
            "pass_rate":      pass_rate,
            "passed":         passed,
            "rejected":       rejected,
        }

        output_path: str = self._save_reviewed(analytical_tree, source.stem)
        print(
            f"[critic] [OK] {len(passed)}/{total} passed "
            f"(rate={pass_rate:.1%}) -> {output_path}"
        )
        return analytical_tree

    # ══════════════════════════════════════════════════════════
    # Private — rubric dimension scorers
    # ══════════════════════════════════════════════════════════

    def _length_score(self, text: str) -> float:
        """
        Score the output's word count against calibrated floor/ceiling bounds.

        Scoring curve
        -------------
        - ``word_count < _MIN_WORDS`` : Linear ramp from 0.0 → 1.0 as the
          count approaches the floor.  Penalises thin, uninformative outputs.
        - ``_MIN_WORDS ≤ word_count ≤ _MAX_WORDS`` : Perfect score of 1.0.
        - ``word_count > _MAX_WORDS`` : Linear decay from 1.0 toward 0.0
          as verbosity exceeds the ceiling.  Penalises runaway generation.
        """
        words: int = len(text.split())

        if words < _MIN_WORDS:
            # Partial credit proportional to how close we are to the floor.
            return round(words / _MIN_WORDS, 4)

        if words > _MAX_WORDS:
            # Excess relative to the ceiling, capped at full penalty.
            excess: int = words - _MAX_WORDS
            penalty: float = min(excess / _MAX_WORDS, 1.0)
            return round(max(0.0, 1.0 - penalty), 4)

        return 1.0

    @staticmethod
    def _diversity_score(text: str) -> float:
        """
        Type-token ratio (TTR) across lowercased word tokens.

        TTR = unique_tokens / total_tokens

        A TTR close to 1.0 indicates high lexical variety; a TTR approaching
        0.0 signals generation loops or near-constant repetition.
        """
        tokens: list[str] = re.findall(r"\b\w+\b", text.lower())
        if not tokens:
            return 0.0
        ttr: float = len(set(tokens)) / len(tokens)
        return round(ttr, 4)

    @staticmethod
    def _format_score(text: str) -> float:
        """
        Structural presence score based on terminal punctuation markers.

        Checks whether the output contains sentences that end with ``.``,
        ``!``, or ``?``.  Rewards outputs structured as coherent prose or
        correctly-terminated lists; penalises raw token dumps.

        Scoring
        -------
        - 0 terminal markers   → 0.10 (minimal grace; not a hard zero)
        - 1–9 terminal markers → linear ramp up to 0.90
        - ≥ 10 terminal markers → 1.0 (plateau; more sentences = good)
        """
        terminal_hits: int = len(re.findall(r"[.!?](?:\s|$)", text))

        if terminal_hits == 0:
            return 0.10

        # Ramp: 1 hit → ~0.19, 9 hits → 0.90, 10+ hits → 1.0
        return round(min(terminal_hits / 10.0, 1.0), 4)

    @staticmethod
    def _coherence_score(text: str) -> float:
        """
        Repetition-penalty coherence score.

        Measures the frequency share of the single most-common word token.
        A natural output keeps this below ~15 %; common function words like
        ``"the"`` or ``"a"`` may legitimately dominate up to that band.

        Penalty calculation
        -------------------
        Let  ``r`` = most_common_count / total_tokens.
        Grace band: first 15 % is forgiven.
        Beyond the grace band the penalty accumulates at 4× rate so that
        a token appearing in 40 % of all positions yields a near-zero score.

            coherence = max(0.0, 1.0 − 4 × max(0.0, r − 0.15))
        """
        tokens: list[str] = re.findall(r"\b\w+\b", text.lower())
        if not tokens:
            return 0.0

        most_common_count: int = Counter(tokens).most_common(1)[0][1]
        repetition_ratio: float = most_common_count / len(tokens)

        raw_penalty: float = max(0.0, repetition_ratio - 0.15)
        coherence: float = max(0.0, 1.0 - raw_penalty * 4.0)
        return round(coherence, 4)

    # ══════════════════════════════════════════════════════════
    # Private — persistence
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _save_reviewed(tree: dict[str, Any], source_stem: str) -> str:
        """
        Write the analytical tree to ``output/reviewed/`` and return the path.

        Filename pattern
        ----------------
        ``reviewed_<source_stem>_YYYYMMDD_HHMMSS.json``
        """
        _REVIEWED_DIR.mkdir(parents=True, exist_ok=True)

        timestamp: str = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename: str = f"reviewed_{source_stem}_{timestamp}.json"
        output_path: Path = _REVIEWED_DIR / filename

        output_path.write_text(
            json.dumps(tree, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return str(output_path.resolve())
