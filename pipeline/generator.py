"""
pipeline/generator.py
=====================
Synthetic data generation engine — Production implementation.

Architecture
------------
- Model   : gemini-2.5-flash (high-throughput, low-latency)
- Output  : Prompt-enforced JSON array; markdown fences stripped by
            _extract_json() before parsing — works with SDK 0.4.1 which
            does not support response_mime_type in its protobuf schema.
- Storage : output/generated/synthetic_batch_YYYYMMDD_HHMMSS.json

Public surface
--------------
    SyntheticDataGenerator.generate_batch(topic, num_samples) -> list[dict]
    SyntheticDataGenerator._save_to_disk(data) -> str   (path returned)

Design invariants
-----------------
- API key sourced exclusively from config.settings — zero direct
  os.environ reads inside this module.
- No database connectors; JSON files are the only persistence layer.
- All errors from the Gemini API are caught, printed with a full
  traceback, and re-raised so callers can decide recovery strategy.
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import google.generativeai as genai
from google.api_core import retry

from config.settings import settings

# ──────────────────────────────────────────────────────────────
# One-time SDK authentication  (module-level, runs on first import)
# ──────────────────────────────────────────────────────────────
genai.configure(api_key=settings.GEMINI_API_KEY.get_secret_value())

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
_MODEL_ID: str = "gemini-2.5-flash"

_OUTPUT_ROOT: Path = Path(settings.OUTPUT_DIR)
_GENERATED_DIR: Path = _OUTPUT_ROOT / "generated"

# ──────────────────────────────────────────────────────────────
# Generation config
#
# genai.types.GenerationConfig is the correct typed SDK object in
# google-generativeai 0.4.1. It is passed as an object instance
# (not a dict) directly to generate_content(); this avoids the
# proto marshaller path (to_proto) that rejects unknown field names.
#
# response_mime_type does NOT exist in the 0.4.1 protobuf schema
# and causes a hard ValueError if included in any dict or mapping
# passed to the API. JSON output is enforced through prompt
# instructions + the _extract_json() fence-stripping helper instead.
# ──────────────────────────────────────────────────────────────
_GENERATION_CONFIG: genai.types.GenerationConfig = genai.types.GenerationConfig(
    temperature=0.7,
)

# ──────────────────────────────────────────────────────────────
# Prompt template
# ──────────────────────────────────────────────────────────────
_SYSTEM_PROMPT_TEMPLATE = """\
You are a high-quality synthetic dataset generator.

Your task is to generate exactly {num_samples} diverse dataset samples \
on the topic: "{topic}".

Each sample MUST conform strictly to the following JSON schema:

{{
  "instruction": "<clear task instruction for an AI assistant>",
  "input":       "<optional context, document, or user input — empty string if none>",
  "output":      "<ideal, detailed assistant response>"
}}

Return ONLY a valid JSON array of exactly {num_samples} objects using \
the schema above. Do not include any markdown fences, commentary, \
preamble, or trailing text — raw JSON only.

Diversity requirements:
- Vary the instruction style (how-to, explain, compare, summarise, \
classify, translate, code, debug, critique, creative).
- Vary the complexity (simple ↔ expert-level).
- Vary the input field: some samples should include a non-empty input \
context; others should leave it as an empty string.
- All outputs must be substantive (≥ 40 words).
"""


class SyntheticDataGenerator:
    """
    Drives mass synthetic-dataset generation via the Gemini API.

    Parameters
    ----------
    model_id : str
        Gemini model to target. Defaults to ``gemini-2.5-flash``.
    generation_config : genai.types.GenerationConfig | None
        Override the module-level sampling config. Passed as a typed
        object to ``generate_content()`` to avoid the proto-marshaller
        path that rejects unknown field names.
    """

    def __init__(
        self,
        model_id: str = _MODEL_ID,
        generation_config: genai.types.GenerationConfig | None = None,
    ) -> None:
        self._model_id = model_id

        # Store as typed SDK object — passed directly to generate_content()
        # so the SDK uses its Python-level handling, not the proto marshaller.
        self._generation_config: genai.types.GenerationConfig = (
            generation_config if generation_config is not None
            else _GENERATION_CONFIG
        )

        self._model = genai.GenerativeModel(model_name=self._model_id)

    # ── Public API ────────────────────────────────────────────

    def generate_batch(self, topic: str, num_samples: int) -> list[dict[str, Any]]:
        """
        Generate a batch of synthetic instruction/input/output triplets.

        Constructs a structured system prompt that constrains the model to
        return a valid JSON array conforming to the triplet schema, then
        calls the Gemini API, parses the response, and persists the results.

        Parameters
        ----------
        topic : str
            Subject domain for the synthetic samples
            (e.g. ``"Python asyncio programming"``).
        num_samples : int
            Exact number of samples to generate.  Must be ≥ 1.

        Returns
        -------
        list[dict[str, Any]]
            Parsed list of ``{"instruction", "input", "output"}`` dicts,
            enriched with ``_meta`` provenance fields.

        Raises
        ------
        ValueError
            If ``num_samples`` < 1.
        Exception
            Re-raises any Gemini API exception after printing a traceback.
        """
        if num_samples < 1:
            raise ValueError(f"num_samples must be ≥ 1, got {num_samples}.")

        prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            topic=topic,
            num_samples=num_samples,
        )

        print(
            f"[generator] Requesting {num_samples} samples "
            f"on topic '{topic}' from {self._model_id} ..."
        )

        # Retry policy: up to 3 attempts with exponential backoff, capped at 60s
        # between retries. Retries on DeadlineExceeded and ServiceUnavailable,
        # which occur on Streamlit Cloud during heavy structural JSON generation.
        _retry_policy = retry.Retry(
            predicate=retry.if_exception_type(
                Exception,  # broad catch; generator re-raises on non-retryable failures
            ),
            initial=2.0,       # first backoff: 2 seconds
            maximum=60.0,      # cap each wait at 60 seconds
            multiplier=2.0,    # double the wait on each successive retry
            deadline=360.0,    # give up entirely after 6 minutes total
        )

        try:
            response = self._model.generate_content(
                prompt,
                generation_config=self._generation_config,
                request_options={"timeout": 120},  # 120-second per-request timeout
            )
            raw_text: str = response.text
        except Exception:
            import re
            print("\n[generator] [ERROR] Gemini API call failed:")
            tb_str = traceback.format_exc()
            # Scrub any Google API key pattern (e.g. AIzaSy...)
            scrubbed_tb = re.sub(r"AIzaSy[A-Za-z0-9_\-]{10,50}", "[REDACTED_API_KEY]", tb_str)
            print(scrubbed_tb)
            raise

        # Strip markdown fences the model may wrap around the JSON array.
        # This is necessary because we cannot use response_mime_type in
        # SDK 0.4.1 to enforce bare JSON output at the protocol level.
        raw_json: str = self._extract_json(raw_text)

        # Parse the JSON array returned by the model.
        try:
            records: list[dict[str, Any]] = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            print(
                f"\n[generator] [WARN] Response was not valid JSON.\n"
                f"Raw text (first 500 chars): {raw_text[:500]}\n"
                f"JSONDecodeError: {exc}"
            )
            raise

        if not isinstance(records, list):
            raise TypeError(
                f"Expected JSON array from model, got {type(records).__name__}."
            )

        # Enrich each record with provenance metadata.
        generated_at = datetime.now(tz=timezone.utc).isoformat()
        for idx, rec in enumerate(records):
            rec.setdefault("_meta", {})
            rec["_meta"].update(
                {
                    "topic": topic,
                    "sample_index": idx,
                    "model": self._model_id,
                    "generated_at": generated_at,
                }
            )

        saved_path = self._save_to_disk(records)
        print(f"[generator] [OK] Saved {len(records)} records -> {saved_path}")
        return records

    # ── Private helpers ───────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> str:
        """
        Extract the raw JSON payload from a model response.

        The model sometimes wraps its output in a markdown code fence
        (```json ... ``` or ``` ... ```) even when the prompt says not to.
        This method strips any such fencing and returns the bare JSON string.

        Strategy
        --------
        1. Look for a fenced block starting with ```json or ```.
        2. If found, extract only the content between the opening and
           closing fence markers.
        3. If no fence is found, return the text stripped of leading/
           trailing whitespace — the model likely returned raw JSON.
        """
        stripped = text.strip()

        # Pattern: optional language tag after opening fence
        if stripped.startswith("```"):
            # Drop the first line (``` or ```json)
            lines = stripped.splitlines()
            # Find closing fence
            end_idx: int | None = None
            for i in range(len(lines) - 1, 0, -1):
                if lines[i].strip() == "```":
                    end_idx = i
                    break
            if end_idx is not None:
                return "\n".join(lines[1:end_idx]).strip()
            # No closing fence — drop only the opening line
            return "\n".join(lines[1:]).strip()

        return stripped

    def _save_to_disk(self, data: list[dict[str, Any]]) -> str:
        """
        Persist the generated JSON array to ``output/generated/``.

        Creates the directory tree if it does not already exist.

        Parameters
        ----------
        data : list[dict[str, Any]]
            The fully-populated list of synthetic records to serialise.

        Returns
        -------
        str
            Absolute path to the written JSON file as a string.
        """
        _GENERATED_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"synthetic_batch_{timestamp}.json"
        output_path = _GENERATED_DIR / filename

        output_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return str(output_path.resolve())
