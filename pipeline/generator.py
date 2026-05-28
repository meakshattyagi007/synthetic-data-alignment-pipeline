"""
pipeline/generator.py
=====================
Synthetic data generation engine — Production implementation.

Architecture
------------
- Router  : OpenRouter (https://openrouter.ai/api/v1) — OpenAI-compatible
            REST endpoint.  Routes the request to Gemini 2.5 Flash on the
            Google backend while bypassing the direct Free Tier 20 RPM cap.
- Client  : openai Python SDK (v1+) in drop-in compatibility mode.
- Output  : Prompt-enforced JSON array; markdown fences stripped by
            _extract_json() before parsing.
- Storage : output/generated/synthetic_batch_YYYYMMDD_HHMMSS.json

Public surface
--------------
    SyntheticDataGenerator.generate_batch(topic, num_samples) -> list[dict]
    SyntheticDataGenerator._save_to_disk(data) -> str   (path returned)

Design invariants
-----------------
- API key sourced exclusively from config.settings — zero raw os.environ
  reads inside this module.
- No database connectors; JSON files are the only persistence layer.
- All errors from the API are caught, printed with a scrubbed traceback,
  and re-raised so callers can decide recovery strategy.
- Rate-limit (HTTP 429) errors are caught natively with an exponential
  back-off retry loop (up to 3 attempts). A proactive 3.5-second inter-
  call throttle keeps the sustained request rate comfortably under any
  tier ceiling.
"""

from __future__ import annotations

import json
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import openai
from openai import OpenAI, RateLimitError

from config.settings import settings

# ──────────────────────────────────────────────────────────────
# OpenRouter client — one instance, module-level singleton.
# Uses the standard OpenAI-compatible SDK pointed at OpenRouter's
# base URL.  The API key is sourced from PipelineSettings so it
# is always validated at boot and never read from os.environ here.
# ──────────────────────────────────────────────────────────────
_client: OpenAI = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=settings.OPENROUTER_API_KEY.get_secret_value(),
)

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
# OpenRouter model slug for Gemini 2.5 Flash.
_MODEL_ID: str = "google/gemini-2.5-flash"

_OUTPUT_ROOT: Path = Path(settings.OUTPUT_DIR)
_GENERATED_DIR: Path = _OUTPUT_ROOT / "generated"

# Sampling parameters passed directly in the chat-completion payload.
_TEMPERATURE: float = 0.7

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
    Drives mass synthetic-dataset generation via OpenRouter → Gemini 2.5 Flash.

    Parameters
    ----------
    model_id : str
        OpenRouter model slug to target.
        Defaults to ``google/gemini-2.5-flash-preview``.
    temperature : float
        Sampling temperature passed to the chat-completion API.
        Defaults to ``0.7``.
    """

    def __init__(
        self,
        model_id: str = _MODEL_ID,
        temperature: float = _TEMPERATURE,
    ) -> None:
        self._model_id   = model_id
        self._temperature = temperature

    # ── Public API ────────────────────────────────────────────

    def generate_batch(self, topic: str, num_samples: int) -> list[dict[str, Any]]:
        """
        Generate a batch of synthetic instruction/input/output triplets.

        Splits ``num_samples`` into mini-chunks of ``CHUNK_SIZE`` samples each
        and issues one API call per chunk. This keeps every response well within
        the ``max_tokens`` window, eliminating ``JSONDecodeError`` truncation on
        large requests.  The existing rate-limit retry policy and proactive
        throttle are applied to every chunk call.

        Parameters
        ----------
        topic : str
            Subject domain for the synthetic samples
            (e.g. ``"Python asyncio programming"``).
        num_samples : int
            Total number of samples to generate.  Must be ≥ 1.

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
            Re-raises any API exception after printing a scrubbed traceback.
        """
        if num_samples < 1:
            raise ValueError(f"num_samples must be ≥ 1, got {num_samples}.")

        # Enforce strict small batch fragmentation to stay completely clear of token limits
        CHUNK_SIZE = 1

        # Build the list of chunk sizes: e.g. num_samples=7 → [2, 2, 2, 1]
        chunks: list[int] = []
        remaining = num_samples
        while remaining > 0:
            chunk = min(CHUNK_SIZE, remaining)
            chunks.append(chunk)
            remaining -= chunk

        total_chunks = len(chunks)
        print(
            f"[generator] Requesting {num_samples} samples on topic '{topic}' "
            f"via OpenRouter -> {self._model_id} "
            f"({total_chunks} chunk(s) of <={CHUNK_SIZE} each) ..."
        )

        # ── Accumulate results across chunks ─────────────────────────────────
        all_records: list[dict[str, Any]] = []

        for chunk_idx, chunk_size in enumerate(chunks, start=1):
            print(
                f"[generator] Chunk {chunk_idx}/{total_chunks}: "
                f"requesting {chunk_size} sample(s) ..."
            )

            # Build a fresh prompt for this exact chunk size so the model
            # knows precisely how many objects to emit.
            chunk_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
                topic=topic,
                num_samples=chunk_size,
            )

            # Delegate the API call (with retry + throttle) to the helper.
            raw_text = self._call_api_with_retry(chunk_prompt)

            # Strip markdown fences the model may wrap around the JSON array.
            raw_json: str = self._extract_json(raw_text)

            # Parse this chunk's JSON array.
            try:
                chunk_records: list[dict[str, Any]] = json.loads(raw_json)
            except json.JSONDecodeError as exc:
                print(
                    f"\n[generator] [WARN] Chunk {chunk_idx} response was not valid JSON.\n"
                    f"Raw text (first 500 chars): {raw_text[:500]}\n"
                    f"JSONDecodeError: {exc}"
                )
                raise

            if not isinstance(chunk_records, list):
                raise TypeError(
                    f"Chunk {chunk_idx}: expected JSON array from model, "
                    f"got {type(chunk_records).__name__}."
                )

            all_records.extend(chunk_records)
            print(
                f"[generator] Chunk {chunk_idx}/{total_chunks}: "
                f"collected {len(chunk_records)} record(s) "
                f"(running total: {len(all_records)}/{num_samples})."
            )

        # ── Enrich every record with provenance metadata ──────────────────────
        generated_at = datetime.now(tz=timezone.utc).isoformat()
        for idx, rec in enumerate(all_records):
            rec.setdefault("_meta", {})
            rec["_meta"].update(
                {
                    "topic":        topic,
                    "sample_index": idx,
                    "model":        self._model_id,
                    "router":       "openrouter.ai",
                    "generated_at": generated_at,
                }
            )

        saved_path = self._save_to_disk(all_records)
        print(f"[generator] [OK] Saved {len(all_records)} records -> {saved_path}")
        return all_records

    # ── Private helpers ───────────────────────────────────────

    def _call_api_with_retry(self, prompt: str) -> str:
        """
        Issue a single chat-completion request to OpenRouter with retry logic.

        Handles ``RateLimitError`` (HTTP 429) with exponential back-off (up to
        3 attempts) and applies a proactive 3.5-second throttle after every
        successful response to keep the sustained request rate under any tier
        ceiling.

        Parameters
        ----------
        prompt : str
            The fully-formatted user prompt to send.

        Returns
        -------
        str
            The raw text content returned by the model.

        Raises
        ------
        RateLimitError
            After ``_MAX_RETRIES`` consecutive 429 responses.
        Exception
            Any other API error, re-raised after printing a scrubbed traceback.
        """
        _MAX_RETRIES:             int   = 3
        _DEFAULT_RATE_LIMIT_SLEEP: float = 60.0  # seconds to wait on 429
        _PROACTIVE_THROTTLE:      float = 3.5   # seconds between successful calls

        attempt: int = 0

        while attempt < _MAX_RETRIES:
            try:
                completion = _client.chat.completions.create(
                    model="google/gemini-2.5-flash",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    temperature=self._temperature,
                    max_tokens=490,  # Limits token allocation to stay under OpenRouter free tier account balance reservation limit (402)
                    timeout=120,       # seconds — prevents hanging on slow cloud hops
                )
                raw_text: str = completion.choices[0].message.content or ""

                # Proactive throttle — keeps sustained RPM under the tier cap.
                print(
                    f"[generator] [THROTTLE] Sleeping {_PROACTIVE_THROTTLE}s "
                    f"to stay under rate-limit ceiling ..."
                )
                time.sleep(_PROACTIVE_THROTTLE)
                return raw_text  # ← success

            except openai.RateLimitError as exc:
                attempt += 1
                # Extract server-suggested delay (e.g. "Retry after 30s").
                delay_match = re.search(r"(\d+)\s*s", str(exc))
                sleep_for = (
                    float(delay_match.group(1)) if delay_match
                    else _DEFAULT_RATE_LIMIT_SLEEP
                )
                print(
                    f"\n[generator] [WARN] Rate limit hit (429 RateLimitError). "
                    f"Attempt {attempt}/{_MAX_RETRIES}. "
                    f"Sleeping {sleep_for:.0f}s before retry ..."
                )
                time.sleep(sleep_for)

                if attempt >= _MAX_RETRIES:
                    print("[generator] [ERROR] Max retries reached on RateLimitError. Aborting.")
                    raise

            except Exception:
                print("\n[generator] [ERROR] OpenRouter API call failed:")
                tb_str = traceback.format_exc()
                scrubbed_tb = re.sub(
                    r"sk-or-[A-Za-z0-9\-_]{10,80}", "[REDACTED_API_KEY]", tb_str
                )
                print(scrubbed_tb)
                raise

        # Unreachable — the loop always returns or raises.
        raise RuntimeError("_call_api_with_retry exited without returning.")  # pragma: no cover


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

        timestamp  = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename   = f"synthetic_batch_{timestamp}.json"
        output_path = _GENERATED_DIR / filename

        output_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return str(output_path.resolve())
