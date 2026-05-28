"""
pipeline/generator.py
=====================
Synthetic data generation engine — Native Google GenAI implementation.
"""

from __future__ import annotations

import time
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field

from config.settings import settings

# ──────────────────────────────────────────────────────────────
# Pydantic schema for structured output compliance
# ──────────────────────────────────────────────────────────────
class SyntheticSample(BaseModel):
    instruction: str = Field(description="clear task instruction for an AI assistant")
    input: str = Field(description="optional context, document, or user input — empty string if none")
    output: str = Field(description="ideal, detailed assistant response")

# Initialize the standard client using the environment variable GEMINI_API_KEY
client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
    http_options={"timeout": 120.0}
)

_OUTPUT_ROOT: Path = Path(settings.OUTPUT_DIR)
_GENERATED_DIR: Path = _OUTPUT_ROOT / "generated"

# Prompt template
_SYSTEM_PROMPT_TEMPLATE = """\
You are a high-quality synthetic dataset generator.

Your task is to generate exactly {num_samples} diverse dataset samples \
on the topic: "{topic}".

Each sample MUST conform strictly to the required schema.

Diversity requirements:
- Vary the instruction style (how-to, explain, compare, summarise, \
classify, translate, code, debug, critique, creative).
- Vary the complexity (simple ↔ expert-level).
- Vary the input field: some samples should include a non-empty input \
context; others should leave it as an empty string.
"""

class SyntheticDataGenerator:
    """
    Drives mass synthetic-dataset generation via native Google GenAI client.
    """

    def __init__(
        self,
        model_id: str = "gemini-2.5-flash",
        temperature: float = 0.7,
    ) -> None:
        self._model_id = model_id
        self._temperature = temperature

    def generate_batch(self, topic: str, num_samples: int) -> list[dict[str, Any]]:
        """
        Generate a batch of synthetic instruction/input/output triplets.
        """
        if num_samples < 1:
            raise ValueError(f"num_samples must be ≥ 1, got {num_samples}.")

        # Enforce strict small batch fragmentation to stay completely clear of token limits
        CHUNK_SIZE = 1

        # Build the list of chunk sizes
        chunks: list[int] = [1] * num_samples
        total_chunks = len(chunks)

        print(
            f"[generator] Requesting {num_samples} samples on topic '{topic}' "
            f"via Google GenAI client ({total_chunks} chunk(s)) ..."
        )

        all_records: list[dict[str, Any]] = []

        for i, chunk_size in enumerate(chunks):
            # Alternate the target model on every odd/even chunk iteration loop between the ultra-fast channels:
            model_target = "gemini-2.5-flash" if i % 2 == 0 else "gemini-1.5-flash"

            print(
                f"[generator] Chunk {i + 1}/{total_chunks}: "
                f"requesting 1 sample using {model_target} ..."
            )

            # Build a fresh prompt
            prompt_content = _SYSTEM_PROMPT_TEMPLATE.format(
                topic=topic,
                num_samples=chunk_size,
            )
            # Force model brevity to save tokens
            prompt_content += (
                "\nGenerate exactly ONE highly compact synthetic data record matching the required JSON schema. "
                "The output text values must be extremely short, using crisp bullet points or single-sentence answers. "
                "Keep descriptions under 40 words total to strictly save tokens."
            )

            # Retry loop for rate-limits (HTTP 429)
            success = False
            raw_json = ""
            
            for attempt in range(3):
                try:
                    response = client.models.generate_content(
                        model=model_target,
                        contents=prompt_content,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=SyntheticSample,
                            temperature=self._temperature,
                        ),
                    )
                    raw_json = response.text
                    success = True
                    break
                except APIError as e:
                    # Catch any temporary 429 quota block cleanly, notify user, and wait out the sliding window
                    print(
                        f"\n[generator] [WARN] Rate limit hit or API error ({e}). "
                        f"Attempt {attempt + 1}/3. Sleeping 30.0s before retry ..."
                    )
                    time.sleep(30.0)
                    continue

            if not success:
                print(f"[generator] [ERROR] Chunk {i + 1} generation failed after all attempts. Skipping.")
                continue

            # Parse the single record safely
            try:
                record = json.loads(raw_json)
                if not isinstance(record, dict):
                    raise TypeError(f"Expected JSON object, got {type(record).__name__}")
                all_records.append(record)
                print(
                    f"[generator] Chunk {i + 1}/{total_chunks}: "
                    f"collected 1 record (running total: {len(all_records)}/{num_samples})."
                )
            except (json.JSONDecodeError, TypeError, Exception) as exc:
                print(
                    f"\n[generator] [WARN] Chunk {i + 1} parse failed: {exc}. Skipping chunk.\n"
                    f"Raw text: {raw_json}\n"
                )
                continue

            # Include a short 2.0-second delay at the bottom of the successful loop execution path
            time.sleep(2.0)

        # ── Enrich every record with provenance metadata ──────────────────────
        generated_at = datetime.now(tz=timezone.utc).isoformat()
        for idx, rec in enumerate(all_records):
            rec.setdefault("_meta", {})
            rec["_meta"].update(
                {
                    "topic":        topic,
                    "sample_index": idx,
                    "model":        "rotated-flash-channels",
                    "router":       "google-genai-native",
                    "generated_at": generated_at,
                }
            )

        saved_path = self._save_to_disk(all_records)
        print(f"[generator] [OK] Saved {len(all_records)} records -> {saved_path}")
        return all_records

    def _save_to_disk(self, data: list[dict[str, Any]]) -> str:
        """
        Persist the generated JSON array to output/generated/.
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
