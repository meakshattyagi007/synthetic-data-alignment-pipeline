import time
import json
import os
import re
import logging
from typing import List, Dict, Any
import openai

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Legacy template alias kept for test-harness compatibility
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT_TEMPLATE = (
    "You are a synthetic data generator. Topic context: {topic}. "
    "Task: Generate exactly {num_samples} samples."
)


class SyntheticDataGenerator:
    """Sequential synthetic-data generator routed through the OpenRouter API
    gateway using the standard OpenAI client interface.

    Includes a bulletproof text-extraction layer that strips markdown fences,
    handles NoneType payloads, and performs brace-boundary fallback trimming
    before JSON parsing — accommodating open-weights models that return
    conversational prefixes or code-block wrappers around their output.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "Configuration Error: OPENROUTER_API_KEY is absent from "
                "environment variables."
            )

        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model_name = "deepseek/deepseek-v4-flash:free"

        logger.info(
            f"OpenRouter client initialized. Active model channel: {self.model_name}"
        )

    # ------------------------------------------------------------------
    # Internal: resilient JSON extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(raw_text: str) -> str:
        """Strip markdown fences and stray text from *raw_text*, returning
        the innermost JSON object string ready for ``json.loads``.

        Strategy (applied in order):
        1. If backtick fences are present, extract the block content via regex.
        2. Strip any residual fence markers if the regex finds no capture group.
        3. Trim leading/trailing characters outside the outermost ``{…}`` pair.
        """
        cleaned = raw_text.strip()

        # Step 1 & 2 — backtick fence removal
        if "```" in cleaned:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
            if match:
                cleaned = match.group(1).strip()
            else:
                # Fallback: strip all fence markers and clean up
                cleaned = re.sub(r"```[a-zA-Z]*", "", cleaned).strip()

        # Step 3 — brace-boundary trim: handle conversational text bleed
        if not cleaned.startswith("{") and "{" in cleaned:
            cleaned = cleaned[cleaned.find("{"):]
        if not cleaned.endswith("}") and "}" in cleaned:
            cleaned = cleaned[: cleaned.rfind("}") + 1]

        return cleaned

    # ------------------------------------------------------------------
    # Batch generation control-flow pipeline
    # ------------------------------------------------------------------

    def generate_batch(self, topic: str, num_samples: int) -> List[Dict[str, Any]]:
        """Generate *num_samples* synthetic JSON records for *topic*.

        Items are processed sequentially (one per API call).  Every response
        passes through ``_extract_json`` before ``json.loads`` to handle
        open-weights model quirks (markdown fences, conversational prefixes).
        """
        master_dataset: List[Dict[str, Any]] = []
        MAX_LOCAL_ATTEMPTS = 3

        logger.info(
            f"Starting sequential dataset construction for topic: "
            f"'{topic}' targeting {num_samples} records."
        )

        for i in range(num_samples):
            logger.info(f"Compiling sample sequence {i + 1}/{num_samples}...")

            prompt_content = (
                f"You are a synthetic data generator. Topic context: {topic}. "
                "Task: Generate exactly ONE highly compact, professional synthetic "
                "record matching your standard input-output data alignment triple "
                "schema. Return ONLY a raw JSON object string. Do not include "
                "markdown code blocks, backticks, or text descriptions. "
                "Output must start with '{' and end with '}'."
            )

            attempt = 0
            success = False

            while attempt < MAX_LOCAL_ATTEMPTS and not success:
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt_content}],
                        temperature=0.3,
                        max_tokens=400,
                    )

                    # Guard 1 — NoneType / empty payload
                    if (
                        not completion
                        or not completion.choices
                        or completion.choices[0].message.content is None
                    ):
                        raise ValueError(
                            "API gateway returned an unreadable or completely "
                            "empty response object."
                        )

                    raw_text = completion.choices[0].message.content.strip()

                    # Guard 2 — blank content string
                    if not raw_text:
                        raise ValueError(
                            "Response content string is empty after stripping."
                        )

                    # Extract and clean JSON from any wrapper text / fences
                    cleaned_json = self._extract_json(raw_text)

                    # Guard 3 — nothing parseable survived extraction
                    if not cleaned_json:
                        raise ValueError(
                            f"JSON extraction yielded an empty string. "
                            f"Raw payload: {raw_text[:120]!r}"
                        )

                    parsed_record = json.loads(cleaned_json, strict=False)

                    if isinstance(parsed_record, list):
                        if len(parsed_record) > 0:
                            master_dataset.append(parsed_record[0])
                        else:
                            raise ValueError("Decoded list array instance was empty.")
                    else:
                        master_dataset.append(parsed_record)

                    success = True
                    logger.info(
                        f"Successfully compiled sample sequence {i + 1}."
                    )

                except Exception as error:
                    attempt += 1
                    logger.warning(
                        f"Data normalization alert on sequence {i + 1}, "
                        f"attempt {attempt}/{MAX_LOCAL_ATTEMPTS}. "
                        f"Details: {str(error)}"
                    )

                    if attempt >= MAX_LOCAL_ATTEMPTS:
                        logger.error(
                            f"Failed parsing validation for row {i + 1}. "
                            "Continuing batch execution."
                        )
                        break

                    time.sleep(3.0)

            if success and i < num_samples - 1:
                time.sleep(1.0)

        logger.info(
            f"Batch sequence execution finalized. "
            f"Captured {len(master_dataset)} verified records."
        )

        # ── Enrich every record with provenance metadata ──────────────────
        from datetime import datetime, timezone
        from pathlib import Path
        from config.settings import settings

        generated_at = datetime.now(tz=timezone.utc).isoformat()
        for idx, rec in enumerate(master_dataset):
            rec.setdefault("_meta", {})
            rec["_meta"].update(
                {
                    "topic":        topic,
                    "sample_index": idx,
                    "model":        self.model_name,
                    "router":       "openrouter",
                    "generated_at": generated_at,
                }
            )

        # ── Persist the generated JSON array to output/generated/ ─────────
        output_root = Path(settings.OUTPUT_DIR)
        generated_dir = output_root / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"synthetic_batch_{timestamp}.json"
        output_path = generated_dir / filename

        output_path.write_text(
            json.dumps(master_dataset, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Saved {len(master_dataset)} records → {output_path.resolve()}")

        logger.info(
            f"Batch processing run finalized. Successfully compiled "
            f"{len(master_dataset)} total records out of {num_samples} requested."
        )
        return master_dataset
