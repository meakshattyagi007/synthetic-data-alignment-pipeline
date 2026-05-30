import time
import json
import os
import logging
from typing import List, Dict, Any
from google import genai
from google.genai import types
from google.genai.errors import APIError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Legacy template alias kept for test-harness compatibility
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT_TEMPLATE = (
    "You are a synthetic data generator. Topic context: {topic}. "
    "Task: Generate exactly {num_samples} samples."
)


class SyntheticDataGenerator:
    """High-throughput synthetic-data generator backed by a multi-key
    rotation pool that scales free-tier API boundaries by up to 12×.

    Configure GEMINI_API_KEY as a comma-separated list of Gemini API keys
    inside your environment / Streamlit Secrets.  The pool engine cycles
    through each key after every sample so that no single key ever sustains
    a 429 back-pressure window alone.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self):
        raw_keys = os.environ.get("GEMINI_API_KEY", "")
        self.api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]

        if not self.api_keys:
            raise ValueError(
                "No API keys found. Please configure GEMINI_API_KEY as a "
                "comma-separated string in Secrets."
            )

        self.current_key_index = 0
        self.model_name = "gemini-2.5-flash"

        logger.info(
            f"Key rotation cluster initialized successfully with "
            f"{len(self.api_keys)} active free-tier keys."
        )
        self.client = genai.Client(api_key=self.api_keys[self.current_key_index])

    # ------------------------------------------------------------------
    # Pool engine rotation helper
    # ------------------------------------------------------------------

    def _rotate_key(self):
        """Advance the active client to the next key slot in the pool ring."""
        if len(self.api_keys) <= 1:
            return
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        logger.info(
            f"🔄 Swapping execution context to API Key Channel Index: "
            f"{self.current_key_index}"
        )
        self.client = genai.Client(api_key=self.api_keys[self.current_key_index])

    # ------------------------------------------------------------------
    # Batch generation control-flow pipeline
    # ------------------------------------------------------------------

    def generate_batch(self, topic: str, num_samples: int) -> List[Dict[str, Any]]:
        """Generate *num_samples* synthetic JSON records for *topic*.

        Each sample is requested individually so the pool rotator can shift
        key context on every iteration, distributing load evenly across the
        entire cluster.  A sample is only skipped when *every* key in the
        pool has returned a rate-limit error back-to-back for that sample.
        """
        master_dataset: List[Dict[str, Any]] = []
        # Ceiling = full pool size: every key must fail before we skip a sample
        MAX_CLUSTER_ATTEMPTS = len(self.api_keys)

        logger.info(
            f"Starting resilient batch generation pipeline for topic: "
            f"'{topic}' targeting {num_samples} records "
            f"(cluster ceiling: {MAX_CLUSTER_ATTEMPTS} keys)."
        )

        for i in range(num_samples):
            logger.info(f"Processing sample sequence {i + 1}/{num_samples}...")

            prompt_content = (
                f"You are a synthetic data generator. Topic context: {topic}. "
                "Task: Generate exactly ONE highly compact, professional synthetic "
                "record matching your standard input-output data alignment triple "
                "schema. Provide short, direct bullet points or single-sentence "
                "values. Keep descriptions under 40 words total to minimize network "
                "text size. Return only raw JSON data fitting the structure without "
                "formatting errors."
            )

            attempt = 0
            success = False

            while attempt < MAX_CLUSTER_ATTEMPTS and not success:
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt_content,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.7,
                        ),
                    )

                    raw_text = response.text
                    if not raw_text:
                        raise ValueError("Empty text payload received.")

                    cleaned_json = raw_text.strip()
                    parsed_record = json.loads(cleaned_json)

                    # Unwrap a single-element list returned by the model
                    if isinstance(parsed_record, list):
                        if len(parsed_record) > 0:
                            master_dataset.append(parsed_record[0])
                        else:
                            raise ValueError("JSON array returned empty.")
                    else:
                        master_dataset.append(parsed_record)

                    success = True
                    logger.info(
                        f"Successfully compiled sample sequence {i + 1}."
                    )

                except APIError as error:
                    attempt += 1
                    logger.warning(
                        f"Key index {self.current_key_index} locked. "
                        f"Intercept on attempt {attempt}/{MAX_CLUSTER_ATTEMPTS}."
                    )

                    # Move context pointer to the next available token channel
                    self._rotate_key()

                    if attempt >= MAX_CLUSTER_ATTEMPTS:
                        # Before skipping completely, initiate a last-resort
                        # 65-second macro system freeze to clear the shared
                        # pool quota umbrella on the Google gateway backend
                        logger.warning(
                            "\U0001f6a8 Entire cluster pool exhausted! "
                            "Enforcing a 65-second macro system reset cooldown..."
                        )
                        time.sleep(65.0)
                        # Reset the local attempt counter to give the cluster
                        # a clean second pass at the exact same sequence item
                        attempt = 0
                        continue

                    # Progressive pacing sleep: scales with consecutive failures
                    # so the shared rate-limit sliding window gets room to breathe.
                    # Routine early rotation → 2.5s | sustained pressure → 5.0s
                    sleep_buffer = 2.5 if attempt < 3 else 5.0
                    logger.info(
                        f"Pacing key switch. Sleeping channel for {sleep_buffer}s..."
                    )
                    time.sleep(sleep_buffer)

                except Exception as gen_err:
                    attempt += 1
                    logger.error(
                        f"Parsing/Unexpected variance: {str(gen_err)}. "
                        "Swapping key..."
                    )
                    self._rotate_key()
                    time.sleep(0.5)

            # Post-item success rotation: balance request loading for the next
            # element only when the current one succeeded
            if success:
                self._rotate_key()
                if i < num_samples - 1:
                    time.sleep(0.5)

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
                    "router":       "google-genai-native",
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
