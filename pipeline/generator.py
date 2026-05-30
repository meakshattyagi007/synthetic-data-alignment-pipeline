import time
import json
import os
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

    Configure OPENROUTER_API_KEY as a single credential string inside your
    environment / Streamlit Secrets.  OpenRouter provides an independent
    network infrastructure that bypasses Google AI Studio project-level
    daily token quota blocks.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "Deployment Exception: OPENROUTER_API_KEY variable is missing "
                "from the environment workspace configuration."
            )

        # Point the client execution layer directly at OpenRouter
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        # Target the stable, fast Gemini 2.5 Flash channel hosted via OpenRouter
        self.model_name = "meta-llama/llama-3.3-70b-instruct:free"

        logger.info(
            f"OpenRouter client initialized. Active model channel: {self.model_name}"
        )

    # ------------------------------------------------------------------
    # Batch generation control-flow pipeline
    # ------------------------------------------------------------------

    def generate_batch(self, topic: str, num_samples: int) -> List[Dict[str, Any]]:
        """Generate *num_samples* synthetic JSON records for *topic*.

        Items are processed sequentially (one per API call) to keep the
        token footprint small and stay within free-tier account limits.
        max_tokens=400 is enforced on every call to guarantee compact,
        cost-free responses and pass OpenRouter balance pre-checks.
        """
        master_dataset: List[Dict[str, Any]] = []
        MAX_LOCAL_ATTEMPTS = 3

        logger.info(
            f"Launching sequential OpenRouter dataset generation for topic: "
            f"'{topic}' targeting {num_samples} records."
        )

        for i in range(num_samples):
            logger.info(f"Compiling sample sequence {i + 1}/{num_samples}...")

            prompt_content = (
                f"You are a synthetic data generator. Topic context: {topic}. "
                "Task: Generate exactly ONE highly compact, professional synthetic "
                "record matching your standard input-output data alignment triple "
                "schema. Provide short, direct bullet points or single-sentence "
                "values. Keep descriptions under 40 words total. "
                "Return only raw JSON data fitting the structure without formatting errors."
            )

            attempt = 0
            success = False

            while attempt < MAX_LOCAL_ATTEMPTS and not success:
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt_content}],
                        response_format={"type": "json_object"},
                        max_tokens=400,
                    )

                    raw_text = completion.choices[0].message.content
                    if not raw_text:
                        raise ValueError(
                            "Received an empty content payload from the API gateway."
                        )

                    cleaned_json = raw_text.strip()
                    parsed_record = json.loads(cleaned_json)

                    if isinstance(parsed_record, list):
                        if len(parsed_record) > 0:
                            master_dataset.append(parsed_record[0])
                        else:
                            raise ValueError("Returned JSON data list array was empty.")
                    else:
                        master_dataset.append(parsed_record)

                    success = True
                    logger.info(f"Successfully compiled sample sequence {i + 1}.")

                except Exception as error:
                    attempt += 1
                    logger.warning(
                        f"OpenRouter proxy warning on sequence {i + 1}, "
                        f"attempt {attempt}/{MAX_LOCAL_ATTEMPTS}. "
                        f"Details: {str(error)}"
                    )

                    if attempt >= MAX_LOCAL_ATTEMPTS:
                        logger.error(
                            f"Failed to compile sequence {i + 1} after maximum "
                            "retries. Continuing batch to preserve app stability."
                        )
                        break

                    # Short cooldown pause before retrying with the client proxy
                    time.sleep(4.0)

            # Maintain a clean 1-second processing step delay between successful items
            if success and i < num_samples - 1:
                time.sleep(1.0)

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
            f"Batch generation completed. Successfully compiled "
            f"{len(master_dataset)} total records out of {num_samples} requested."
        )
        return master_dataset
