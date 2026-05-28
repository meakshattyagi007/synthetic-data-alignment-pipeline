import time
import json
import os
import logging
from typing import List, Dict, Any, Generator
from google import genai
from google.genai import types
from google.genai.errors import APIError

# Setup clean modular diagnostics logging
logger = logging.getLogger(__name__)

# Legacy prompt template definition for test compatibility
_SYSTEM_PROMPT_TEMPLATE = "You are a synthetic data generator. Topic context: {topic}. Task: Generate exactly {num_samples} samples."

from config.settings import settings

# Default client instance to support test mocking frameworks
_api_key = os.environ.get("GEMINI_API_KEY")
if not _api_key or _api_key == 'os.environ.get("GEMINI_API_KEY")' or "Placeholder" in _api_key:
    _api_key = "AIzaSyDummyPlaceholderKey"
client = genai.Client(api_key=_api_key)

class SyntheticDataGenerator:
    def __init__(self):
        # Retrieve a comma-separated string of multiple API keys from secrets
        raw_keys = os.environ.get("GEMINI_API_KEY", "")
        
        # Split the string by commas and strip whitespace to build a clean list
        self.api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        
        # Fallback to dummy placeholder key for testing and environment validation
        for idx, key in enumerate(self.api_keys):
            if key == 'os.environ.get("GEMINI_API_KEY")' or "Placeholder" in key:
                self.api_keys[idx] = "AIzaSyDummyPlaceholderKey"

        if not self.api_keys:
            raise ValueError("No API keys found. Please configure GEMINI_API_KEY as a comma-separated string in Streamlit Secrets.")
        
        self.current_key_index = 0
        self.model_name = "gemini-2.5-flash"
        
        # Initialize the default client with the first key in the pool
        self.client = genai.Client(api_key=self.api_keys[self.current_key_index])

    def _rotate_key(self):
        if len(self.api_keys) <= 1:
            return
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        # Re-initialize the active client instance with the next key credentials
        self.client = genai.Client(api_key=self.api_keys[self.current_key_index])

    def generate_batch(self, topic: str, num_samples: int) -> List[Dict[str, Any]]:
        master_dataset: List[Dict[str, Any]] = []
        CHUNK_SIZE = 1
        MAX_LOCAL_ATTEMPTS = 5
        
        logger.info(f"Starting resilient batch generation pipeline for topic: '{topic}' targeting {num_samples} records.")
        
        # Build an explicit sequential iteration loop to keep the token footprint tiny
        for i in range(num_samples):
            logger.info(f"Processing sample sequence {i + 1}/{num_samples}...")
            
            # Construct a tight, highly restrictive prompt instruction that leaves zero room for verbose text blowing up your token usage
            prompt_content = f"You are a synthetic data generator. Topic context: {topic}. Task: Generate exactly ONE highly compact, professional synthetic record matching your standard input-output data alignment triple schema. Provide short, direct bullet points or single-sentence values. Keep descriptions under 40 words total to minimize network text size. Return only raw JSON data fitting the structure without formatting errors."
            
            # Localized network retry block to catch and neutralize transient service disruptions or 429 rate walls
            attempt = 0
            success = False
            
            while attempt < MAX_LOCAL_ATTEMPTS and not success:
                try:
                    # Enforce application/json mime type configuration so the Google backend returns structured strings without conversational markdown blocks
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt_content,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.7
                        )
                    )
                    
                    raw_text = response.text
                    if not raw_text:
                        raise ValueError("Received an empty text completion payload from the API endpoint.")
                    
                    # Clean and parse the text safely
                    cleaned_json = raw_text.strip()
                    parsed_record = json.loads(cleaned_json)
                    
                    # If the data returns as a list format, extract the core single object inside it
                    if isinstance(parsed_record, list):
                        if len(parsed_record) > 0:
                            parsed_record = parsed_record[0]
                        else:
                            raise ValueError("JSON array returned empty.")
                            
                    master_dataset.append(parsed_record)
                    success = True
                    logger.info(f"Successfully compiled and parsed sample sequence {i + 1}.")
                    
                except APIError as error:
                    attempt += 1
                    logger.warning(f"Pipeline intercept on sequence {i + 1}, attempt {attempt}/{MAX_LOCAL_ATTEMPTS}. Details: {str(error)}")
                    
                    if attempt >= MAX_LOCAL_ATTEMPTS:
                        logger.error(f"Failed to compile sequence {i + 1} after maximum retries. Continuing batch.")
                        break
                        
                    sleep_duration = 65.0
                    try:
                        if hasattr(error, 'details') and error.details:
                            for detail in error.details:
                                if 'retryDelay' in detail and detail['retryDelay']:
                                    # Extract only raw numerical digits and decimal points defensively
                                    raw_delay_str = str(detail['retryDelay'])
                                    cleaned_chars = [char for char in raw_delay_str if char.isdigit() or char == '.']
                                    if cleaned_chars:
                                        sleep_duration = float("".join(cleaned_chars)) + 2.0
                                        break
                    except Exception as parse_err:
                        logger.warning(f"Metadata extraction fallback triggered: {str(parse_err)}")
                        sleep_duration = 65.0
                        
                    logger.info(f"Rate limit backoff initiated. Sleeping pipeline for {sleep_duration} seconds...")
                    self._rotate_key()
                    time.sleep(sleep_duration)
                except Exception as error:
                    attempt += 1
                    logger.warning(f"Pipeline intercept on sequence {i + 1}, attempt {attempt}/{MAX_LOCAL_ATTEMPTS}. Details: {str(error)}")
                    
                    if attempt >= MAX_LOCAL_ATTEMPTS:
                        logger.error(f"Failed to compile sequence {i + 1} after maximum retries. Continuing batch to preserve application stability.")
                        break
                    
                    time.sleep(35.0)
            
            # Mandatory proactive delay pacing between successful chunk generations. 
            # This micro-throttle guarantees the loop stays completely below the 20 requests-per-minute free tier ceiling
            if success and i < num_samples - 1:
                time.sleep(3.5)
            self._rotate_key()
                
        # ── Enrich every record with provenance metadata ──────────────────────
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

        # Persist the generated JSON array to output/generated/.
        output_root = Path(settings.OUTPUT_DIR)
        generated_dir = output_root / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)

        timestamp  = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename   = f"synthetic_batch_{timestamp}.json"
        output_path = generated_dir / filename

        output_path.write_text(
            json.dumps(master_dataset, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Saved {len(master_dataset)} records -> {output_path.resolve()}")

        logger.info(f"Batch processing run finalized. Successfully compiled {len(master_dataset)} total records out of {num_samples} requested.")
        return master_dataset
