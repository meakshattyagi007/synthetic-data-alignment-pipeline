"""
config/settings.py
==================
Single-source-of-truth environment configurator for the pipeline.

Boot behaviour
--------------
- Loads .env from the project root (absolute path resolved at import time).
- Validates all required fields immediately via Pydantic Settings.
- If GEMINI_API_KEY is absent or empty the process crashes with a
  human-readable stdout message — no silent failures.

Usage (anywhere in the project):
    from config.settings import settings
    key = settings.GEMINI_API_KEY.get_secret_value()
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ──────────────────────────────────────────────────────────────
# Pre-load .env so that pydantic-settings can see the variables.
# Resolve the project root as the parent of *this* file's parent.
# ──────────────────────────────────────────────────────────────
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
_ENV_FILE: Path = _PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=_ENV_FILE, override=False)


# ──────────────────────────────────────────────────────────────
# Settings model
# ──────────────────────────────────────────────────────────────
class PipelineSettings(BaseSettings):
    """
    Environment-validated configuration block.

    All fields are sourced from environment variables (case-insensitive).
    Missing required fields raise a ValidationError at import time, which
    is caught below and converted into a friendly crash message.
    """

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Extra env vars are silently ignored — keeps the model focused.
        extra="ignore",
    )

    # ── Required ─────────────────────────────────────────────
    GEMINI_API_KEY: SecretStr

    # ── Optional (with sensible defaults) ───────────────────
    OUTPUT_DIR: str = "output"

    # ── Validators ───────────────────────────────────────────
    @field_validator("GEMINI_API_KEY", mode="before")
    @classmethod
    def _api_key_must_not_be_empty(cls, v: object) -> object:
        """Reject an explicitly empty string so callers get a clear error."""
        if isinstance(v, str) and v.strip() == "":
            raise ValueError(
                "GEMINI_API_KEY is set but contains only whitespace. "
                "Provide a valid API key."
            )
        return v

    @field_validator("OUTPUT_DIR", mode="after")
    @classmethod
    def _ensure_output_dir_exists(cls, v: str) -> str:
        """Create the output directory tree on first boot if absent."""
        target = _PROJECT_ROOT / v
        target.mkdir(parents=True, exist_ok=True)
        return v


# ──────────────────────────────────────────────────────────────
# Instantiate — crash loudly on validation failure.
# ──────────────────────────────────────────────────────────────
def _scrub_secrets(text: str) -> str:
    """Scrub Google API key patterns and validation input values from error output."""
    import re
    # Mask any string starting with AIzaSy
    text = re.sub(r"AIzaSy[A-Za-z0-9_\-]{10,50}", "[REDACTED_API_KEY]", text)
    
    # Mask raw input values on lines referencing sensitive keys
    lines = []
    for line in text.splitlines():
        if "GEMINI_API_KEY" in line or "api_key" in line.lower():
            line = re.sub(r"input_value=['\"][^'\"]*['\"]", "input_value='[REDACTED]'", line)
        lines.append(line)
    return "\n".join(lines)


try:
    settings = PipelineSettings()  # type: ignore[call-arg]
except Exception as exc:  # pydantic.ValidationError or similar
    # Reconfigure stderr to UTF-8 so output is safe on any terminal codepage.
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    _SEPARATOR = "=" * 70
    print(_SEPARATOR, file=sys.stderr)
    print("  [BOOT FAILURE] PIPELINE BOOT FAILURE -- Environment Validation Error", file=sys.stderr)
    print(_SEPARATOR, file=sys.stderr)
    
    # Securely scrub any secrets in the exception string representation before printing
    scrubbed_exc = _scrub_secrets(str(exc))
    print(f"\n{scrubbed_exc}\n", file=sys.stderr)
    
    print(
        "  Fix: Ensure a .env file exists at the project root containing:\n"
        "       GEMINI_API_KEY=<your-google-ai-studio-key>\n",
        file=sys.stderr,
    )
    print(_SEPARATOR, file=sys.stderr)
    sys.exit(1)


# ──────────────────────────────────────────────────────────────
# Standalone validation smoke-test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Force UTF-8 stdout so this is safe from cmd.exe (cp1252) terminals.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    print("[OK] Environment validation passed.")
    print(f"     OUTPUT_DIR             : {settings.OUTPUT_DIR}")
    print(f"     GEMINI_API_KEY (masked): {'*' * 8}{settings.GEMINI_API_KEY.get_secret_value()[-4:]}")
