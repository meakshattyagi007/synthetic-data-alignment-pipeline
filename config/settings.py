"""
config/settings.py
==================
Single-source-of-truth environment configurator for the pipeline.

Boot behaviour
--------------
- Loads .env from the project root (absolute path resolved at import time).
- Validates all present fields immediately via Pydantic Settings.
- Both API key fields are optional so the process never crashes on boot
  due to a missing key — runtime components are responsible for asserting
  the specific key they require.

Usage (anywhere in the project):
    from config.settings import settings
    key = settings.OPENROUTER_API_KEY  # str | None
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

from dotenv import load_dotenv
from typing import Optional
from pydantic import field_validator
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

    # ── API keys — both optional for forward/backward compatibility ──────
    # Primary: OpenRouter gateway credential (routes via openrouter.ai)
    OPENROUTER_API_KEY: Optional[str] = None
    # Legacy: Google AI Studio direct key (retained for compatibility)
    GEMINI_API_KEY: Optional[str] = None

    # ── General settings ─────────────────────────────────────────────────
    OUTPUT_DIR: str = "output"

    # ── Validators ───────────────────────────────────────────────────────
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
    """Scrub OpenRouter/Google API key patterns from error output."""
    import re
    # Mask OpenRouter keys (sk-or-...)
    text = re.sub(r"sk-or-[A-Za-z0-9\-_]{10,80}", "[REDACTED_API_KEY]", text)
    # Mask any residual Google-style keys (AIzaSy...)
    text = re.sub(r"AIzaSy[A-Za-z0-9_\-]{10,50}", "[REDACTED_API_KEY]", text)

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

    scrubbed_exc = _scrub_secrets(str(exc))
    print(f"\n{scrubbed_exc}\n", file=sys.stderr)

    print(
        "  Fix: Ensure a .env file exists at the project root containing:\n"
        "       OPENROUTER_API_KEY=<your-openrouter-key>\n"
        "  Obtain a key at: https://openrouter.ai/keys\n",
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
    print(f"     OUTPUT_DIR         : {settings.OUTPUT_DIR}")
    or_key = settings.OPENROUTER_API_KEY or ""
    gem_key = settings.GEMINI_API_KEY or ""
    print(f"     OPENROUTER_API_KEY : {'*' * 8 + or_key[-4:] if or_key else '[not set]'}")
    print(f"     GEMINI_API_KEY     : {'*' * 8 + gem_key[-4:] if gem_key else '[not set]'}")
