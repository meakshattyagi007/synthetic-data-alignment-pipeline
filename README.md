# Automated Synthetic Data Generator & Model Alignment Pipeline

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Model Targeting](https://img.shields.io/badge/model-Gemini%202.5%20Flash-brightgreen.svg)](https://aistudio.google.com/)
[![Architecture Core](https://img.shields.io/badge/architecture-Pydantic%20%7C%20Streamlit-orange.svg)](https://docs.pydantic.dev/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade, localized synthetic dataset generation engine and RLHF-style alignment pipeline. This project generates structured instruction-tuning datasets utilizing the **Google Gemini 2.5 Flash** model, executes algorithmic rubric evaluations on the generated outputs, and presents the resulting metrics via a sleek, dark-themed Streamlit visualization dashboard.

---

## 📐 System Architecture Flow

The system operates strictly as a local-first pipeline, eliminating database runtime dependency drifts by serializing execution batches directly into local JSON matrices:

```
[ Topic Input ]
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│              Gemini 2.5 Generator Engine                │
│  - Formats instructions with exact JSON array schema    │
│  - Bypasses SDK 0.4.1 proto-marshal restrictions        │
└─────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│        Local Raw Batch Persistent Storage               │
│  - Writes to output/generated/synthetic_batch_*.json     │
└─────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│             Algorithmic Alignment Critic                │
│  - Programmatic 4-rubric scoring (0.0 to 1.0)           │
│  - Evaluates: Length, Diversity, Format, Coherence       │
│  - Partitions batch into Passed & Rejected pools        │
└─────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│        Evaluated Batch Persistent Storage               │
│  - Writes to output/reviewed/reviewed_*.json            │
└─────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│          Single-Axis Streamlit Dashboard                │
│  - KPI metric cards (Total, Pass Rate %, Avg Score)     │
│  - Rubric distribution plots & tabular data inspector   │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Component Deep-Dive

### 1. Secure Environment Validator (`config/settings.py`)
- **Single Source of Truth**: All configurations are handled centrally using Pydantic Settings.
- **Fail-Fast Boot Protocol**: If required environment variables (like `GEMINI_API_KEY`) are missing, empty, or invalid, the boot process is instantly aborted.
- **Traceback Hardening**: Employs a custom `_scrub_secrets` regex filter to sanitize console validation messages, ensuring sensitive API keys are never printed in plaintext in standard error logs.

### 2. Mass Prompting & Extraction Engine (`pipeline/generator.py`)
- **JSON Schema Enforcement**: Constraints the generative model to output raw JSON objects mapping `instruction`, `input`, and `output` fields.
- **SDK Compatibility Layout**: Uses a clean pythonic `genai.types.GenerationConfig` object (setting `temperature=0.7`) to bypass the google-generativeai 0.4.1 protobuf marshaller validation crashes.
- **Fail-Safe Extraction**: Features static markdown code fence extractor logic to strip formatting delimiters (` ```json ` or ` ``` `) from model responses before parsing, preventing JSON load failures.
- **Traceback Protection**: Traceback errors on API failures are processed through a regex sanitizer to eliminate potential key exposures.

### 3. Programmatic Rubric Critic (`pipeline/alignment_critic.py`)
Scores each instruction-tuning triplet across four dimensions (normalized between `0.0` and `1.0`):
*   **Length Score**: Calibrates word count against floor (40 words) and ceiling (600 words) limits to penalize both incomplete responses and verbose runaway generations.
*   **Diversity Score**: Calculates the Type-Token Ratio (TTR) across lowercased tokens to flag generation loops or low lexical variety.
*   **Format Score**: Validates the presence of correct terminal punctuation (`.`, `!`, `?`) to filter raw token dumps.
*   **Coherence Score**: Applies a progressive repetition penalty based on the frequency share of the most common token (grace threshold of 15%).
*   **Path Traversal Sandbox**: Enforces strict relative path resolution boundaries (`.is_relative_to()`) to prevent accessing local system files outside the `output` sandbox directory.

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation
Ensure you are running **Python 3.12**. Install dependencies using the pinned requirements matrix:

```powershell
pip install -r requirements.txt
```

### 2. Configure Environment Context
Clone the template file to `.env` at the root of the workspace directory:

```powershell
copy .env.example .env
```

Open `.env` and configure your API key securely:
```env
GEMINI_API_KEY=AIzaSy...your_actual_api_key...
OUTPUT_DIR=output
```
*Note: The `.gitignore` configuration automatically prevents your local `.env` and generated data files from being tracked in version control.*

### 3. Run the Pipeline
Run the fully automated launch script, which performs a three-stage diagnostic check (virtual environment detection, configuration schemas pre-flight, and port binding constraints) before booting the service:

```powershell
run_system.bat
```

Once running, access the Streamlit visualization panel in your web browser at:
**[http://localhost:8502](http://localhost:8502)**
