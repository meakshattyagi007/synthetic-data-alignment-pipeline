# Automated Synthetic Data Generator & Model Alignment Pipeline

A structured Python pipeline for generating synthetic LLM datasets via Google Gemini and evaluating alignment quality through a multi-dimension rubric engine.

---

## Directory Layout

```
.
├── .env.example              ← Copy to .env and add GEMINI_API_KEY
├── requirements.txt          ← Pinned dependency matrix
├── config/
│   ├── __init__.py
│   └── settings.py           ← Pydantic-Settings environment validator (single source of truth)
├── pipeline/
│   ├── __init__.py
│   ├── generator.py          ← SyntheticDataGenerator — Gemini mass-prompting engine
│   └── alignment_critic.py   ← AlignmentCritic — rubric scoring + anomaly rejection
├── dashboard/
│   ├── __init__.py
│   └── app.py                ← Streamlit metric dashboard (single-axis only)
└── output/                   ← Auto-created; JSON result matrices written here
    ├── generated/            ← Raw generation records
    └── reviewed/             ← Scored + filtered review payloads
```

---

## Quick Start

### 1 — Install dependencies
```powershell
pip install -r requirements.txt
```

### 2 — Configure environment
```powershell
Copy-Item .env.example .env
# Edit .env and set GEMINI_API_KEY=<your-key>
```

### 3 — Validate environment (standalone boot check)
```powershell
python config/settings.py
```
✅ Success prints masked key and output directory.  
⛔ Failure prints a human-readable error and exits with code 1.

### 4 — Launch the dashboard
```powershell
streamlit run dashboard/app.py
```

---

## Architectural Constraints

| Rule | Enforcement |
|------|-------------|
| Single source of truth for config | `config/settings.py` — all modules import `settings` from here |
| No database connectors | All persistence is structured JSON in `output/` |
| Crash-on-boot for missing env | `pydantic-settings` + `sys.exit(1)` with human-readable message |
| No dual-axis charts | `dashboard/app.py` — explicit architectural rule, single Y-axis only |
| Pinned dependencies | `requirements.txt` — exact versions, no ranges |

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | ✅ Yes | — | Google AI Studio API key |
| `OUTPUT_DIR` | No | `output` | Root directory for JSON output matrices |
