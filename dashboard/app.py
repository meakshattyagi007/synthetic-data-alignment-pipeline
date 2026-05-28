"""
dashboard/app.py
================
Streamlit frontend for the Automated Synthetic Data Generator &
Model Alignment Pipeline.

Layout
------
SIDEBAR  → Topic prompt · num_samples slider · pass_threshold slider ·
           "Generate Dataset" action button
HEADER   → Three KPI metric cards (total, pass rate, avg alignment score)
CHART    → Single Plotly bar chart — avg per-dimension scores for the
           latest reviewed batch  (single-axis only, no yaxis2)
INSPECTOR→ Tabbed dataframes for Passed / Rejected pools; expandable
           cards showing instruction + output text per record

Design rules (enforced)
-----------------------
- No deprecated Plotly `titlefont` property anywhere.
- No dual-axis charts (yaxis2 is entirely absent).
- All Plotly titles use `title=dict(text="…")` nested syntax.
- Directory paths use pathlib with graceful exist-checks.
- Python 3.12 compatible syntax throughout.
"""

from __future__ import annotations

import os
import sys

# Dynamically discover the absolute root path of the repository
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Force inject it to the front of python's scanning lookup array
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import json
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.settings import settings

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be the very first Streamlit call)
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Synthetic Data Pipeline",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# GLOBAL STYLES
# ══════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── App background ── */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(160deg, #0a0d18 0%, #0f1320 60%, #111827 100%);
    }
    [data-testid="stHeader"] { background: transparent; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1021 0%, #111629 100%);
        border-right: 1px solid #1e2540;
    }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #141929 0%, #1a2035 100%);
        border: 1px solid #252d4a;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.35);
        transition: border-color 0.2s;
    }
    [data-testid="metric-container"]:hover {
        border-color: #4f5fc4;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #e2e8ff !important;
    }
    [data-testid="stMetricLabel"] {
        color: #7080b0 !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
    }

    /* ── Section headers ── */
    .section-label {
        color: #6472c4;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #1e2540;
    }

    /* ── Status pills ── */
    .pill-pass {
        display: inline-block;
        background: rgba(80,220,140,0.15);
        color: #50dc8c;
        border: 1px solid rgba(80,220,140,0.3);
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .pill-fail {
        display: inline-block;
        background: rgba(248,100,100,0.15);
        color: #f86464;
        border: 1px solid rgba(248,100,100,0.3);
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.72rem;
        font-weight: 600;
    }

    /* ── Generate button ── */
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #3b4fd4 0%, #5a3fc4 100%);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1.2rem;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.02em;
        width: 100%;
        transition: opacity 0.2s, transform 0.1s;
        box-shadow: 0 4px 16px rgba(59,79,212,0.35);
    }
    div[data-testid="stButton"] > button:hover {
        opacity: 0.88;
        transform: translateY(-1px);
    }
    div[data-testid="stButton"] > button:active {
        transform: translateY(0px);
    }

    /* ── Tab strip ── */
    button[data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.82rem;
        letter-spacing: 0.04em;
    }

    /* ── Expander ── */
    details summary {
        font-size: 0.85rem;
        color: #9daae0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════
# PATH CONSTANTS
# ══════════════════════════════════════════════════════════════
_OUTPUT_ROOT: Path = Path(settings.OUTPUT_DIR)
_GENERATED_DIR: Path = _OUTPUT_ROOT / "generated"
_REVIEWED_DIR: Path = _OUTPUT_ROOT / "reviewed"

# ══════════════════════════════════════════════════════════════
# DATA HELPERS
# ══════════════════════════════════════════════════════════════

def _latest_reviewed_file() -> Path | None:
    """Return the most recently written file in output/reviewed/, or None."""
    if not _REVIEWED_DIR.exists():
        return None
    files = sorted(_REVIEWED_DIR.glob("reviewed_*.json"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _load_reviewed_tree(path: Path) -> dict[str, Any]:
    """Load one reviewed analytical tree from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def _all_reviewed_trees() -> list[dict[str, Any]]:
    """Load all reviewed trees, newest first."""
    if not _REVIEWED_DIR.exists():
        return []
    files = sorted(
        _REVIEWED_DIR.glob("reviewed_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    trees: list[dict[str, Any]] = []
    for f in files:
        try:
            trees.append(json.loads(f.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return trees


def _latest_generated_file() -> Path | None:
    """Return the most recently saved generated batch file."""
    if not _GENERATED_DIR.exists():
        return None
    files = sorted(_GENERATED_DIR.glob("synthetic_batch_*.json"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _build_records_df(tree: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build (passed_df, rejected_df) DataFrames from a reviewed analytical tree.
    Each row is a scored triplet record.
    """

    def _flatten(pool: list[dict[str, Any]]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for rec in pool:
            scores = rec.get("scores", {})
            rows.append(
                {
                    "instruction": rec.get("instruction", ""),
                    "input": rec.get("input", ""),
                    "output": rec.get("output", ""),
                    "topic": rec.get("_meta", {}).get("topic", ""),
                    "length_score": scores.get("length_score", 0.0),
                    "diversity_score": scores.get("diversity_score", 0.0),
                    "format_score": scores.get("format_score", 0.0),
                    "coherence_score": scores.get("coherence_score", 0.0),
                    "aggregate_score": rec.get("aggregate_score", 0.0),
                    "status": rec.get("status", ""),
                }
            )
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    return (
        _flatten(tree.get("passed", [])),
        _flatten(tree.get("rejected", [])),
    )


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<h1 style='font-size:1.2rem;font-weight:700;color:#c8d0f0;margin-bottom:0'>🧬 Pipeline Controls</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:0.72rem;color:#5a6490;margin-top:2px'>Synthetic Data Generator & Alignment Critic</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    topic: str = st.text_input(
        "Generation Topic",
        value="Python machine learning best practices",
        placeholder="e.g. REST API design patterns",
        help="Subject domain for the synthetic instruction/output pairs.",
    )

    num_samples: int = st.slider(
        "Number of Samples",
        min_value=5,
        max_value=50,
        value=10,
        step=1,
        help="How many instruction/input/output triplets to generate in one batch.",
    )

    pass_threshold: float = st.slider(
        "Pass Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.55,
        step=0.05,
        help="Records with an aggregate alignment score below this are rejected.",
    )

    st.divider()

    run_clicked: bool = st.button(
        "⚡  Generate Dataset",
        use_container_width=True,
        help="Call Gemini, evaluate alignment, and save results to disk.",
    )

    st.divider()
    st.caption(f"Output root: `{settings.OUTPUT_DIR}/`")
    st.caption("Model: `gemini-2.5-flash`")

# ══════════════════════════════════════════════════════════════
# GENERATION ACTION
# ══════════════════════════════════════════════════════════════
if run_clicked:
    if not topic.strip():
        st.sidebar.error("Please enter a topic before generating.")
    else:
        # Lazy import keeps cold-start fast when user is just browsing.
        from pipeline.generator import SyntheticDataGenerator
        from pipeline.alignment_critic import AlignmentCritic

        status_box = st.empty()
        prog = st.progress(0, text="Initialising generation pipeline …")

        try:
            # Step 1 — Generate
            prog.progress(15, text=f"Calling gemini-2.5-flash for {num_samples} samples …")
            gen = SyntheticDataGenerator()
            gen.generate_batch(topic=topic.strip(), num_samples=num_samples)

            # Step 2 — Locate the file just saved
            prog.progress(55, text="Locating saved batch file …")
            latest_gen = _latest_generated_file()
            if latest_gen is None:
                st.error("Generation succeeded but no batch file was found in output/generated/.")
                st.stop()

            # Step 3 — Evaluate alignment
            prog.progress(70, text="Running alignment critic …")
            critic = AlignmentCritic(pass_threshold=pass_threshold)
            tree = critic.review_dataset_batch(file_path=str(latest_gen))

            prog.progress(100, text="Done.")
            prog.empty()

            passed_n: int  = tree.get("passed_count", 0)
            rejected_n: int = tree.get("rejected_count", 0)
            total_n: int    = tree.get("total", 0)
            rate: float     = tree.get("pass_rate", 0.0)

            status_box.success(
                f"✅  Generated **{total_n}** samples — "
                f"**{passed_n} passed** / {rejected_n} rejected "
                f"(pass rate {rate:.1%}). Dashboard refreshed."
            )
            # Bust the Streamlit cache so the new file is picked up immediately.
            st.cache_data.clear()

        except Exception:
            import re
            prog.empty()
            status_box.error("Generation failed — see details below.")
            tb_str = traceback.format_exc()
            # Redact any Google API key pattern (AIzaSy...) from the traceback
            scrubbed_tb = re.sub(r"AIzaSy[A-Za-z0-9_\-]{10,50}", "[REDACTED_API_KEY]", tb_str)
            st.code(scrubbed_tb, language="python")

# ══════════════════════════════════════════════════════════════
# MAIN DASHBOARD HEADER
# ══════════════════════════════════════════════════════════════
st.markdown(
    "<h1 style='font-size:1.9rem;font-weight:700;color:#dde3ff;margin-bottom:0'>"
    "🧬 Synthetic Data Pipeline</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#5a6490;margin-top:4px;font-size:0.85rem'>"
    "Alignment metrics sourced from local JSON output matrices · "
    "single-axis charts · no database dependencies</p>",
    unsafe_allow_html=True,
)
st.divider()

# ══════════════════════════════════════════════════════════════
# LOAD LATEST REVIEWED TREE
# ══════════════════════════════════════════════════════════════
latest_file: Path | None = _latest_reviewed_file()
tree_loaded: dict[str, Any] = _load_reviewed_tree(latest_file) if latest_file else {}

has_data: bool = bool(tree_loaded)

# ══════════════════════════════════════════════════════════════
# KPI METRIC CARDS
# ══════════════════════════════════════════════════════════════
total_generated: int   = tree_loaded.get("total", 0)
passed_count: int      = tree_loaded.get("passed_count", 0)
rejected_count: int    = tree_loaded.get("rejected_count", 0)
pass_rate_val: float   = tree_loaded.get("pass_rate", 0.0)

# Compute average aggregate score across all records in the latest batch.
all_records_flat: list[dict[str, Any]] = (
    tree_loaded.get("passed", []) + tree_loaded.get("rejected", [])
)
avg_alignment: float = (
    round(
        sum(r.get("aggregate_score", 0.0) for r in all_records_flat)
        / max(len(all_records_flat), 1),
        3,
    )
    if all_records_flat
    else 0.0
)

kpi1, kpi2, kpi3 = st.columns(3)

kpi1.metric(
    label="Total Generated Samples",
    value=f"{total_generated:,}",
    help="Total records in the most recently reviewed batch.",
)
kpi2.metric(
    label="Pass Rate",
    value=f"{pass_rate_val * 100:.1f}%",
    help=f"Records scoring ≥ {pass_threshold} aggregate alignment score.",
)
kpi3.metric(
    label="Avg Alignment Score",
    value=f"{avg_alignment:.3f}",
    help="Arithmetic mean of aggregate alignment scores across all records in the latest batch.",
)

st.divider()

# ══════════════════════════════════════════════════════════════
# CHARTS — single-axis only
# ══════════════════════════════════════════════════════════════
if not has_data:
    st.info(
        "No reviewed data found yet. Enter a topic in the sidebar and click "
        "**⚡ Generate Dataset** to produce your first batch.",
        icon="📂",
    )
else:
    passed_df, rejected_df = _build_records_df(tree_loaded)
    all_df: pd.DataFrame = pd.concat([passed_df, rejected_df], ignore_index=True)

    chart_left, chart_right = st.columns(2, gap="large")

    # ── LEFT: Average dimension scores for the latest batch ──
    with chart_left:
        st.markdown(
            '<p class="section-label">Avg Rubric Dimension Scores — Latest Batch</p>',
            unsafe_allow_html=True,
        )

        _dims = ["length_score", "diversity_score", "format_score", "coherence_score"]
        _labels = ["Length", "Diversity", "Format", "Coherence"]
        _palette = ["#6272e4", "#48c5e8", "#f5c842", "#48e8a4"]
        _avgs: list[float] = (
            [round(all_df[d].mean(), 4) for d in _dims]
            if not all_df.empty
            else [0.0, 0.0, 0.0, 0.0]
        )

        fig_bar = go.Figure()
        fig_bar.add_trace(
            go.Bar(
                x=_labels,
                y=_avgs,
                marker=dict(
                    color=_palette,
                    line=dict(color="rgba(255,255,255,0.06)", width=1),
                ),
                text=[f"{v:.3f}" for v in _avgs],
                textposition="outside",
                textfont=dict(color="#c8d0f0", size=12),
                width=0.52,
            )
        )
        fig_bar.update_layout(
            plot_bgcolor="#111827",
            paper_bgcolor="#111827",
            font=dict(color="#9daae0", family="Inter, sans-serif", size=12),
            margin=dict(l=10, r=10, t=16, b=10),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                tickfont=dict(color="#9daae0", size=11),
            ),
            yaxis=dict(
                title=dict(text="Average Score (0–1)", font=dict(color="#6472c4", size=11)),
                range=[0, 1.18],
                gridcolor="#1a2240",
                gridwidth=1,
                zeroline=False,
                tickfont=dict(color="#6472c4", size=10),
            ),
            showlegend=False,
            bargap=0.3,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── RIGHT: Aggregate score distribution histogram ──
    with chart_right:
        st.markdown(
            '<p class="section-label">Aggregate Score Distribution — Latest Batch</p>',
            unsafe_allow_html=True,
        )

        fig_hist = go.Figure()

        if not all_df.empty:
            fig_hist.add_trace(
                go.Histogram(
                    x=all_df["aggregate_score"],
                    nbinsx=16,
                    marker=dict(
                        color="#4f5fc4",
                        opacity=0.80,
                        line=dict(color="rgba(255,255,255,0.08)", width=1),
                    ),
                    name="Aggregate Score",
                )
            )
            # Threshold marker
            fig_hist.add_vline(
                x=pass_threshold,
                line_dash="dot",
                line_color="#f86464",
                line_width=2,
                annotation=dict(
                    text=f"Threshold {pass_threshold}",
                    font=dict(color="#f86464", size=11),
                    bgcolor="rgba(248,100,100,0.12)",
                    bordercolor="rgba(248,100,100,0.3)",
                    borderwidth=1,
                    borderpad=4,
                    yref="paper",
                    y=0.97,
                ),
            )

        fig_hist.update_layout(
            plot_bgcolor="#111827",
            paper_bgcolor="#111827",
            font=dict(color="#9daae0", family="Inter, sans-serif", size=12),
            margin=dict(l=10, r=10, t=16, b=10),
            xaxis=dict(
                title=dict(text="Aggregate Alignment Score", font=dict(color="#6472c4", size=11)),
                gridcolor="#1a2240",
                zeroline=False,
                tickfont=dict(color="#6472c4", size=10),
            ),
            yaxis=dict(
                title=dict(text="Record Count", font=dict(color="#6472c4", size=11)),
                gridcolor="#1a2240",
                zeroline=False,
                tickfont=dict(color="#6472c4", size=10),
            ),
            showlegend=False,
            bargap=0.08,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # DATASET INSPECTOR
    # ══════════════════════════════════════════════════════════
    st.divider()
    st.markdown(
        '<p class="section-label">Dataset Inspector — Passed &amp; Rejected Records</p>',
        unsafe_allow_html=True,
    )

    tab_pass, tab_fail = st.tabs(
        [f"✅  Passed  ({passed_count})", f"❌  Rejected  ({rejected_count})"]
    )

    _SCORE_COLS = ["length_score", "diversity_score", "format_score", "coherence_score", "aggregate_score"]

    def _render_pool(pool_df: pd.DataFrame, pool_label: str) -> None:
        """Render a scored pool as a summary table + expandable record cards."""
        if pool_df.empty:
            st.info(f"No {pool_label} records in the latest batch.", icon="🗂️")
            return

        # Summary dataframe — scores + truncated instruction
        summary_df = pool_df[["instruction", *_SCORE_COLS]].copy()
        summary_df["instruction"] = summary_df["instruction"].str[:90] + "…"

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "instruction": st.column_config.TextColumn("Instruction (preview)", width="large"),
                "length_score": st.column_config.ProgressColumn(
                    "Length", format="%.3f", min_value=0.0, max_value=1.0, width="small"
                ),
                "diversity_score": st.column_config.ProgressColumn(
                    "Diversity", format="%.3f", min_value=0.0, max_value=1.0, width="small"
                ),
                "format_score": st.column_config.ProgressColumn(
                    "Format", format="%.3f", min_value=0.0, max_value=1.0, width="small"
                ),
                "coherence_score": st.column_config.ProgressColumn(
                    "Coherence", format="%.3f", min_value=0.0, max_value=1.0, width="small"
                ),
                "aggregate_score": st.column_config.ProgressColumn(
                    "Aggregate", format="%.3f", min_value=0.0, max_value=1.0, width="small"
                ),
            },
        )

        # Expandable per-record detail cards
        st.markdown("---")
        st.markdown(
            "<p style='font-size:0.75rem;color:#5a6490;font-weight:600;"
            "letter-spacing:0.08em;text-transform:uppercase'>Full Record Detail</p>",
            unsafe_allow_html=True,
        )
        for idx, row in pool_df.iterrows():
            agg: float = row["aggregate_score"]
            pill_class: str = "pill-pass" if row["status"] == "PASSED" else "pill-fail"
            label_text: str = row["status"]

            preview: str = str(row["instruction"])[:72]
            if len(str(row["instruction"])) > 72:
                preview += "…"

            with st.expander(
                f"#{int(idx) + 1}  ·  {preview}  "
                f"  [score: {agg:.3f}]",
                expanded=False,
            ):
                st.markdown(
                    f'<span class="{pill_class}">{label_text}</span>'
                    f"&nbsp;&nbsp;<span style='color:#5a6490;font-size:0.75rem'>"
                    f"Aggregate: <strong style='color:#c8d0f0'>{agg:.4f}</strong></span>",
                    unsafe_allow_html=True,
                )
                st.markdown("**Instruction**")
                st.markdown(
                    f"<div style='background:#0d1021;border:1px solid #1e2540;"
                    f"border-radius:8px;padding:0.8rem 1rem;color:#c8d0f0;"
                    f"font-size:0.88rem;line-height:1.6'>{row['instruction']}</div>",
                    unsafe_allow_html=True,
                )

                input_text: str = str(row.get("input", "")).strip()
                if input_text:
                    st.markdown("**Input Context**")
                    st.markdown(
                        f"<div style='background:#0d1021;border:1px solid #1e2540;"
                        f"border-radius:8px;padding:0.8rem 1rem;color:#9daae0;"
                        f"font-size:0.85rem;line-height:1.6'>{input_text}</div>",
                        unsafe_allow_html=True,
                    )

                st.markdown("**Output**")
                st.markdown(
                    f"<div style='background:#0d1021;border:1px solid #252d4a;"
                    f"border-radius:8px;padding:0.8rem 1rem;color:#c8d0f0;"
                    f"font-size:0.88rem;line-height:1.7'>{row['output']}</div>",
                    unsafe_allow_html=True,
                )

                score_c1, score_c2, score_c3, score_c4 = st.columns(4)
                score_c1.metric("Length",    f"{row['length_score']:.3f}")
                score_c2.metric("Diversity", f"{row['diversity_score']:.3f}")
                score_c3.metric("Format",    f"{row['format_score']:.3f}")
                score_c4.metric("Coherence", f"{row['coherence_score']:.3f}")

    with tab_pass:
        _render_pool(passed_df, "passed")

    with tab_fail:
        _render_pool(rejected_df, "rejected")

    # ── Historical batch selector ──────────────────────────────
    st.divider()
    all_trees = _all_reviewed_trees()
    if len(all_trees) > 1:
        st.markdown(
            '<p class="section-label">Historical Batch Overview</p>',
            unsafe_allow_html=True,
        )
        hist_rows: list[dict[str, Any]] = []
        for t in all_trees:
            hist_rows.append(
                {
                    "Reviewed At": t.get("reviewed_at", "")[:19].replace("T", " "),
                    "Source File": Path(t.get("source_file", "")).name,
                    "Total": t.get("total", 0),
                    "Passed": t.get("passed_count", 0),
                    "Rejected": t.get("rejected_count", 0),
                    "Pass Rate": f"{t.get('pass_rate', 0.0):.1%}",
                }
            )
        st.dataframe(
            pd.DataFrame(hist_rows),
            use_container_width=True,
            hide_index=True,
        )
#   P l a t f o r m   p r o d u c t i o n   b u i l d   t r i g g e r   v e r i f i c a t i o n   m a r k :   2 0 2 6 - 0 5 - 2 8  
 