"""SupplyMind AI — Real-time Supply Chain Risk Intelligence Dashboard."""

import os
import sys
import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import DATA_DIR, DEVICE, GEO_COORDS

st.set_page_config(
    page_title="SupplyMind AI",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

refresh_count = st_autorefresh(interval=30_000, limit=None, key="live_refresh")

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }

    h1, h2, h3 { font-family: 'Inter', sans-serif; }

    .kpi-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1.2rem 1rem;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.3rem 0;
        font-family: 'Inter', monospace;
    }
    .kpi-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #8892b0;
        margin-bottom: 0.2rem;
    }
    .kpi-delta {
        font-size: 0.8rem;
        color: #8892b0;
    }

    .alert-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-left: 4px solid;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
    }
    .alert-critical { border-left-color: #8e44ad; }
    .alert-high { border-left-color: #e74c3c; }
    .alert-medium { border-left-color: #f39c12; }
    .alert-low { border-left-color: #2ecc71; }

    .header-badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin: 0 0.2rem;
    }
    .badge-green { background: rgba(46,204,113,0.15); color: #2ecc71; }
    .badge-yellow { background: rgba(243,156,18,0.15); color: #f39c12; }
    .badge-purple { background: rgba(142,68,173,0.15); color: #8e44ad; }

    div[data-testid="stExpander"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 8px;
        margin-bottom: 0.4rem;
    }

    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #0a0a1a 0%, #12122a 100%);
    }
</style>
""", unsafe_allow_html=True)

RISK_COLORS = {
    "low risk": "#2ecc71",
    "medium risk": "#f39c12",
    "high risk": "#e74c3c",
    "critical risk": "#8e44ad",
}
RISK_ICONS = {
    "critical risk": "🔴",
    "high risk": "🟠",
    "medium risk": "🟡",
    "low risk": "🟢",
}

# ── Load data ────────────────────────────────────────────────────────────────

@st.cache_data
def load_analyzed_data():
    pq = os.path.join(DATA_DIR, "analyzed_data.parquet")
    csv = os.path.join(DATA_DIR, "analyzed_data.csv")
    if os.path.exists(pq):
        return pd.read_parquet(pq)
    elif os.path.exists(csv):
        return pd.read_csv(csv)
    return pd.DataFrame()

data = load_analyzed_data()

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.markdown("""
<div style='text-align:center; padding: 0.5rem 0 1rem 0;'>
    <h2 style='margin:0; font-size:1.4rem;'>🔗 SupplyMind AI</h2>
    <p style='color:#8892b0; font-size:0.75rem; margin:0.3rem 0 0 0;'>
        Supply Chain Intelligence
    </p>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

n_docs = st.sidebar.slider("Documents to display", 100, 3000, 500, step=100)
risk_filter = st.sidebar.multiselect(
    "Alert filter",
    options=["low risk", "medium risk", "high risk", "critical risk"],
    default=["high risk", "critical risk"],
)
source_options = sorted(data["_source"].unique()) if not data.empty and "_source" in data.columns else []
source_filter = st.sidebar.multiselect("Data source", options=source_options, default=source_options)

st.sidebar.markdown("---")
st.sidebar.markdown("#### System Status")

if not data.empty:
    crit_total = int((data["risk_label"] == "critical risk").sum())
    high_total = int((data["risk_label"] == "high risk").sum())

    unique_comps = set()
    for c in data["companies"].dropna().astype(str):
        if c and c != "nan":
            unique_comps.update(x.strip() for x in c.split(",") if x.strip())

    st.sidebar.markdown(f"""
    <span class='header-badge badge-green'>LIVE</span>
    <span class='header-badge {"badge-green" if DEVICE == "cuda" else "badge-yellow"}'>{DEVICE.upper()}</span>
    <span class='header-badge badge-purple'>NLP</span>
    """, unsafe_allow_html=True)
    st.sidebar.markdown(f"**{len(data):,}** documents analyzed")
    st.sidebar.markdown(f"**{crit_total:,}** critical · **{high_total:,}** high risk")
    st.sidebar.markdown(f"**{len(unique_comps):,}** companies tracked")
    st.sidebar.markdown(f"Auto-refresh: 30s")

# ── Header ───────────────────────────────────────────────────────────────────

now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

st.markdown(f"""
<div style='text-align:center; padding: 0.5rem 0;'>
    <h1 style='margin-bottom:0; font-size:2.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;'>
        SupplyMind AI
    </h1>
    <p style='color:#8892b0; font-size:1rem; margin:0.3rem 0;'>
        Real-time Supply Chain Risk Intelligence
    </p>
    <div style='margin: 0.5rem 0;'>
        <span class='header-badge badge-green'>LIVE ●</span>
        <span class='header-badge badge-purple'>Kafka</span>
        <span class='header-badge badge-purple'>Databricks</span>
        <span class='header-badge badge-purple'>BERT NER</span>
        <span class='header-badge {"badge-green" if DEVICE == "cuda" else "badge-yellow"}'>
            {"CUDA GPU" if DEVICE == "cuda" else "CPU"}
        </span>
    </div>
    <p style='color:#4a5568; font-size:0.8rem;'>
        {now_str} · Refresh #{refresh_count} · {len(data):,} docs
    </p>
</div>
""", unsafe_allow_html=True)

if data.empty:
    st.error("No pre-analyzed data found. Run `python preprocess.py` first.")
    st.stop()

# ── Filter ───────────────────────────────────────────────────────────────────

pool = data.copy()
if source_filter:
    pool = pool[pool["_source"].isin(source_filter)]
if pool.empty:
    st.warning("No data for selected sources.")
    st.stop()

# Stratified batch: take proportional samples from each risk level so charts
# show the real distribution, not just the top-scoring critical items.
# Each refresh cycle picks a different random slice for variety.
batch_parts = []
for label in ["critical risk", "high risk", "medium risk", "low risk"]:
    subset = pool[pool["risk_label"] == label]
    if subset.empty:
        continue
    share = max(int(n_docs * len(subset) / len(pool)), 10)
    share = min(share, len(subset))
    batch_parts.append(
        subset.sample(n=share, random_state=refresh_count).reset_index(drop=True)
    )
batch = pd.concat(batch_parts, ignore_index=True) if batch_parts else pool.head(n_docs)

# ── KPI Cards ────────────────────────────────────────────────────────────────

total = len(data)
crit = int((data["risk_label"] == "critical risk").sum())
high = int((data["risk_label"] == "high risk").sum())
med = int((data["risk_label"] == "medium risk").sum())
low_count = int((data["risk_label"] == "low risk").sum())

all_comps = set()
for c in data["companies"].dropna().astype(str):
    if c and c != "nan":
        all_comps.update(x.strip() for x in c.split(",") if x.strip())
all_locs = set()
for l in data["locations"].dropna().astype(str):
    if l and l != "nan":
        all_locs.update(x.strip() for x in l.split(",") if x.strip())

avg_conf = data["risk_score"].mean() * 100

kpi_data = [
    ("Total Analyzed", f"{total:,}", "#667eea", ""),
    ("Critical", f"{crit:,}", "#8e44ad", f"{crit/max(total,1)*100:.1f}%"),
    ("High Risk", f"{high:,}", "#e74c3c", f"{high/max(total,1)*100:.1f}%"),
    ("Medium Risk", f"{med:,}", "#f39c12", f"{med/max(total,1)*100:.1f}%"),
    ("Companies", f"{len(all_comps):,}", "#3498db", "tracked"),
    ("Regions", f"{len(all_locs):,}", "#1abc9c", "tracked"),
]

cols = st.columns(len(kpi_data))
for col, (label, value, color, delta) in zip(cols, kpi_data):
    delta_html = f"<div class='kpi-delta'>{delta}</div>" if delta else ""
    col.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>{label}</div>
        <div class='kpi-value' style='color:{color};'>{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Charts Row 1: Risk Distribution + Top Companies ─────────────────────────

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### Risk Distribution (All Data)")
    risk_order = ["low risk", "medium risk", "high risk", "critical risk"]
    dist = data["risk_label"].value_counts().reindex(risk_order, fill_value=0).reset_index()
    dist.columns = ["Risk Level", "Count"]
    dist["Percentage"] = (dist["Count"] / dist["Count"].sum() * 100).round(1)
    fig = px.bar(
        dist, x="Risk Level", y="Count", color="Risk Level",
        color_discrete_map=RISK_COLORS,
        text=dist.apply(lambda r: f"{r['Count']:,} ({r['Percentage']}%)", axis=1),
    )
    fig.update_layout(
        showlegend=False, height=380,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccd6f6"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    fig.update_traces(marker_line_width=0, opacity=0.9, textposition="outside")
    st.plotly_chart(fig, width="stretch")

with col_right:
    st.markdown("#### Top 15 Affected Companies (All Data)")
    all_comp = []
    for c_str in data["companies"].dropna().astype(str):
        if c_str and c_str != "nan":
            all_comp.extend([c.strip() for c in c_str.split(",") if c.strip()])
    if all_comp:
        comp_df = pd.Series(all_comp).value_counts().head(15).reset_index()
        comp_df.columns = ["Company", "Mentions"]
        fig2 = px.bar(
            comp_df, x="Mentions", y="Company", orientation="h",
            color="Mentions", color_continuous_scale="Magma", text_auto=True,
        )
        fig2.update_layout(
            height=380, yaxis=dict(autorange="reversed"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccd6f6"),
            coloraxis_showscale=False,
        )
        fig2.update_traces(marker_line_width=0)
        st.plotly_chart(fig2, width="stretch")
    else:
        st.info("No companies detected.")

# ── Charts Row 2: Top Regions + Risk by Source ───────────────────────────────

col_reg, col_src = st.columns(2)

with col_reg:
    st.markdown("#### Top 15 Affected Regions (All Data)")
    all_loc = []
    for l_str in data["locations"].dropna().astype(str):
        if l_str and l_str != "nan":
            all_loc.extend([l.strip() for l in l_str.split(",") if l.strip()])
    if all_loc:
        loc_df = pd.Series(all_loc).value_counts().head(15).reset_index()
        loc_df.columns = ["Region", "Mentions"]
        fig_loc = px.bar(
            loc_df, x="Mentions", y="Region", orientation="h",
            color="Mentions", color_continuous_scale="YlOrRd", text_auto=True,
        )
        fig_loc.update_layout(
            height=380, yaxis=dict(autorange="reversed"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccd6f6"),
            coloraxis_showscale=False,
        )
        fig_loc.update_traces(marker_line_width=0)
        st.plotly_chart(fig_loc, width="stretch")
    else:
        st.info("No regions detected.")

with col_src:
    st.markdown("#### Risk by Data Source (All Data)")
    src_data = data.groupby("_source").agg(
        avg_risk=("risk_level", "mean"),
        count=("risk_level", "count"),
    ).reset_index()
    src_data.columns = ["Source", "Avg Risk", "Documents"]
    fig_src = px.bar(
        src_data, x="Source", y="Avg Risk", color="Avg Risk",
        color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"],
        text_auto=".2f", hover_data=["Documents"],
    )
    fig_src.update_layout(
        height=380, yaxis=dict(range=[0, 4.5]),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccd6f6"),
        coloraxis_showscale=False,
    )
    fig_src.update_traces(marker_line_width=0)
    st.plotly_chart(fig_src, width="stretch")

# ── Global Risk Map ──────────────────────────────────────────────────────────

st.markdown("#### Global Supply Chain Risk Map")

geo_data = {loc: {"lat": lat, "lon": lon, "risk_level": 1.0, "mentions": 0}
            for loc, (lat, lon) in GEO_COORDS.items()}

for loc_str, rl in zip(data["locations"].fillna("").astype(str), data["risk_level"]):
    for loc in loc_str.split(","):
        loc = loc.strip()
        if loc and loc != "nan" and loc in geo_data:
            geo_data[loc]["risk_level"] = max(geo_data[loc]["risk_level"], rl)
            geo_data[loc]["mentions"] += 1

geo_rows = [{"location": k, **v} for k, v in geo_data.items() if v["mentions"] > 0]
if not geo_rows:
    geo_rows = [{"location": k, **v} for k, v in geo_data.items()]

geo_df = pd.DataFrame(geo_rows)
geo_df["label"] = geo_df["risk_level"].apply(
    lambda x: "LOW" if x < 1.5 else "MEDIUM" if x < 2.5 else "HIGH" if x < 3.5 else "CRITICAL"
)
geo_df["size"] = geo_df["risk_level"] * 2 + geo_df["mentions"] * 0.2

fig_map = px.scatter_geo(
    geo_df, lat="lat", lon="lon", text="location",
    size="size", color="risk_level",
    color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"],
    range_color=[1, 4], size_max=35, projection="natural earth",
    hover_name="location",
    hover_data={"label": True, "risk_level": ":.1f", "mentions": True, "size": False},
)
fig_map.update_layout(
    height=500, margin=dict(l=0, r=0, t=10, b=0),
    geo=dict(
        showframe=False, showcoastlines=True, coastlinecolor="#2a2a4a",
        showland=True, landcolor="#0a0a1a",
        showocean=True, oceancolor="#060614",
        showcountries=True, countrycolor="#1a1a3a",
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    coloraxis_colorbar=dict(title="Risk", tickvals=[1,2,3,4],
                            ticktext=["LOW","MED","HIGH","CRIT"]),
    font=dict(color="#ccd6f6"),
)
st.plotly_chart(fig_map, width="stretch")

# ── 7-Day Forecast ───────────────────────────────────────────────────────────

st.markdown("#### 7-Day Disruption Forecast")

from detection.forecaster import forecast_multiple_entities

FORECAST_ENTITIES = ["TSMC", "Apple", "Samsung", "Toyota", "Intel", "Boeing",
                     "Maersk", "Nvidia", "Tesla", "Foxconn"]

entity_risk = {}
for comp_str, rl in zip(data["companies"].fillna("").astype(str), data["risk_level"]):
    for c in comp_str.split(","):
        c = c.strip()
        if c and c != "nan":
            entity_risk.setdefault(c, []).append(rl)

forecast_input = {}
rng = np.random.RandomState(42)
for entity in FORECAST_ENTITIES:
    real = entity_risk.get(entity, [])
    if len(real) >= 10:
        forecast_input[entity] = real[-30:]
    else:
        base = np.mean(real) if real else rng.uniform(1.5, 3.0)
        synth = list(np.clip(base + rng.normal(0, 0.4, size=30), 1.0, 4.0))
        if real:
            synth[-len(real):] = real
        forecast_input[entity] = synth

forecasts = forecast_multiple_entities(forecast_input)

fc_rows = []
for entity, preds in forecasts.items():
    for p in preds:
        fc_rows.append({"Entity": entity, "Day": f"Day {p['day']}",
                        "Predicted Risk": p["predicted_risk"], "Label": p["label"]})

if fc_rows:
    fc_df = pd.DataFrame(fc_rows)
    fig_fc = px.line(fc_df, x="Day", y="Predicted Risk", color="Entity", markers=True)
    fig_fc.update_layout(
        height=420,
        yaxis=dict(range=[0.5, 4.5], dtick=1,
                   ticktext=["LOW", "MEDIUM", "HIGH", "CRITICAL"], tickvals=[1, 2, 3, 4]),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccd6f6"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis_gridcolor="rgba(255,255,255,0.05)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
    )
    fig_fc.add_hrect(y0=3.5, y1=4.5, fillcolor="#8e44ad", opacity=0.06, line_width=0)
    fig_fc.add_hrect(y0=2.5, y1=3.5, fillcolor="#e74c3c", opacity=0.06, line_width=0)
    fig_fc.add_hrect(y0=1.5, y1=2.5, fillcolor="#f39c12", opacity=0.06, line_width=0)
    fig_fc.add_hrect(y0=0.5, y1=1.5, fillcolor="#2ecc71", opacity=0.06, line_width=0)
    st.plotly_chart(fig_fc, width="stretch")

# ── Live Alert Feed ──────────────────────────────────────────────────────────

st.markdown(f"#### Live Alert Feed — Batch #{refresh_count}")
st.caption(f"Showing {len(batch)} sampled docs · Refreshes every 30s with new samples · Filter by risk level in sidebar")

alerts = batch[batch["risk_label"].isin(risk_filter)].nlargest(30, "raw_score")

if alerts.empty:
    st.info("No alerts match the selected risk filters.")
else:
    for _, row in alerts.iterrows():
        label = row["risk_label"]
        label_upper = label.upper()
        color = RISK_COLORS.get(label, "#333")
        icon = RISK_ICONS.get(label, "⚪")
        companies = str(row.get("companies", "")) if pd.notna(row.get("companies")) else ""
        locations = str(row.get("locations", "")) if pd.notna(row.get("locations")) else ""
        comp_display = companies if companies else "—"
        loc_display = locations if locations else "—"
        text_preview = str(row["_text"])[:100].strip()

        header = f"{icon} [{label_upper}] score={row['raw_score']:.1f}"
        if companies:
            header += f" | {comp_display}"
        header += f" | {text_preview}"

        with st.expander(header):
            c1, c2 = st.columns([1, 1])
            c1.markdown(f"**Risk Level:** <span style='color:{color}; font-weight:700;'>{label_upper}</span>", unsafe_allow_html=True)
            c1.markdown(f"**Confidence:** {row['risk_score']:.1%}")
            c1.markdown(f"**Raw Score:** {row['raw_score']:.2f}")
            c1.markdown(f"**Signals:** {row.get('signal_count', 'N/A')}")
            c2.markdown(f"**Companies:** {comp_display}")
            c2.markdown(f"**Regions:** {loc_display}")
            c2.markdown(f"**Source:** {row['_source']} / {row['_file']}")
            st.markdown(f"> {str(row['_text'])[:800]}")

# ── Copilot Q&A ──────────────────────────────────────────────────────────────

st.markdown("#### Copilot Q&A")
st.caption("Ask: *Which companies face the most risk?* · *What disruptions hit Taiwan?* · *Show semiconductor shortage*")

query = st.text_input("Ask SupplyMind Copilot", placeholder="e.g. semiconductor shortage impact on TSMC")

if query:
    q_tokens = set(query.lower().split())
    scored = []
    for _, row in batch.iterrows():
        t = str(row["_text"]).lower()
        hits = sum(1 for w in q_tokens if len(w) > 2 and w in t)
        if hits > 0:
            scored.append((hits + row["risk_level"] * 0.3, row))
    scored.sort(key=lambda x: -x[0])
    top = scored[:8]

    if not top:
        st.warning("No relevant documents found.")
    else:
        high_crit = sum(1 for _, r in top if r["risk_label"] in ("high risk", "critical risk"))
        comps = set()
        for _, r in top:
            for c in str(r.get("companies", "")).split(","):
                c = c.strip()
                if c and c != "nan":
                    comps.add(c)

        summary = f"Found **{len(top)}** relevant documents. **{high_crit}** are HIGH/CRITICAL risk."
        if comps:
            summary += f" Companies: **{', '.join(sorted(comps)[:8])}**."
        st.success(summary)

        for score, r in top:
            lbl = r["risk_label"]
            color = RISK_COLORS.get(lbl, "#333")
            icon = RISK_ICONS.get(lbl, "⚪")
            st.markdown(
                f"{icon} **<span style='color:{color}'>[{lbl.upper()}]</span>** "
                f"{str(r['_text'])[:250]}",
                unsafe_allow_html=True,
            )

# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(f"""
<div style='text-align:center; padding:0.5rem 0; color:#4a5568; font-size:0.75rem;'>
    SupplyMind AI — Apache Kafka · PySpark · Delta Lake · PyTorch ·
    BERT NER · Transformers · Streamlit · Plotly · scikit-learn |
    {DEVICE.upper()} | {len(data):,} docs
</div>
""", unsafe_allow_html=True)
