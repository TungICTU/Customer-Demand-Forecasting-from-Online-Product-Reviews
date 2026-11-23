import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# 1. PAGE CONFIG & ACADEMIC REPORT STYLING
# =============================================================================
st.set_page_config(
    page_title="Consumer Demand Forecasting from Online Reviews",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    [data-testid="stDecoration"] { display: none; }
    header[data-testid="stHeader"] {
        background: rgba(248, 250, 252, 0.9);
        backdrop-filter: blur(8px);
    }

    .stApp { background-color: #F8FAFC; color: #0F172A; }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1260px;
    }

    /* Native bordered containers as clean cards */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0 !important;
        border-radius: 14px !important;
        padding: 1.1rem 1.25rem !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }

    /* Minimalist Editorial Header */
    .report-header {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.4rem 1.75rem;
        margin-bottom: 1.25rem;
    }
    .report-kicker {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #4338CA;
        margin-bottom: 6px;
    }
    .report-title {
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #0F172A;
        margin: 0 0 6px 0;
        line-height: 1.25;
    }
    .report-abstract {
        font-size: 0.88rem;
        color: #475569;
        margin: 0;
        line-height: 1.55;
        max-width: 920px;
    }

    /* KPI Metrics */
    .kpi {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1rem 1.15rem;
    }
    .kpi.primary { border-top: 3px solid #4338CA; }
    .kpi.positive { border-top: 3px solid #0D9488; }
    .kpi-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748B;
    }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.65rem;
        font-weight: 600;
        color: #0F172A;
        margin: 6px 0 4px 0;
        line-height: 1.1;
    }
    .kpi-note { font-size: 12px; color: #475569; }

    /* Section Titles inside cards */
    .sec-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0 0 2px 0;
    }
    .sec-sub {
        font-size: 0.78rem;
        color: #64748B;
        margin: 0 0 12px 0;
    }

    /* Clean Segmented Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: #FFFFFF;
        padding: 5px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        margin-bottom: 1.1rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 36px;
        border-radius: 8px;
        padding: 0 16px;
        font-size: 0.84rem;
        font-weight: 600;
        color: #475569;
        border: none !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [aria-selected="true"] {
        background-color: #0F172A !important;
        color: #FFFFFF !important;
    }

    /* Review Cards */
    .rev-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.7rem;
    }
    .rev-card.pos { border-left: 3px solid #0D9488; }
    .rev-card.neg { border-left: 3px solid #BE123C; }
    .rev-meta {
        display: flex;
        justify-content: space-between;
        font-size: 11px;
        color: #64748B;
        margin-bottom: 5px;
    }
    .rev-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10.5px;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 999px;
    }
    .rev-tag.pos { background: #CCFBF1; color: #115E59; }
    .rev-tag.neg { background: #FFE4E6; color: #9F1239; }
    .rev-title { font-size: 13px; font-weight: 600; color: #0F172A; margin-bottom: 4px; }
    .rev-body { font-size: 12px; color: #334155; line-height: 1.5; margin-bottom: 7px; }
    .asp-pill {
        display: inline-block;
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        color: #334155;
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        padding: 1px 6px;
        border-radius: 5px;
        margin-right: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOT_CFG = {"displayModeBar": False}


def style_fig(fig, height=340, x_title="", y_title=""):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color="#334155"),
        height=height,
        margin=dict(l=10, r=14, t=34, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None),
        xaxis=dict(title=x_title, showgrid=False, linecolor="#E2E8F0"),
        yaxis=dict(title=y_title, gridcolor="#F1F5F9", zerolinecolor="#E2E8F0"),
        hovermode="x unified",
    )
    return fig


def section_heading(title: str, subtitle: str = ""):
    st.markdown(f'<div class="sec-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="sec-sub">{subtitle}</div>', unsafe_allow_html=True)


def kpi(label: str, value: str, note: str, style_cls: str = ""):
    st.markdown(
        f"""
        <div class="kpi {style_cls}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def df_to_latex(df: pd.DataFrame, caption: str) -> str:
    cols = "l" + "r" * (len(df.columns) - 1)
    header = " & ".join(str(c).replace("_", "\\_").replace("%", "\\%") for c in df.columns)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\hline",
        f"{header} \\\\",
        "\\hline",
    ]
    for _, row in df.iterrows():
        vals = [str(v).replace("_", "\\_").replace("%", "\\%") for v in row.values]
        lines.append(" & ".join(vals) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


# =============================================================================
# 2. LOAD DATA DIRECTLY FROM app_data/
# =============================================================================
DATA_DIR = Path(__file__).parent / "app_data"
RES_DIR = DATA_DIR / "results"


@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    meta = json.loads((DATA_DIR / "meta.json").read_text(encoding="utf-8"))
    products = pd.read_csv(DATA_DIR / "products.csv")
    weekly = pd.read_parquet(DATA_DIR / "weekly_demand.parquet")
    nlp = pd.read_parquet(DATA_DIR / "weekly_nlp_features.parquet")
    pred = pd.read_parquet(DATA_DIR / "predictions.parquet")
    reviews = pd.read_parquet(DATA_DIR / "review_samples.parquet")

    weekly["week"] = pd.to_datetime(weekly["week"])
    nlp["week"] = pd.to_datetime(nlp["week"])
    pred["week"] = pd.to_datetime(pred["week"])

    results = {}
    if RES_DIR.exists():
        for p in sorted(RES_DIR.glob("*.csv")):
            if p.stem == "v1_vs_v2":
                raw = pd.read_csv(p, header=[0, 1])
                clean_cols = []
                for a, b in raw.columns:
                    if "Unnamed" in str(b) or str(b).strip() == "":
                        clean_cols.append(str(a).replace("feature_set", "Features").replace("model", "Model"))
                    else:
                        clean_cols.append(f"{str(a).capitalize()} {b}")
                raw.columns = clean_cols
                results[p.stem] = raw
            else:
                results[p.stem] = pd.read_csv(p)

    # Translate Vietnamese labels in exported CSVs for English presentation
    if "volume_split" in results:
        vs = results["volume_split"].copy()
        if "group" in vs.columns:
            vs["group"] = vs["group"].replace({"nhiều review": "High Volume", "ít review": "Low Volume"})
        vs = vs.rename(
            columns={
                "group": "Volume Group",
                "model": "Model",
                "chuỗi cải thiện (F2→F4)": "Improved Series",
                "p Wilcoxon (F2→F4)": "Wilcoxon p-value",
            }
        )
        results["volume_split"] = vs

    if "significance" in results:
        sg = results["significance"].copy().rename(
            columns={
                "model": "Model",
                "so_sánh": "Comparison",
                "tỉ lệ chuỗi cải thiện": "Improved Series",
                "thay đổi MAE TB %": "Mean MAE Δ (%)",
                "p (Wilcoxon)": "Wilcoxon p-value",
            }
        )
        results["significance"] = sg

    train_end = pd.Timestamp(meta.get("train_end", "2021-12-27"))
    tr = weekly[(weekly["week"] <= train_end) & (weekly["launched"] == True)]
    vol = tr.groupby("series_id")["n_verified"].mean()
    median_vol = float(vol.median()) if len(vol) else 0.0

    products["reviews_per_week"] = products["series_id"].map(vol).fillna(0.0)
    products["volume_group"] = np.where(
        products["reviews_per_week"] >= median_vol, "High Volume", "Low Volume"
    )
    return meta, products, weekly, nlp, pred, reviews, results, median_vol


if not DATA_DIR.exists():
    st.error("Data directory not found.")
    st.stop()

meta, products, weekly, nlp, pred, reviews, results, median_vol = load_data()

# =============================================================================
# 3. SIDEBAR — EXPERIMENTAL CONFIGURATION
# =============================================================================
with st.sidebar:
    st.markdown("#### Experimental Setup")
    best_m = meta.get("best_model", {})
    with st.container(border=True):
        st.caption(
            f"**Dataset:** Amazon Reviews 2023\n\n"
            f"**Normalization:** `{meta.get('norm', 'window')}` ({meta.get('version', 'v2')})\n\n"
            f"**Sliding Window:** `L = {meta.get('L', 26)}` → `H = {meta.get('H', 4)}` wks\n\n"
            f"**Train Split:** ≤ `{meta.get('train_end', '-')}`\n\n"
            f"**Validation Split:** ≤ `{meta.get('val_end', '-')}`\n\n"
            f"**Selected Model:** `{best_m.get('model', '-')} {best_m.get('feature_set', '-')}`\n\n"
            f"**Volume Median:** `{median_vol:.1f}` rev/wk"
        )

    st.markdown("#### Feature Hierarchy")
    with st.container(border=True):
        st.caption(
            "**F0** — Historical demand\n\n"
            "**F1** — F0 + Calendar & holidays\n\n"
            "**F2** — F1 + Rating & helpfulness\n\n"
            "**F3** — F2 + RoBERTa sentiment\n\n"
            "**F4** — F3 + 7 zero-shot aspects"
        )

# =============================================================================
# 4. EDITORIAL REPORT HEADER
# =============================================================================
n_series = int(weekly["series_id"].nunique())
n_cats = int(weekly["category"].nunique())
n_weeks = int(weekly["week"].nunique())
tot_reviews = int(weekly["n_reviews"].sum())
tot_verified = int(weekly["n_verified"].sum())

st.markdown(
    f"""
    <div class="report-header">
        <div class="report-kicker">Capstone Research Report • Amazon Reviews 2023</div>
        <h1 class="report-title">Consumer Demand Forecasting from Online Product Reviews</h1>
        <p class="report-abstract">
            Empirical evaluation of temporal NLP signals and global deep learning architectures across
            <b>{n_series}</b> product time series in <b>{n_cats}</b> categories ({n_weeks} weeks,
            {tot_reviews:,} reviews), comparing statistical baselines, gradient boosting, and neural forecasters
            under window-level normalization.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# 5. MAIN REPORT SECTIONS
# =============================================================================
tab_ov, tab_exp, tab_abl, tab_nlp, tab_meth = st.tabs(
    ["Overview", "Forecast Explorer", "Ablation & Normalization", "NLP & Volume Effect", "Methodology"]
)

# -----------------------------------------------------------------------------
# TAB 1: OVERVIEW
# -----------------------------------------------------------------------------
with tab_ov:
    ens_df = results.get("metrics_ensemble", pd.DataFrame())
    vol_df = results.get("volume_split", pd.DataFrame())

    best_val, best_note = "-", ""
    if not ens_df.empty and best_m:
        rb = ens_df[(ens_df["model"] == best_m.get("model")) & (ens_df["feature_set"] == best_m.get("feature_set"))]
        rm = ens_df[ens_df["model"] == "MA4"]
        if not rb.empty:
            bm = float(rb["MASE"].iloc[0])
            bw = float(rb["WAPE"].iloc[0])
            best_val = f"{bm:.3f}"
            if not rm.empty:
                pct = (bm / float(rm["MASE"].iloc[0]) - 1) * 100
                best_note = f"{best_m.get('model')} {best_m.get('feature_set')} • WAPE {bw:.3f} ({pct:+.1f}% vs MA4)"
            else:
                best_note = f"{best_m.get('model')} {best_m.get('feature_set')} • WAPE {bw:.3f}"

    nlp_val, nlp_note = "-", ""
    if not vol_df.empty:
        hv = vol_df[(vol_df["Volume Group"] == "High Volume") & (vol_df["Model"] == "GRU")]
        if not hv.empty:
            r = hv.iloc[0]
            nlp_val = f"{float(r['MASE F4']):.4f}"
            dc = [c for c in vol_df.columns if "MASE %" in c]
            d_str = f"{float(r[dc[0]]):+.2f}%" if dc else ""
            p_str = f"p = {float(r['Wilcoxon p-value']):.4f}" if "Wilcoxon p-value" in r else ""
            nlp_note = f"GRU F4 (High Volume) • {d_str} ({p_str})"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Time Series", f"{n_series}", f"{n_cats} categories × {n_weeks} weeks")
    with c2:
        kpi("Review Corpus", f"{tot_reviews:,}", f"{tot_verified:,} verified purchases")
    with c3:
        kpi("Best Test MASE", best_val, best_note, style_cls="primary")
    with c4:
        kpi("High-Volume NLP", nlp_val, nlp_note, style_cls="positive")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        with st.container(border=True):
            section_heading("Test MASE by Category", "Baseline MA4 vs GRU without NLP (F2) and with full NLP (F4)")
            by_cat = results.get("mase_by_category", pd.DataFrame())
            if not by_cat.empty:
                cc = by_cat.columns[0]
                pcols = [c for c in ["MA4", "GRU F2", "GRU F4", "LightGBM F4"] if c in by_cat.columns]
                mdf = by_cat.melt(id_vars=cc, value_vars=pcols, var_name="Model", value_name="MASE")
                mdf[cc] = mdf[cc].astype(str).str.replace("_", " ")
                fig = px.bar(
                    mdf, x=cc, y="MASE", color="Model", barmode="group",
                    color_discrete_sequence=["#CBD5E1", "#4338CA", "#0D9488", "#0284C7"],
                )
                st.plotly_chart(style_fig(fig, 330, "", "MASE (lower is better)"), use_container_width=True, config=PLOT_CFG)

    with col_r:
        with st.container(border=True):
            section_heading("Ensemble Model Leaderboard", "Top 10 configurations ranked by Test MASE")
            if not ens_df.empty:
                top = ens_df.sort_values("MASE").head(10).copy()
                top["Config"] = top["model"] + " " + top["feature_set"].replace("-", "(Baseline)")
                top["Type"] = np.where(
                    top["feature_set"] == "-", "Baseline",
                    np.where(top["feature_set"].isin(["F3", "F4"]), "With NLP", "No NLP"),
                )
                top = top.iloc[::-1]
                fig = px.bar(
                    top, x="MASE", y="Config", orientation="h", color="Type",
                    text=top["MASE"].map(lambda x: f"{x:.3f}"),
                    color_discrete_map={"No NLP": "#4338CA", "With NLP": "#0D9488", "Baseline": "#D97706"},
                )
                fig.update_traces(textposition="outside", cliponaxis=False)
                fig.update_xaxes(range=[0, float(top["MASE"].max()) * 1.15])
                st.plotly_chart(style_fig(fig, 330, "Test MASE", ""), use_container_width=True, config=PLOT_CFG)

    with st.container(border=True):
        section_heading("Dataset Summary by Category", "Active post-launch weeks across 2019–2022")
        eda = (
            weekly[weekly["launched"] == True]
            .groupby("category")
            .agg(
                series=("series_id", "nunique"),
                reviews=("n_reviews", "sum"),
                verified=("n_verified", "sum"),
                mean_wk=("n_verified", "mean"),
                zero_pct=("n_verified", lambda s: (s == 0).mean() * 100),
            )
            .reset_index()
        )
        eda["category"] = eda["category"].str.replace("_", " ")
        eda.columns = ["Category", "Series", "Total Reviews", "Verified Reviews", "Mean / Week", "Zero-Review Weeks (%)"]
        st.dataframe(
            eda.style.format({"Total Reviews": "{:,.0f}", "Verified Reviews": "{:,.0f}", "Mean / Week": "{:.2f}", "Zero-Review Weeks (%)": "{:.1f}%"}),
            use_container_width=True,
            hide_index=True,
        )

# -----------------------------------------------------------------------------
# TAB 2: FORECAST EXPLORER
# -----------------------------------------------------------------------------
with tab_exp:
    with st.container(border=True):
        f1, f2, f3, f4 = st.columns([1.2, 1.0, 2.2, 0.8])
        with f1:
            sel_cat = st.selectbox("Category", ["All"] + sorted(products["category"].dropna().unique().tolist()))
        with f2:
            sel_vol = st.selectbox("Review Volume", ["All", "High Volume", "Low Volume"])

        pf = products.copy()
        if sel_cat != "All":
            pf = pf[pf["category"] == sel_cat]
        if sel_vol != "All":
            pf = pf[pf["volume_group"] == sel_vol]
        pf = pf.sort_values("total", ascending=False)

        if pf.empty:
            st.warning("No products match the selected filters.")
            st.stop()

        with f3:
            opts = pf["series_id"].tolist()
            lmap = {r["series_id"]: f"[{r['parent_asin']}] {str(r.get('title', r['series_id']))[:60]}" for _, r in pf.iterrows()}
            sel_sid = st.selectbox("Product", opts, format_func=lambda x: lmap.get(x, x))
        with f4:
            sel_h = st.selectbox("Horizon", sorted(pred["horizon"].unique().tolist()), index=0)

        m1, m2 = st.columns([1.1, 2.5])
        avail = sorted([m for m in pred["model"].unique() if m not in ("Naive", "SeasonalNaive", "MA4")])
        with m1:
            sel_model = st.selectbox("Model", avail, index=avail.index("GRU") if "GRU" in avail else 0)
        with m2:
            sel_lines = st.multiselect("Forecast Series", ["F0", "F2", "F4", "MA4"], default=["F0", "F4", "MA4"])

    pr = products[products["series_id"] == sel_sid].iloc[0]
    price_str = f"${float(pr['price']):.2f}" if pd.notna(pr.get("price")) else "N/A"
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        kpi("Product ASIN", str(pr["parent_asin"]), str(pr["category"]).replace("_", " "))
    with p2:
        kpi("Price & Rating", price_str, f"Average rating: {pr.get('average_rating', 'N/A')} / 5")
    with p3:
        kpi("Verified Demand", f"{int(pr.get('total', 0)):,}", f"Train mean: {pr['reviews_per_week']:.1f} / week")
    with p4:
        kpi("Volume Segment", pr["volume_group"], f"Split threshold: {median_vol:.1f} / week", style_cls="positive" if pr["volume_group"] == "High Volume" else "")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    gl, gr = st.columns([2.1, 1.0])
    with gl:
        with st.container(border=True):
            section_heading(f"Weekly Demand Forecast — {sel_model} (h = {sel_h})", "Dashed vertical line marks the start of the test period")
            w_s = weekly[weekly["series_id"] == sel_sid].sort_values("week")
            p_s = pred[(pred["series_id"] == sel_sid) & (pred["horizon"] == sel_h)].sort_values("week")
            val_end = pd.Timestamp(meta.get("val_end", "2022-06-27"))
            w_show = w_s[w_s["week"] >= val_end - pd.Timedelta(weeks=36)]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=w_show["week"], y=w_show["n_verified"], mode="lines+markers",
                    name="Actual", line=dict(color="#0F172A", width=2.2), marker=dict(size=4),
                    fill="tozeroy", fillcolor="rgba(15, 23, 42, 0.04)",
                )
            )
            cmap = {"F0": "#4338CA", "F2": "#0284C7", "F4": "#0D9488", "MA4": "#D97706"}
            for lk in sel_lines:
                sub = p_s[p_s["model"] == "MA4"] if lk == "MA4" else p_s[(p_s["model"] == sel_model) & (p_s["feature_set"] == lk)]
                if not sub.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=sub["week"], y=sub["y_pred"], mode="lines",
                            name="MA4" if lk == "MA4" else f"{sel_model} {lk}",
                            line=dict(color=cmap[lk], width=2, dash="dot" if lk == "MA4" else "solid"),
                        )
                    )
            fig.add_vline(x=val_end, line_dash="dash", line_color="#94A3B8")
            st.plotly_chart(style_fig(fig, 360, "", "Verified Reviews / Week"), use_container_width=True, config=PLOT_CFG)

    with gr:
        with st.container(border=True):
            section_heading("Aspect & Sentiment Profile", "Mean weekly share of reviews mentioning each aspect")
            n_s = nlp[nlp["series_id"] == sel_sid]
            acols = [c for c in n_s.columns if c.startswith("asp_")]
            if not n_s.empty and acols:
                am = (n_s[acols].mean() * 100).sort_values()
                adf = pd.DataFrame({"Aspect": [c.replace("asp_", "").capitalize() for c in am.index], "Share": am.values})
                fig_a = px.bar(
                    adf, x="Share", y="Aspect", orientation="h",
                    text=adf["Share"].map(lambda v: f"{v:.1f}%"),
                    color_discrete_sequence=["#0284C7"],
                )
                fig_a.update_traces(textposition="outside", cliponaxis=False)
                fig_a.update_xaxes(range=[0, max(float(adf["Share"].max()) * 1.25, 5)])
                st.plotly_chart(style_fig(fig_a, 275, "Share (%)", ""), use_container_width=True, config=PLOT_CFG)

                st.caption(
                    f"Mean sentiment: **{float(n_s['sent_mean'].mean()):+.2f}** • "
                    f"Positive: **{float(n_s['pos_ratio'].mean())*100:.1f}%** • "
                    f"Negative: **{float(n_s['neg_ratio'].mean())*100:.1f}%**"
                )

    with st.container(border=True):
        section_heading("Representative Customer Reviews", "Highest-helpfulness positive and negative samples scored by RoBERTa")
        r_s = reviews[reviews["series_id"] == sel_sid].copy()
        if r_s.empty:
            st.info("No review samples available for this product.")
        else:
            rf = st.radio("Sentiment filter", ["All", "Positive", "Negative"], horizontal=True, label_visibility="collapsed")
            if rf == "Positive":
                r_s = r_s[r_s["label"] == "pos"]
            elif rf == "Negative":
                r_s = r_s[r_s["label"] == "neg"]

            arcols = [c for c in r_s.columns if c.startswith("asp_")]
            rc1, rc2 = st.columns(2)
            for i_r, (_, rr) in enumerate(r_s.head(8).iterrows()):
                lbl = str(rr.get("label", "pos"))
                tag = f"{'Positive' if lbl == 'pos' else 'Negative'} ({float(rr.get('sent', 0)):+.2f})"
                rating_n = int(round(float(rr.get("rating", 5))))
                chips = "".join(f'<span class="asp-pill">{c.replace("asp_", "")}</span>' for c in arcols if rr.get(c, 0) == 1)
                html = f"""
                <div class="rev-card {lbl}">
                    <div class="rev-meta">
                        <span class="rev-tag {lbl}">{tag}</span>
                        <span>{rating_n}/5 • {int(rr.get('helpful_vote', 0))} helpful</span>
                    </div>
                    <div class="rev-title">{str(rr.get('title', ''))[:120]}</div>
                    <div class="rev-body">{str(rr.get('text', ''))[:320]}</div>
                    <div>{chips}</div>
                </div>
                """
                (rc1 if i_r % 2 == 0 else rc2).markdown(html, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 3: ABLATION & NORMALIZATION
# -----------------------------------------------------------------------------
with tab_abl:
    abl = results.get("ablation_mase", pd.DataFrame())
    if not abl.empty:
        with st.container(border=True):
            section_heading("Feature Ablation Study (F0 → F4)", "Test MASE across progressively enriched feature sets")
            mc = abl.columns[0]
            fsc = [c for c in ["F0", "F1", "F2", "F3", "F4"] if c in abl.columns]
            am = abl.melt(id_vars=mc, value_vars=fsc, var_name="Feature Set", value_name="MASE")
            fig = px.bar(
                am, x=mc, y="MASE", color="Feature Set", barmode="group",
                color_discrete_sequence=["#4338CA", "#6366F1", "#0284C7", "#14B8A6", "#0D9488"],
            )
            ens_df = results.get("metrics_ensemble", pd.DataFrame())
            if not ens_df.empty and "MA4" in ens_df["model"].values:
                m4 = float(ens_df.loc[ens_df["model"] == "MA4", "MASE"].iloc[0])
                fig.add_hline(y=m4, line_dash="dash", line_color="#D97706", annotation_text=f"MA4 ({m4:.3f})")
            fig.update_yaxes(range=[float(am["MASE"].min()) * 0.92, float(am["MASE"].max()) * 1.06])
            st.plotly_chart(style_fig(fig, 360, "", "Test MASE"), use_container_width=True, config=PLOT_CFG)

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            section_heading("Normalization Impact (v1 Global → v2 Window)", "Relative MASE reduction after adopting 8-week local window scaling")
            v12 = results.get("v1_vs_v2", pd.DataFrame())
            if not v12.empty:
                st.dataframe(v12, use_container_width=True, hide_index=True, height=340)
    with c2:
        with st.container(border=True):
            section_heading("Wilcoxon Signed-Rank Tests", "Paired per-series MAE comparisons across 150 products")
            sig = results.get("significance", pd.DataFrame())
            if not sig.empty:
                st.dataframe(sig, use_container_width=True, hide_index=True, height=340)

    comp = results.get("model_comparison", pd.DataFrame())
    if not comp.empty:
        with st.container(border=True):
            h1, h2, h3 = st.columns([2.8, 0.8, 0.8])
            with h1:
                section_heading("Complete Model Benchmark", "Test metrics averaged across 3 random seeds (mean ± std)")
            with h2:
                st.download_button(
                    "Export CSV",
                    data=comp.to_csv(index=False).encode("utf-8"),
                    file_name="model_comparison.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with h3:
                st.download_button(
                    "Export LaTeX",
                    data=df_to_latex(comp, "Model comparison on the test split (mean ± std across 3 seeds).").encode("utf-8"),
                    file_name="model_comparison.tex",
                    mime="text/plain",
                    use_container_width=True,
                )
            st.dataframe(comp, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# TAB 4: NLP & VOLUME EFFECT
# -----------------------------------------------------------------------------
with tab_nlp:
    vol_df = results.get("volume_split", pd.DataFrame())
    nl, nr = st.columns([1.4, 1.0])

    with nl:
        with st.container(border=True):
            section_heading("NLP Effect Stratified by Review Volume", "Relative MASE change from F2 to F4 (negative values indicate improvement)")
            if not vol_df.empty:
                dcols = [c for c in vol_df.columns if "MASE %" in c]
                if dcols:
                    fig = px.bar(
                        vol_df, x="Model", y=dcols[0], color="Volume Group", barmode="group",
                        color_discrete_sequence=["#0D9488", "#BE123C"],
                    )
                    fig.add_hline(y=0, line_color="#64748B", line_width=1)
                    st.plotly_chart(style_fig(fig, 340, "", "MASE Δ F2 → F4 (%)"), use_container_width=True, config=PLOT_CFG)

    with nr:
        with st.container(border=True):
            section_heading("Global Aspect Prevalence", "Average weekly share of zero-shot aspect mentions")
            acols = [c for c in nlp.columns if c.startswith("asp_")]
            if acols:
                gm = (nlp[acols].mean() * 100).sort_values()
                gdf = pd.DataFrame({"Aspect": [c.replace("asp_", "").capitalize() for c in gm.index], "Share": gm.values})
                fig = px.bar(
                    gdf, x="Share", y="Aspect", orientation="h",
                    text=gdf["Share"].map(lambda v: f"{v:.1f}%"),
                    color_discrete_sequence=["#4338CA"],
                )
                fig.update_traces(textposition="outside", cliponaxis=False)
                fig.update_xaxes(range=[0, float(gdf["Share"].max()) * 1.25])
                st.plotly_chart(style_fig(fig, 340, "Mean Weekly Share (%)", ""), use_container_width=True, config=PLOT_CFG)

    if not vol_df.empty:
        with st.container(border=True):
            h1, h2 = st.columns([3.5, 0.9])
            with h1:
                section_heading("Volume-Stratified Evaluation Table", f"Split at train median ({median_vol:.1f} verified reviews/week)")
            with h2:
                st.download_button(
                    "Export LaTeX",
                    data=df_to_latex(vol_df, "Forecasting performance stratified by review volume.").encode("utf-8"),
                    file_name="volume_split.tex",
                    mime="text/plain",
                    use_container_width=True,
                )
            st.dataframe(vol_df, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# TAB 5: METHODOLOGY
# -----------------------------------------------------------------------------
with tab_meth:
    m1, m2 = st.columns(2)
    with m1:
        with st.container(border=True):
            section_heading("1. Pipeline Architecture", "Leakage-free temporal design from raw reviews to multi-horizon forecasts")
            st.markdown(
                """
                - **Data Curation:** Selected 150 active product series across 5 Amazon Reviews 2023 categories (2019–2022, 208 weekly steps). Weekly verified purchase review count serves as the demand proxy.
                - **Temporal NLP Extraction:** Scored every review with `cardiffnlp/twitter-roberta-base-sentiment-latest` and tagged 7 domain-agnostic aspects using `all-MiniLM-L6-v2` cosine similarity (threshold 0.35). Missing weeks are forward-filled strictly from past values.
                - **Window Normalization (v2):** Targets are log-transformed (`log1p`) and centered using the trailing 8-week local mean within each 26-week input window, eliminating post-COVID level shift.
                - **Global Forecasters:** Evaluated statistical baselines (`Naive`, `SeasonalNaive`, `MA4`), tabular ML (`Ridge`, `LightGBM`), and sequence neural models (`LSTM`, `GRU`, `Transformer`) across 3 random seeds.
                """
            )
    with m2:
        with st.container(border=True):
            section_heading("2. Evaluation Protocol & Metrics", "Chronological split and scale-independent accuracy metrics")
            st.markdown(
                f"""
                - **Chronological Split:** Train ≤ `{meta.get('train_end', '2021-12-27')}`, Validation ≤ `{meta.get('val_end', '2022-06-27')}`, Test > `{meta.get('val_end', '2022-06-27')}` (horizon `H = {meta.get('H', 4)}` weeks).
                - **Primary Metric (MASE):** Mean Absolute Scaled Error normalized by each series' in-sample naive 1-step error, enabling fair aggregation across heterogeneous product scales.
                - **Secondary Metrics:** Weighted Absolute Percentage Error (WAPE), MAE, and RMSE on the original count scale.
                - **Statistical Testing:** Paired Wilcoxon signed-rank tests over per-series MAE and Spearman correlation across review volume density.
                """
            )

    with st.container(border=True):
        section_heading("3. Dataset Attribution", "Primary public benchmark source")
        st.code(
            "@article{hou2024bridging,\n"
            "  title={Bridging Language and Items for Retrieval and Recommendation},\n"
            "  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},\n"
            "  journal={arXiv preprint arXiv:2403.03952},\n"
            "  year={2024}\n"
            "}",
            language="bibtex",
        )
