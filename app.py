
"""Customer Demand Forecasting from Online Product Reviews.

Figma-inspired Streamlit analytics dashboard.

Pages:
1. Demand Forecasting
2. Reviews & Sentiment
3. Model Results
4. About

Run:
    streamlit run app.py
"""

import json
import os
import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA = Path(
    os.environ.get(
        "APP_DATA",
        Path(__file__).parent / "app_data"
    )
)

st.set_page_config(
    page_title="DemandLens | AI Demand Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Design tokens
PRIMARY = "#635BFF"
SECONDARY = "#8B85FF"
DARK = "#19233B"
MUTED = "#77839A"
GREEN = "#10B981"
RED = "#F16D86"
ORANGE = "#F4A65B"
GREY = "#94A3B8"

ASPECT_NAMES = {
    "asp_quality": "Quality",
    "asp_durability": "Durability",
    "asp_value": "Value for money",
    "asp_shipping": "Shipping",
    "asp_usability": "Ease of use",
    "asp_compatibility": "Compatibility",
    "asp_gift": "Gift"
}


# ============================================================
# 2. FIGMA-INSPIRED DESIGN SYSTEM
# ============================================================

st.markdown("""
<style>
@import url(
    'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap'
);

:root {
    --primary: #635BFF;
    --primary-light: #EEEDFF;
    --background: #F7F8FC;
    --surface: #FFFFFF;
    --text: #19233B;
    --muted: #77839A;
    --border: #E6EAF2;
    --success: #10B981;
}

/* Global */

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: var(--background);
    color: var(--text);
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 4rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
}

h1, h2, h3, h4 {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: var(--text);
    letter-spacing: -0.035em;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: 9px 11px;
    border-radius: 10px;
    margin-bottom: 4px;
    transition: background 0.2s ease;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: #F4F3FF;
}

[data-testid="stSidebar"] [role="radiogroup"]
label:has(input:checked) {
    background: var(--primary-light);
    color: #514BD3;
}

/* Brand */

.brand-box {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 6px 0 22px;
}

.brand-icon {
    width: 44px;
    height: 44px;
    border-radius: 13px;
    background: linear-gradient(
        135deg, #635BFF, #9E85FF
    );
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 25px;
    color: white;
}

.brand-name {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 800;
    font-size: 22px;
    color: #19233B;
    letter-spacing: -0.05em;
}

.brand-caption {
    font-size: 10px;
    font-weight: 800;
    color: #91A0B5;
    letter-spacing: 1.4px;
}

/* Page header */

.page-eyebrow {
    color: #635BFF;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.page-title {
    color: #19233B;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 33px;
    font-weight: 800;
    letter-spacing: -1.3px;
    margin-bottom: 7px;
    line-height: 1.3;
}

.page-subtitle {
    color: #77839A;
    font-size: 14px;
    margin-bottom: 28px;
}

/* Product panel */

.product-panel {
    background: linear-gradient(
        110deg, #FFFFFF, #F7F6FF
    );
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 22px 25px;
    margin-bottom: 25px;
}

.product-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 18px;
    font-weight: 800;
    color: #19233B;
    margin-bottom: 8px;
}

.product-details {
    font-size: 13px;
    color: #77839A;
}

/* KPI metric cards */

[data-testid="stMetric"] {
    background: white;
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 22px;
    min-height: 125px;
    box-shadow: 0 6px 25px rgba(30, 43, 76, 0.04);
    transition: transform 0.2s ease,
                box-shadow 0.2s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(30, 43, 76, 0.08);
}

[data-testid="stMetricLabel"] {
    font-size: 13px;
    color: #77839A;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #19233B;
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -0.5px;
}

/* Plotly containers */

[data-testid="stPlotlyChart"] {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 12px;
    box-shadow: 0 6px 24px rgba(30, 43, 76, 0.035);
    overflow: hidden;
}

/* Form widgets */

[data-baseweb="select"] > div {
    border-radius: 11px;
    background: white;
}

[data-baseweb="input"] > div {
    border-radius: 11px;
    background: white;
}

.stButton > button {
    border-radius: 11px;
    font-weight: 700;
    transition: 0.2s;
}

.stButton > button[kind="primary"] {
    background: #635BFF;
    color: white;
    border: 0;
}

.stButton > button[kind="primary"]:hover {
    background: #514BD3;
}

/* Tables */

[data-testid="stDataFrame"] {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
}

/* Expanders */

[data-testid="stExpander"] {
    background: white;
    border: 1px solid var(--border);
    border-radius: 14px;
}

/* Tabs */

.stTabs [data-baseweb="tab-list"] {
    background: white;
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 5px;
    gap: 6px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    font-weight: 700;
}

.stTabs [aria-selected="true"] {
    background: #EEEDFF;
    color: #635BFF;
}

/* Section headings */

.section-heading {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 19px;
    font-weight: 800;
    color: #19233B;
    margin-top: 25px;
    margin-bottom: 14px;
}

/* Pipeline cards */

.pipeline-card {
    background: white;
    border: 1px solid #E6EAF2;
    border-radius: 16px;
    padding: 24px;
    min-height: 195px;
    margin-bottom: 12px;
}

.pipeline-icon {
    font-size: 29px;
    margin-bottom: 12px;
}

.pipeline-title {
    font-weight: 800;
    font-size: 16px;
    color: #19233B;
    margin-bottom: 8px;
}

.pipeline-description {
    color: #77839A;
    font-size: 13px;
    line-height: 1.7;
}

/* Captions */

[data-testid="stCaptionContainer"] {
    color: #77839A;
}

/* Mobile */

@media (max-width: 950px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 1rem;
    }

    .page-title {
        font-size: 25px;
    }

    [data-testid="stMetric"] {
        padding: 15px;
        min-height: 95px;
    }

    [data-testid="stMetricValue"] {
        font-size: 20px;
    }

    .product-panel {
        padding: 17px;
    }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. SHARED UI COMPONENTS
# ============================================================

def page_header(eyebrow, title, subtitle):
    st.markdown(
        f"""
        <div class="page-eyebrow">
            {html.escape(eyebrow)}
        </div>
        <div class="page-title">
            {html.escape(title)}
        </div>
        <div class="page-subtitle">
            {html.escape(subtitle)}
        </div>
        """,
        unsafe_allow_html=True
    )


def section_title(title):
    st.markdown(
        f'<div class="section-heading">'
        f'{html.escape(title)}</div>',
        unsafe_allow_html=True
    )


def style_chart(fig, height=420):
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="DM Sans",
            size=12,
            color="#64748B"
        ),
        margin=dict(
            l=30,
            r=25,
            t=35,
            b=45
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            font=dict(color="#19233B")
        ),
        legend=dict(
            orientation="h",
            y=-0.22,
            x=0
        ),
        xaxis=dict(
            showgrid=False,
            linecolor="#E6EAF2"
        ),
        yaxis=dict(
            gridcolor="#F0F2F7",
            zeroline=False
        )
    )

    return fig


def pipeline_card(icon, title, description):
    st.markdown(
        f"""
        <div class="pipeline-card">
            <div class="pipeline-icon">{icon}</div>
            <div class="pipeline-title">
                {html.escape(title)}
            </div>
            <div class="pipeline-description">
                {html.escape(description)}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 4. DATA LOADING
# ============================================================

@st.cache_data
def load(data_dir: str):
    data_path = Path(data_dir)

    required = [
        "predictions.parquet",
        "weekly_demand.parquet",
        "weekly_nlp_features.parquet",
        "products.csv"
    ]

    missing = [
        name
        for name in required
        if not (data_path / name).exists()
    ]

    if missing:
        return None, missing

    d = {
        "pred": pd.read_parquet(
            data_path / "predictions.parquet"
        ),
        "weekly": pd.read_parquet(
            data_path / "weekly_demand.parquet"
        ),
        "nlp": pd.read_parquet(
            data_path / "weekly_nlp_features.parquet"
        ),
        "products": pd.read_csv(
            data_path / "products.csv"
        ),
        "reviews": (
            pd.read_parquet(
                data_path / "review_samples.parquet"
            )
            if (data_path / "review_samples.parquet").exists()
            else pd.DataFrame()
        ),
        "meta": (
            json.loads(
                (data_path / "meta.json").read_text(
                    encoding="utf-8"
                )
            )
            if (data_path / "meta.json").exists()
            else {}
        )
    }

    for key in ("pred", "weekly", "nlp"):
        for column in ("week", "origin_week"):
            if column in d[key].columns:
                d[key][column] = pd.to_datetime(
                    d[key][column]
                )

    d["pred"]["label"] = np.where(
        d["pred"]["feature_set"] == "-",
        d["pred"]["model"],
        d["pred"]["model"]
        + " · "
        + d["pred"]["feature_set"]
    )

    d["products"]["title"] = (
        d["products"]["title"].fillna(
            d["products"]["series_id"]
        )
    )

    return d, []


@st.cache_data
def result_table(name, data_dir=str(DATA), **kwargs):
    path = Path(data_dir) / "results" / name

    if path.exists():
        return pd.read_csv(path, **kwargs)

    return None


D, missing = load(str(DATA))

if D is None:
    page_header(
        "Setup required",
        "Dataset not found",
        "The dashboard needs the exported application data."
    )

    st.error(
        f"Missing files in `{DATA}`: "
        + ", ".join(missing)
    )

    st.info(
        "Run notebook 4 and extract app_data.zip "
        "into the app_data/ folder beside app.py."
    )

    st.stop()


pred = D["pred"]
weekly = D["weekly"]
nlp = D["nlp"]
products = D["products"]
reviews = D["reviews"]
meta = D["meta"]

test_start = pred["week"].min()

best = meta.get("best_model", {})

default_main = (
    f"{best.get('model', 'GRU')} · "
    f"{best.get('feature_set', 'F0')}"
)

labels = sorted(
    pred["label"].unique(),
    key=lambda value: (
        value.count("·") == 0,
        value
    )
)


# ============================================================
# 5. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown(
    """
    <div class="brand-box">
        <div class="brand-icon">◈</div>
        <div>
            <div class="brand-name">DemandLens</div>
            <div class="brand-caption">
                AI DEMAND INTELLIGENCE
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.caption("WORKSPACE")

page = st.sidebar.radio(
    "Navigation",
    [
        "Forecast",
        "Reviews & sentiment",
        "Model results",
        "About"
    ],
    label_visibility="collapsed"
)

st.sidebar.divider()

st.sidebar.caption("PRODUCT FILTERS")

cats = sorted(
    products["category"].dropna().unique()
)

cat = st.sidebar.selectbox(
    "Category",
    cats
)

prods = (
    products[products["category"] == cat]
    .sort_values("title")
)

product_lookup = prods.set_index("series_id")[
    "title"
].to_dict()

pick = st.sidebar.selectbox(
    "Product",
    prods["series_id"],
    format_func=lambda sid: (
        str(product_lookup.get(sid, sid))[:75]
    )
)

prod = products.set_index("series_id").loc[pick]

st.sidebar.divider()

st.sidebar.caption(
    "Amazon Reviews 2023 · NLP-powered demand analytics"
)


# ============================================================
# 6. PRODUCT HELPERS
# ============================================================

def product_header():
    details = [
        f"Category: {cat.replace('_', ' ')}"
    ]

    if pd.notna(prod.get("price")):
        details.append(
            f"Price: \${prod['price']}"
        )

    if pd.notna(prod.get("average_rating")):
        rating_count = prod.get("rating_number", 0)

        if pd.isna(rating_count):
            rating_count = 0

        details.append(
            f"Rating: {prod['average_rating']} ★ "
            f"({int(rating_count):,} ratings)"
        )

    safe_title = html.escape(str(prod["title"]))
    safe_details = html.escape("  •  ".join(details))

    st.markdown(
        f"""
        <div class="product-panel">
            <div class="product-title">
                {safe_title}
            </div>
            <div class="product-details">
                {safe_details}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def series(sid):
    w = weekly[
        weekly["series_id"] == sid
    ].sort_values("week")

    if "launched" in w.columns:
        return w[w["launched"]]

    return w


# ============================================================
# 7. PAGE: DEMAND FORECASTING
# ============================================================

if page == "Forecast":

    page_header(
        "Forecasting workspace",
        "Demand Forecasting",
        "Explore historical demand, compare AI models, "
        "and evaluate prediction accuracy."
    )

    product_header()

    w = series(pick)

    pp = pred[
        pred["series_id"] == pick
    ].copy()

    pp["abs_err"] = (
        pp["y_pred"] - pp["y_true"]
    ).abs()

    tbl = (
        pp.groupby("label")
        .agg(
            MAE=("abs_err", "mean"),
            actual=("y_true", "sum"),
            err=("abs_err", "sum")
        )
        .assign(
            WAPE=lambda t: (
                t["err"]
                / t["actual"].clip(lower=1e-9)
            )
        )[["MAE", "WAPE"]]
        .sort_values("MAE")
    )

    recent = (
        w[w["week"] < test_start]
        .tail(8)["n_verified"]
        .mean()
    )

    if tbl.empty:
        st.warning(
            "No forecast results are available "
            "for this product."
        )
        st.stop()

    best_model = tbl.index[0]

    best_mae = tbl.iloc[0]["MAE"]

    test_weeks = pp["week"].nunique()

    # KPI section
    section_title("Performance Overview")

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.metric(
            "Average Weekly Demand",
            (
                f"{recent:.1f}"
                if pd.notna(recent)
                else "N/A"
            ),
            help="Average of the last 8 weeks before test"
        )

    with k2:
        st.metric(
            "Best Model MAE",
            f"{best_mae:.2f}"
        )

    with k3:
        st.metric(
            "Best Performing Model",
            best_model
        )

    with k4:
        st.metric(
            "Test Weeks",
            f"{test_weeks}"
        )

    st.write("")

    # Forecast visualization
    section_title("Demand Prediction Analysis")

    chart_col, filter_col = st.columns(
        [3, 1],
        gap="large"
    )

    with filter_col:
        st.markdown("##### Forecast Settings")

        defaults = [
            model
            for model in [
                default_main,
                "GRU · F4",
                "MA4"
            ]
            if model in labels
        ]

        if not defaults and labels:
            defaults = [labels[0]]

        choose = st.multiselect(
            "Forecast Models",
            labels,
            default=list(dict.fromkeys(defaults))
        )

        h = st.radio(
            "Forecast Horizon",
            [1, 2, 3, 4],
            horizontal=True,
            format_func=lambda x: f"{x}W"
        )

        hist_weeks = st.slider(
            "History Shown (Weeks)",
            min_value=26,
            max_value=208,
            value=78,
            step=13
        )

        st.caption(
            "Select models and forecast horizon "
            "to update the visualization."
        )

    with chart_col:

        p = pred[
            (pred["series_id"] == pick)
            & (pred["horizon"] == h)
        ]

        start = (
            test_start
            - pd.Timedelta(weeks=hist_weeks)
        )

        fig = go.Figure()

        if not w.empty:
            fig.add_vrect(
                x0=test_start,
                x1=w["week"].max(),
                fillcolor="#F0EFFF",
                opacity=0.7,
                line_width=0,
                layer="below",
                annotation_text="Test period",
                annotation_position="top left"
            )

        visible_history = w[
            w["week"] >= start
        ]

        fig.add_trace(
            go.Scatter(
                x=visible_history["week"],
                y=visible_history["n_verified"],
                mode="lines",
                name="Actual Demand",
                line=dict(
                    color=DARK,
                    width=3
                ),
                hovertemplate=(
                    "%{x|%d %b %Y}"
                    "<br>Actual: %{y:.2f}"
                    "<extra></extra>"
                )
            )
        )

        palette = [
            PRIMARY,
            GREEN,
            RED,
            ORANGE,
            "#9B72E8",
            "#16A3CE",
            GREY,
            "#CC7894"
        ]

        for i, lab in enumerate(choose):

            q = p[
                p["label"] == lab
            ].sort_values("week")

            fig.add_trace(
                go.Scatter(
                    x=q["week"],
                    y=q["y_pred"],
                    mode="lines",
                    name=lab,
                    line=dict(
                        color=palette[i % len(palette)],
                        width=2.4,
                        dash=(
                            "dot"
                            if "·" not in lab
                            else "solid"
                        )
                    )
                )
            )

        fig = style_chart(fig, height=450)

        fig.update_layout(
            yaxis_title="Verified Reviews / Week"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    section_title("Model Error Comparison")

    with st.expander(
        "View detailed model performance",
        expanded=True
    ):
        st.dataframe(
            tbl.style.format({
                "MAE": "{:.2f}",
                "WAPE": "{:.1%}"
            }),
            use_container_width=True
        )

        st.download_button(
            "Download Metrics CSV",
            data=tbl.to_csv().encode("utf-8"),
            file_name="forecast_metrics.csv",
            mime="text/csv"
        )

    st.caption(
        "Demand proxy = verified-purchase reviews per week. "
        "Models forecast the next 4 weeks using the previous "
        "26 weeks. MA4 represents the 4-week moving average."
    )


# ============================================================
# 8. PAGE: REVIEWS & SENTIMENT
# ============================================================

elif page == "Reviews & sentiment":

    page_header(
        "Customer intelligence",
        "Reviews & Sentiment",
        "Discover customer opinions, sentiment trends, "
        "and key product attributes."
    )

    product_header()

    w = series(pick)

    f = (
        nlp[nlp["series_id"] == pick]
        .merge(
            w[["week", "n_verified"]],
            on="week"
        )
        .sort_values("week")
    )

    if f.empty:
        st.warning(
            "No NLP data is available for this product."
        )
        st.stop()

    # Sentiment summary
    section_title("Sentiment Overview")

    sentiment = f["sent_mean"].mean()
    negative = f["neg_ratio"].mean()
    avg_demand = f["n_verified"].mean()

    k1, k2, k3 = st.columns(3)

    with k1:
        st.metric(
            "Average Sentiment",
            f"{sentiment:+.3f}"
        )

    with k2:
        st.metric(
            "Negative Review Ratio",
            f"{negative:.1%}"
        )

    with k3:
        st.metric(
            "Average Weekly Demand",
            f"{avg_demand:.1f}"
        )

    st.write("")

    section_title("Sentiment & Demand Trends")

    smooth = st.slider(
        "Rolling Average (Weeks)",
        min_value=1,
        max_value=13,
        value=4
    )

    r = (
        f.set_index("week")
        .rolling(
            smooth,
            min_periods=1
        )
        .mean(numeric_only=True)
        .reset_index()
    )

    fig = make_subplots(
        specs=[[{"secondary_y": True}]]
    )

    fig.add_trace(
        go.Scatter(
            x=r["week"],
            y=r["n_verified"],
            name="Verified Demand",
            line=dict(
                color=DARK,
                width=3
            )
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=r["week"],
            y=r["sent_mean"],
            name="Mean Sentiment",
            line=dict(
                color=PRIMARY,
                width=2.5
            )
        ),
        secondary_y=True
    )

    fig.add_trace(
        go.Scatter(
            x=r["week"],
            y=r["neg_ratio"],
            name="Negative Review Share",
            line=dict(
                color=RED,
                width=2.5
            )
        ),
        secondary_y=True
    )

    fig.add_vline(
        x=test_start,
        line_dash="dot",
        line_color=GREY
    )

    fig = style_chart(fig, height=440)

    fig.update_yaxes(
        title_text="Demand",
        secondary_y=False
    )

    fig.update_yaxes(
        title_text="Sentiment / Share",
        secondary_y=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Aspect analysis
    asp_cols = [
        c
        for c in f.columns
        if c.startswith("asp_")
    ]

    if asp_cols:

        section_title("What Customers Talk About")

        chosen = st.multiselect(
            "Review Aspects",
            asp_cols,
            default=[
                c
                for c in [
                    "asp_quality",
                    "asp_durability",
                    "asp_gift"
                ]
                if c in asp_cols
            ],
            format_func=lambda c: ASPECT_NAMES.get(c, c)
        )

        fig2 = go.Figure()

        aspect_palette = [
            PRIMARY,
            GREEN,
            ORANGE,
            RED,
            "#9B72E8",
            "#16A3CE",
            GREY
        ]

        for i, aspect in enumerate(chosen):

            fig2.add_trace(
                go.Scatter(
                    x=r["week"],
                    y=r[aspect],
                    name=ASPECT_NAMES.get(
                        aspect,
                        aspect
                    ),
                    mode="lines",
                    line=dict(
                        color=aspect_palette[
                            i % len(aspect_palette)
                        ],
                        width=2.5
                    )
                )
            )

        fig2 = style_chart(fig2, height=350)

        fig2.update_layout(
            yaxis_title="Share of Reviews",
            yaxis_tickformat=".0%"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        overall = (
            f[asp_cols]
            .mean()
            .rename(index=ASPECT_NAMES)
            .sort_values(ascending=False)
        )

        with st.expander(
            "Overall Review Aspect Distribution"
        ):
            st.dataframe(
                overall.to_frame(
                    "Share of Reviews"
                ).style.format({
                    "Share of Reviews": "{:.1%}"
                }),
                use_container_width=True
            )

    # Review samples
    if not reviews.empty:

        section_title("Most Helpful Customer Reviews")

        rv = reviews[
            reviews["series_id"] == pick
        ]

        col_pos, col_neg = st.columns(
            2,
            gap="large"
        )

        review_groups = [
            (col_pos, "pos", "Positive Reviews"),
            (col_neg, "neg", "Negative Reviews")
        ]

        for col, lab, heading in review_groups:

            with col:

                st.markdown(f"##### {heading}")

                sub = rv[
                    rv["label"] == lab
                ]

                if sub.empty:
                    st.info(
                        "No reviews in this category."
                    )

                for _, x in sub.iterrows():

                    date = pd.to_datetime(
                        x["date"]
                    ).date()

                    stars = "★" * int(x["rating"])

                    review_title = str(
                        x["title"]
                    )[:60]

                    with st.expander(
                        f"{stars} {review_title} — {date}"
                    ):

                        st.write(x["text"])

                        st.caption(
                            f"Sentiment: {x['sent']:+.2f}"
                            f"  •  "
                            f"{int(x['helpful_vote'])} "
                            "helpful votes"
                        )


# ============================================================
# 9. PAGE: MODEL RESULTS
# ============================================================

elif page == "Model results":

    page_header(
        "Machine learning evaluation",
        "Model Performance",
        "Compare models, evaluate feature contributions, "
        "and analyze prediction errors."
    )

    section_title("Experiment Overview")

    k1, k2, k3 = st.columns(3)

    with k1:
        st.metric(
            "Test Period",
            "Jul–Dec 2022"
        )

    with k2:
        st.metric(
            "Feature Sets",
            "F0 – F4"
        )

    with k3:
        st.metric(
            "Evaluation Metric",
            "MASE"
        )

    st.caption(
        "MASE measures prediction error relative to "
        "a naive forecast. Lower values are better. "
        "Deep learning results average three seeds."
    )

    ens = result_table(
        "metrics_ensemble.csv"
    )

    abl = result_table(
        "ablation_mase.csv",
        index_col=0
    )

    # Ablation chart
    if abl is not None:

        section_title("Feature Ablation Analysis")

        sets = [
            c
            for c in ["F0", "F1", "F2", "F3", "F4"]
            if c in abl.columns
        ]

        if sets:

            fig = go.Figure()

            shades = [
                PRIMARY,
                "#938AFF",
                GREEN,
                ORANGE,
                RED
            ]

            for i, feature in enumerate(sets):

                fig.add_trace(
                    go.Bar(
                        x=abl.index,
                        y=abl[feature],
                        name=feature,
                        marker_color=shades[i]
                    )
                )

            if (
                ens is not None
                and "model" in ens.columns
                and "MASE" in ens.columns
                and (ens["model"] == "MA4").any()
            ):

                ma4 = ens.loc[
                    ens["model"] == "MA4",
                    "MASE"
                ].iloc[0]

                fig.add_hline(
                    y=ma4,
                    line_dash="dash",
                    line_color=GREY,
                    annotation_text=f"MA4 {ma4:.3f}"
                )

            fig = style_chart(
                fig,
                height=430
            )

            fig.update_layout(
                barmode="group",
                yaxis_title="MASE (Test)"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.caption(
                "F0: Demand only  •  "
                "F1: Calendar  •  "
                "F2: Review statistics  •  "
                "F3: Sentiment  •  "
                "F4: Review aspects"
            )

    # Result tables
    section_title("Detailed Experiment Results")

    tabs = st.tabs([
        "Model Comparison",
        "Significance",
        "By Category",
        "Review Volume",
        "v1 → v2"
    ])

    with tabs[0]:

        t = result_table(
            "model_comparison.csv"
        )

        if t is not None:
            st.dataframe(
                t,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No comparison file available.")

    with tabs[1]:

        t = result_table(
            "significance.csv"
        )

        if t is not None:

            float_columns = {
                c: "{:.4f}"
                for c in t.columns
                if t[c].dtype.kind == "f"
            }

            st.dataframe(
                t.style.format(float_columns),
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Paired Wilcoxon test on per-product MAE. "
                "p < 0.05 indicates statistical significance."
            )

        else:
            st.info("No significance file available.")

    with tabs[2]:

        t = result_table(
            "mase_by_category.csv",
            index_col=0
        )

        if t is not None:

            st.dataframe(
                t.round(3),
                use_container_width=True
            )

        else:
            st.info("No category results available.")

    with tabs[3]:

        t = result_table(
            "volume_split.csv"
        )

        if t is not None:

            st.dataframe(
                t.round(4),
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Products are split by median verified "
                "review volume during the training period."
            )

        else:
            st.info(
                "Run notebook 4 to generate these results."
            )

    with tabs[4]:

        t = result_table(
            "v1_vs_v2.csv",
            header=[0, 1]
        )

        if t is not None:

            t.columns = [
                " ".join(
                    str(c)
                    for c in col
                    if not str(c).startswith("Unnamed")
                ).strip()
                for col in t.columns
            ]

            st.dataframe(
                t.round(3),
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "v1 uses training-mean scaling. "
                "v2 normalizes each input window "
                "by its recent demand level. "
                "Negative Δ indicates v2 is better."
            )

        else:
            st.info("No v1 comparison results found.")


# ============================================================
# 10. PAGE: ABOUT
# ============================================================

else:

    page_header(
        "Project overview",
        "About DemandLens",
        "AI-powered customer demand forecasting "
        "from online product reviews."
    )

    section_title("Research Question")

    st.markdown(
        """
        Can customer reviews and NLP-derived signals
        improve the accuracy of weekly product
        demand forecasting?
        """
    )

    section_title("Data Overview")

    first_week = weekly["week"].min().date()
    last_week = weekly["week"].max().date()

    total_products = products[
        "series_id"
    ].nunique()

    total_categories = products[
        "category"
    ].nunique()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Products",
            f"{total_products:,}"
        )

    with c2:
        st.metric(
            "Categories",
            str(total_categories)
        )

    with c3:
        st.metric(
            "Data Source",
            "Amazon 2023"
        )

    st.caption(
        f"Weekly demand series: {first_week} → "
        f"{last_week}. Demand is approximated by "
        "the number of verified-purchase reviews."
    )

    section_title("AI Research Pipeline")

    p1, p2 = st.columns(2, gap="large")

    with p1:

        pipeline_card(
            "📦",
            "01. Data Preparation",
            "Select products with dense review histories "
            "and construct weekly demand time series."
        )

    with p2:

        pipeline_card(
            "🧠",
            "02. NLP Feature Extraction",
            "Extract RoBERTa sentiment and seven "
            "product aspects using sentence embeddings."
        )

    p3, p4 = st.columns(2, gap="large")

    with p3:

        pipeline_card(
            "📈",
            "03. Demand Forecasting",
            "Forecast four weeks ahead using 26 weeks "
            "of history with Ridge, LightGBM, "
            "LSTM, GRU and Transformer models."
        )

    with p4:

        pipeline_card(
            "📊",
            "04. Model Evaluation",
            "Use time-based train, validation and "
            "test splits with MASE, WAPE, "
            "ablation studies and Wilcoxon tests."
        )

    section_title("Model Architecture")

    st.markdown(
        """
        The forecasting pipeline includes:

        - **Baselines:** Naive, Seasonal Naive, MA4
        - **Machine Learning:** Ridge, LightGBM
        - **Deep Learning:** LSTM, GRU, Transformer
        - **NLP:** RoBERTa sentiment and seven aspects
        - **Feature Groups:** F0, F1, F2, F3, F4
        - **Normalization:** Per-window normalization (v2)
        """
    )

    section_title("Evaluation Strategy")

    train_end = meta.get(
        "train_end",
        "2021-12"
    )

    val_end = meta.get(
        "val_end",
        "2022-06"
    )

    st.markdown(
        f"""
        **Training:** through {train_end}

        **Validation:** through {val_end}

        **Testing:** Jul–Dec 2022

        **Evaluation:** MASE, WAPE,
        MAE and paired Wilcoxon testing.
        """
    )

    section_title("Research Findings")

    st.markdown(
        """
        **Key findings from the project:**

        - All evaluated models outperform MA4
          after per-window normalization.
        - Historical demand alone provides
          the best average forecasting accuracy.
        - NLP features can reduce large errors.
        - Review-rich product categories benefit
          more from sentiment and aspect signals.
        - NLP does not improve average scaled
          error across all products.
        """
    )

    st.divider()

    st.caption(
        "DemandLens · Customer Demand Forecasting "
        "from Online Product Reviews · "
        "Powered by Python, Streamlit, "
        "Plotly and NLP"
    )
