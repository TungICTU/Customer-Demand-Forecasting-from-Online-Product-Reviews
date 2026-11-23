"""Demand forecasting from product reviews — demo app.

Reads the files exported by notebook 4 (app_data/) and shows:
  1. Forecasts per product (actual vs. model forecasts on the test period)
  2. Review sentiment & aspects over time, with sample reviews
  3. Model results (comparison, ablation, significance, by category)
  4. About (pipeline overview)

Run:  streamlit run app.py
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

DATA = Path(os.environ.get("APP_DATA", Path(__file__).parent / "app_data"))
st.set_page_config(page_title="Customer Demand Forecasting from Reviews", layout="wide")

GREY, BLUE, RED, ORANGE = "#5f6368", "#1a73e8", "#d93025", "#e37400"
ASPECT_NAMES = {"asp_quality": "Quality", "asp_durability": "Durability", "asp_value": "Value for money",
                "asp_shipping": "Shipping", "asp_usability": "Ease of use",
                "asp_compatibility": "Compatibility", "asp_gift": "Gift"}


# ----------------------------------------------------------------------------- data
@st.cache_data
def load(data_dir: str):
    DATA = Path(data_dir)
    need = ["predictions.parquet", "weekly_demand.parquet", "weekly_nlp_features.parquet", "products.csv"]
    missing = [f for f in need if not (DATA / f).exists()]
    if missing:
        return None, missing
    d = {
        "pred": pd.read_parquet(DATA / "predictions.parquet"),
        "weekly": pd.read_parquet(DATA / "weekly_demand.parquet"),
        "nlp": pd.read_parquet(DATA / "weekly_nlp_features.parquet"),
        "products": pd.read_csv(DATA / "products.csv"),
        "reviews": pd.read_parquet(DATA / "review_samples.parquet")
        if (DATA / "review_samples.parquet").exists() else pd.DataFrame(),
        "meta": json.load(open(DATA / "meta.json")) if (DATA / "meta.json").exists() else {},
    }
    for k in ("pred", "weekly", "nlp"):
        for c in ("week", "origin_week"):
            if c in d[k]:
                d[k][c] = pd.to_datetime(d[k][c])
    d["pred"]["label"] = np.where(d["pred"]["feature_set"] == "-", d["pred"]["model"],
                                  d["pred"]["model"] + " · " + d["pred"]["feature_set"])
    d["products"]["title"] = d["products"]["title"].fillna(d["products"]["series_id"])
    return d, []


@st.cache_data
def result_table(name, data_dir=str(DATA), **kw):
    path = Path(data_dir) / "results" / name
    return pd.read_csv(path, **kw) if path.exists() else None


D, missing = load(str(DATA))
if D is None:
    st.error(f"Missing data files in `{DATA}`: {', '.join(missing)}.\n\n"
             "Run notebook 4 and unzip `app_data.zip` next to `app.py`.")
    st.stop()

pred, weekly, nlp, products, reviews, meta = (D[k] for k in ("pred", "weekly", "nlp", "products", "reviews", "meta"))
test_start = pred["week"].min()
best = meta.get("best_model", {})
default_main = f"{best.get('model', 'GRU')} · {best.get('feature_set', 'F0')}"
labels = sorted(pred["label"].unique(), key=lambda s: (s.count("·") == 0, s))


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("Customer Demand Forecasting from Reviews")
page = st.sidebar.radio("Page", ["Forecast", "Reviews & sentiment", "Model results", "About"])

st.sidebar.markdown("---")
cats = sorted(products["category"].unique())
cat = st.sidebar.selectbox("Category", cats)
prods = products[products["category"] == cat].sort_values("title")
pick = st.sidebar.selectbox(
    "Product", prods["series_id"],
    format_func=lambda sid: prods.set_index("series_id").loc[sid, "title"][:70])
prod = products.set_index("series_id").loc[pick]


def product_header():
    st.subheader(prod["title"])
    bits = [f"**Category:** {cat.replace('_', ' ')}"]
    if pd.notna(prod.get("price")):
        bits.append(f"**Price:** ${prod['price']}")
    if pd.notna(prod.get("average_rating")):
        bits.append(f"**Rating:** {prod['average_rating']}★ ({int(prod.get('rating_number', 0) or 0):,} ratings)")
    st.caption(" · ".join(bits))


def series(sid):
    w = weekly[weekly["series_id"] == sid].sort_values("week")
    return w[w["launched"]] if "launched" in w else w


# ----------------------------------------------------------------------------- page 1
if page == "Forecast":
    product_header()
    c1, c2 = st.columns([3, 1])
    with c2:
        choose = st.multiselect("Forecasts model", labels,
                                default=[l for l in [default_main, "GRU · F4", "MA4"] if l in labels])
        h = st.radio("Forecast horizon (weeks ahead)", [1, 2, 3, 4], horizontal=True)
        hist_weeks = st.slider("History shown (weeks)", 26, 208, 78, step=13)

    w = series(pick)
    p = pred[(pred["series_id"] == pick) & (pred["horizon"] == h)]
    start = test_start - pd.Timedelta(weeks=hist_weeks)

    fig = go.Figure()
    fig.add_vrect(x0=test_start, x1=w["week"].max(), fillcolor="#f1f3f4", line_width=0, layer="below",
                  annotation_text="test period", annotation_position="top left")
    fig.add_trace(go.Scatter(x=w.loc[w["week"] >= start, "week"], y=w.loc[w["week"] >= start, "n_verified"],
                             name="Actual", line=dict(color="#202124", width=2)))
    palette = [BLUE, RED, GREY, ORANGE, "#188038", "#9334e6", "#00796b", "#795548"]
    for i, lab in enumerate(choose):
        q = p[p["label"] == lab].sort_values("week")
        fig.add_trace(go.Scatter(x=q["week"], y=q["y_pred"], name=lab,
                                 line=dict(color=palette[i % len(palette)], width=1.6,
                                           dash="dot" if "·" not in lab else "solid")))
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      yaxis_title="verified reviews / week",
                      legend=dict(orientation="h", y=-0.15))
    with c1:
        st.plotly_chart(fig, width="stretch")

    # metrics for this product
    pp = pred[pred["series_id"] == pick].copy()
    pp["abs_err"] = (pp["y_pred"] - pp["y_true"]).abs()
    tbl = (pp.groupby("label").agg(MAE=("abs_err", "mean"), actual=("y_true", "sum"), err=("abs_err", "sum"))
           .assign(WAPE=lambda t: t["err"] / t["actual"].clip(lower=1e-9))[["MAE", "WAPE"]]
           .sort_values("MAE"))
    k1, k2, k3, k4 = st.columns(4)
    recent = w[w["week"] < test_start].tail(8)["n_verified"].mean()
    k1.metric("Avg demand (Last 8 weeks before test)", f"{recent:.1f} / week")
    if default_main in tbl.index and "MA4" in tbl.index:
        k2.metric(f"MAE · {default_main}", f"{tbl.loc[default_main, 'MAE']:.2f}",
                  f"{(tbl.loc[default_main, 'MAE'] / tbl.loc['MA4', 'MAE'] - 1) * 100:+.1f}% vs MA4",
                  delta_color="inverse")
    k3.metric("Best model for this product", tbl.index[0])
    k4.metric("Test weeks", f"{pp['week'].nunique()}")
    with st.expander("Error of every model on this product (test period, all horizons)"):
        st.dataframe(tbl.style.format({"MAE": "{:.2f}", "WAPE": "{:.1%}"}), width="stretch")
    st.caption("Demand proxy = number of Amazon *verified-purchase* reviews per week. "
               "Forecasts are made from the previous 26 weeks; MA4 = average of the last 4 weeks.")

# ----------------------------------------------------------------------------- page 2
elif page == "Reviews & sentiment":
    product_header()
    w = series(pick)
    f = nlp[nlp["series_id"] == pick].merge(w[["week", "n_verified"]], on="week").sort_values("week")
    smooth = st.slider("Smoothing (weeks, rolling mean)", 1, 13, 4)
    r = f.set_index("week").rolling(smooth, min_periods=1).mean(numeric_only=True).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=r["week"], y=r["n_verified"], name="Demand (verified reviews / week)",
                             line=dict(color="#202124", width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=r["week"], y=r["sent_mean"], name="Mean sentiment (−1 … 1)",
                             line=dict(color=BLUE, width=1.5)), secondary_y=True)
    fig.add_trace(go.Scatter(x=r["week"], y=r["neg_ratio"], name="Share of negative reviews",
                             line=dict(color=RED, width=1.5)), secondary_y=True)
    fig.add_vline(x=test_start, line_dash="dot", line_color=GREY)
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=-0.18))
    fig.update_yaxes(title_text="demand", secondary_y=False)
    fig.update_yaxes(title_text="sentiment / share", secondary_y=True)
    st.plotly_chart(fig, width="stretch")

    asp_cols = [c for c in f.columns if c.startswith("asp_")]
    if asp_cols:
        st.markdown("#### What reviewers talk about")
        chosen = st.multiselect("Aspects", asp_cols, default=[c for c in ["asp_quality", "asp_durability", "asp_gift"]
                                                              if c in asp_cols],
                                format_func=lambda c: ASPECT_NAMES.get(c, c))
        fig2 = go.Figure()
        for c in chosen:
            fig2.add_trace(go.Scatter(x=r["week"], y=r[c], name=ASPECT_NAMES.get(c, c)))
        fig2.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_tickformat=".0%",
                           yaxis_title="share of reviews mentioning", legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig2, width="stretch")
        overall = f[asp_cols].mean().rename(index=ASPECT_NAMES).sort_values(ascending=False)
        st.caption("Overall share of reviews per aspect: " +
                   ", ".join(f"{k} {v:.0%}" for k, v in overall.items()))

    if len(reviews):
        st.markdown("#### Most helpful reviews")
        rv = reviews[reviews["series_id"] == pick]
        col_pos, col_neg = st.columns(2)
        for col, lab, head in [(col_pos, "pos", "👍 Positive"), (col_neg, "neg", "👎 Negative")]:
            with col:
                st.markdown(f"**{head}**")
                sub = rv[rv["label"] == lab]
                if sub.empty:
                    st.caption("No reviews in the sample.")
                for _, x in sub.iterrows():
                    date = pd.to_datetime(x["date"]).date()
                    with st.expander(f"{'★' * int(x['rating'])} {str(x['title'])[:60]} — {date}"):
                        st.write(x["text"])
                        st.caption(f"sentiment {x['sent']:+.2f} · {int(x['helpful_vote'])} helpful votes")

# ----------------------------------------------------------------------------- page 3
elif page == "Model results":
    st.subheader("Model results (test period, Jul–Dec 2022)")
    st.caption("MASE: error relative to a naive forecast, averaged over products — lower is better, "
               "below 1 beats the naive forecast. Deep models: mean of 3 seeds.")

    ens = result_table("metrics_ensemble.csv")
    abl = result_table("ablation_mase.csv", index_col=0)
    if abl is not None:
        sets = [c for c in ["F0", "F1", "F2", "F3", "F4"] if c in abl.columns]
        fig = go.Figure()
        shades = ["#1a73e8", "#b0b3b8", "#9aa0a6", "#80868b", "#5f6368"]
        for s_, col in zip(sets, shades):
            fig.add_trace(go.Bar(x=abl.index, y=abl[s_], name=s_, marker_color=col))
        if ens is not None and (ens["model"] == "MA4").any():
            ma4 = ens.loc[ens["model"] == "MA4", "MASE"].iloc[0]
            fig.add_hline(y=ma4, line_dash="dash", line_color=GREY, annotation_text=f"MA4 {ma4:.3f}")
        lo = np.nanmin(abl[sets].values)
        fig.update_layout(barmode="group", height=380, margin=dict(l=10, r=10, t=30, b=10),
                          yaxis=dict(title="MASE (test)", range=[lo * 0.95, None]),
                          title="Ablation: F0 = demand only · F1 + calendar · F2 + review stats · "
                                "F3 + sentiment · F4 + aspects")
        st.plotly_chart(fig, width="stretch")
        st.caption("The y-axis does not start at zero, to make small differences visible.")

    tabs = st.tabs(["Model comparison", "Significance", "By category", "By review volume", "v1 → v2"])
    with tabs[0]:
        t = result_table("model_comparison.csv")
        st.dataframe(t, width="stretch", hide_index=True) if t is not None else st.info("No file.")
    with tabs[1]:
        t = result_table("significance.csv")
        if t is not None:
            st.dataframe(t.style.format({c: "{:.4f}" for c in t.columns if t[c].dtype.kind == "f"}),
                         width="stretch", hide_index=True)
            st.caption("Paired Wilcoxon test on per-product MAE. p < 0.05 = the difference is unlikely to be chance.")
        else:
            st.info("No file.")
    with tabs[2]:
        t = result_table("mase_by_category.csv", index_col=0)
        st.dataframe(t.round(3), width="stretch") if t is not None else st.info("No file.")
    with tabs[3]:
        t = result_table("volume_split.csv")
        if t is not None:
            st.dataframe(t.round(4), width="stretch", hide_index=True)
            st.caption("Products split at the median number of verified reviews per week (training period).")
        else:
            st.info("Run notebook 4 to create this table.")
    with tabs[4]:
        t = result_table("v1_vs_v2.csv", header=[0, 1])
        if t is not None:
            t.columns = [" ".join(c for c in col if not c.startswith("Unnamed")).strip() for col in t.columns]
            st.dataframe(t.round(3), width="stretch", hide_index=True)
            st.caption("v1 scaled each product by its training mean; v2 normalises each input window by its "
                       "recent level. Negative Δ = v2 better.")
        else:
            st.info("No v1 results found.")

# ----------------------------------------------------------------------------- page 4
else:
    st.subheader("About this project")
    st.markdown(f"""
**Question:** Do signals extracted from customer reviews (NLP) improve weekly demand forecasts?

**Data:** Amazon Reviews 2023 (McAuley Lab, UCSD, in 5 categories, {products['series_id'].nunique()} products,
weekly series from {weekly['week'].min().date()} to {weekly['week'].max().date()}.
Demand proxy = verified-purchase reviews per week.

**Pipeline**
1. *Data preparation*: select products with dense review histories, build weekly series.
2. *NLP*: RoBERTa sentiment for every review (checked against star ratings), 7 aspects
   (quality, durability, value, shipping, ease of use, compatibility, gift) via sentence embeddings,
   aggregated per week.
3. *Forecasting*: 26 weeks in → 4 weeks out; baselines (Naive, Seasonal Naive, MA4), Ridge, LightGBM,
   LSTM, GRU, Transformer as global models; feature ablation F0 → F4; per-window normalisation (v2).
4. *Evaluation*: time-based split (train ≤ {meta.get('train_end', '2021-12')}, validation to
   {meta.get('val_end', '2022-06')}, test after); MASE, WAPE, Wilcoxon tests.

**Main findings:** All models beat MA4 after per-window normalisation; demand history alone gives the
best average accuracy; NLP features reduce large errors and help in review-rich categories, but do not
improve average scaled error.
""")
