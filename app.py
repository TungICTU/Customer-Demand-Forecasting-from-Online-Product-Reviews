import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# 1. CẤU HÌNH TRANG & CSS CHUẨN STREAMLIT CONTAINER
# =============================================================================
st.set_page_config(
    page_title="DemandPulse AI — Dự báo nhu cầu tiêu dùng",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Ẩn thanh trang trí thừa của Streamlit */
    [data-testid="stDecoration"] { display: none; }
    header[data-testid="stHeader"] { background: rgba(248, 250, 252, 0.85); backdrop-filter: blur(8px); }

    .stApp { background-color: #F8FAFC; }
    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    /* Style trực tiếp cho st.container(border=True) thành Card chuẩn Figma */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0 !important;
        border-radius: 16px !important;
        padding: 1.15rem 1.25rem !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    /* Hero Banner */
    .hero-wrap {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 55%, #312E81 100%);
        color: #FFFFFF;
        padding: 1.6rem 2rem;
        border-radius: 18px;
        margin-bottom: 1.25rem;
        box-shadow: 0 8px 24px -8px rgba(15, 23, 42, 0.25);
    }
    .hero-pill {
        display: inline-block;
        background: rgba(129, 140, 248, 0.2);
        border: 1px solid rgba(165, 180, 252, 0.35);
        color: #E0E7FF;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 999px;
        margin-bottom: 10px;
    }
    .hero-title {
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0 0 6px 0;
        color: #FFFFFF;
    }
    .hero-desc {
        font-size: 0.88rem;
        color: #CBD5E1;
        margin: 0;
        line-height: 1.6;
        max-width: 900px;
    }

    /* KPI Card */
    .kpi-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }
    .kpi-box.indigo {
        border-color: #C7D2FE;
        background: linear-gradient(180deg, #FFFFFF 0%, #F5F7FF 100%);
    }
    .kpi-box.emerald {
        border-color: #A7F3D0;
        background: linear-gradient(180deg, #FFFFFF 0%, #F0FDF9 100%);
    }
    .kpi-lbl {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748B;
    }
    .kpi-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.7rem;
        font-weight: 700;
        color: #0F172A;
        margin: 6px 0 4px 0;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 12px;
        color: #475569;
    }

    /* Card Heading */
    .card-title {
        font-size: 0.98rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0 0 2px 0;
    }
    .card-sub {
        font-size: 0.78rem;
        color: #64748B;
        margin: 0 0 12px 0;
    }

    /* Segmented Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: #FFFFFF;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px;
        border-radius: 8px;
        padding: 0 16px;
        font-size: 0.84rem;
        font-weight: 600;
        color: #475569;
        border: none !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [aria-selected="true"] {
        background-color: #4F46E5 !important;
        color: #FFFFFF !important;
    }

    /* Review Card */
    .rev-item {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.75rem;
    }
    .rev-item.pos { border-left: 3.5px solid #059669; }
    .rev-item.neg { border-left: 3.5px solid #E11D48; }
    .rev-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 11px;
        color: #64748B;
        margin-bottom: 6px;
    }
    .rev-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 999px;
    }
    .rev-pill.pos { background: #DCFCE7; color: #166534; }
    .rev-pill.neg { background: #FFE4E6; color: #9F1239; }
    .rev-h { font-size: 13px; font-weight: 700; color: #0F172A; margin-bottom: 4px; }
    .rev-p { font-size: 12px; color: #334155; line-height: 1.55; margin-bottom: 8px; }
    .chip {
        display: inline-block;
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        color: #334155;
        font-family: 'JetBrains Mono', monospace;
        font-size: 10.5px;
        padding: 1px 7px;
        border-radius: 6px;
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


def card_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="card-sub">{subtitle}</div>', unsafe_allow_html=True)


def kpi_card(label: str, value: str, sub: str, tone: str = ""):
    st.markdown(
        f"""
        <div class="kpi-box {tone}">
            <div class="kpi-lbl">{label}</div>
            <div class="kpi-val">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# 2. NẠP DỮ LIỆU GỐC TỪ app_data/
# =============================================================================
DATA_DIR = Path(__file__).parent / "app_data"
RES_DIR = DATA_DIR / "results"


@st.cache_data(show_spinner="Đang tải dữ liệu từ app_data/...")
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
                raw.columns = [
                    a if ("Unnamed" in str(b) or str(b).strip() == "") else f"{a} ({b})"
                    for a, b in raw.columns
                ]
                results[p.stem] = raw
            else:
                results[p.stem] = pd.read_csv(p)

    train_end = pd.Timestamp(meta.get("train_end", "2021-12-27"))
    tr = weekly[(weekly["week"] <= train_end) & (weekly["launched"] == True)]
    vol = tr.groupby("series_id")["n_verified"].mean()
    median_vol = float(vol.median()) if len(vol) else 0.0

    products["reviews_per_week"] = products["series_id"].map(vol).fillna(0.0)
    products["volume_group"] = np.where(
        products["reviews_per_week"] >= median_vol, "Nhiều review", "Ít review"
    )
    return meta, products, weekly, nlp, pred, reviews, results, median_vol


if not DATA_DIR.exists():
    st.error("Không tìm thấy thư mục `app_data/` cạnh `app.py`.")
    st.stop()

meta, products, weekly, nlp, pred, reviews, results, median_vol = load_data()

# =============================================================================
# 3. SIDEBAR GỌN GÀNG
# =============================================================================
with st.sidebar:
    st.markdown("### 📈 DemandPulse AI")
    st.caption("Amazon Reviews 2023 • Temporal NLP + Deep Learning")
    st.divider()

    best_m = meta.get("best_model", {})
    with st.container(border=True):
        st.markdown("**Thông số thực nghiệm**")
        st.caption(
            f"• Phiên bản: **{meta.get('version', 'v2')}** (`{meta.get('norm', 'window')}`)\n\n"
            f"• Cửa sổ trượt: **L={meta.get('L', 26)} → H={meta.get('H', 4)}** tuần\n\n"
            f"• Train ≤ `{meta.get('train_end', '-')}`\n\n"
            f"• Val ≤ `{meta.get('val_end', '-')}`\n\n"
            f"• Best Val: **{best_m.get('model', '-')} {best_m.get('feature_set', '-')}**\n\n"
            f"• Trung vị mật độ: **{median_vol:.1f}** rev/tuần"
        )

    with st.container(border=True):
        st.markdown("**Quy ước bộ đặc trưng**")
        st.caption(
            "**F0**: Lịch sử nhu cầu\n\n"
            "**F1**: + Lịch & mùa lễ\n\n"
            "**F2**: + Rating & Helpful\n\n"
            "**F3**: + Cảm xúc (RoBERTa)\n\n"
            "**F4**: + 7 Khía cạnh (Aspects)"
        )

# =============================================================================
# 4. HERO HEADER
# =============================================================================
n_series = int(weekly["series_id"].nunique())
n_cats = int(weekly["category"].nunique())
n_weeks = int(weekly["week"].nunique())
tot_reviews = int(weekly["n_reviews"].sum())
tot_verified = int(weekly["n_verified"].sum())

st.markdown(
    f"""
    <div class="hero-wrap">
        <div class="hero-pill">SIC Capstone • Amazon Reviews 2023 • Model {meta.get("version", "v2").upper()}</div>
        <div class="hero-title">Dự báo nhu cầu tiêu dùng từ dữ liệu mạng xã hội bằng NLP & Học sâu</div>
        <p class="hero-desc">
            Đánh giá trên <b>{n_series}</b> chuỗi sản phẩm thuộc <b>{n_cats}</b> ngành hàng ({n_weeks} tuần,
            {tot_reviews:,} reviews) — đối chiếu Baseline, Machine Learning, Deep Learning và kiểm chứng hiệu ứng NLP
            theo mật độ đánh giá.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# 5. BỐN MÀN HÌNH CHÍNH
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Tổng quan", "🔎 Dự báo sản phẩm", "🧪 Ablation & v1→v2", "✨ NLP & Mật độ review"]
)

# -----------------------------------------------------------------------------
# TAB 1: TỔNG QUAN
# -----------------------------------------------------------------------------
with tab1:
    ens_df = results.get("metrics_ensemble", pd.DataFrame())
    vol_df = results.get("volume_split", pd.DataFrame())

    best_val, best_sub = "-", "Từ metrics_ensemble.csv"
    if not ens_df.empty and best_m:
        rb = ens_df[(ens_df["model"] == best_m.get("model")) & (ens_df["feature_set"] == best_m.get("feature_set"))]
        rm = ens_df[ens_df["model"] == "MA4"]
        if not rb.empty:
            bm = float(rb["MASE"].iloc[0])
            bw = float(rb["WAPE"].iloc[0])
            best_val = f"{bm:.3f}"
            if not rm.empty:
                pct = (bm / float(rm["MASE"].iloc[0]) - 1) * 100
                best_sub = f"{best_m.get('model')} {best_m.get('feature_set')} • WAPE {bw:.3f} ({pct:+.1f}% vs MA4)"
            else:
                best_sub = f"{best_m.get('model')} {best_m.get('feature_set')} • WAPE {bw:.3f}"

    nlp_val, nlp_sub = "-", "Xem tab NLP & Mật độ"
    if not vol_df.empty:
        hv = vol_df[vol_df["group"].astype(str).str.contains("nhiều", case=False, na=False) & (vol_df["model"] == "GRU")]
        if not hv.empty:
            r = hv.iloc[0]
            nlp_val = f"{float(r['MASE F4']):.4f}"
            dc = [c for c in vol_df.columns if "MASE %" in c]
            pc = [c for c in vol_df.columns if "Wilcoxon" in c]
            nlp_sub = f"GRU F4 nhiều review • {float(r[dc[0]]):+.2f}% (p={float(r[pc[0]]):.4f})" if dc and pc else "GRU F4 nhiều review"

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Quy mô chuỗi", f"{n_series}", f"{n_cats} ngành × {n_weeks} tuần")
    with k2:
        kpi_card("Tổng đánh giá", f"{tot_reviews:,}", f"{tot_verified:,} verified purchase")
    with k3:
        kpi_card("Best Model (Test MASE)", best_val, best_sub, tone="indigo")
    with k4:
        kpi_card("NLP nhóm nhiều review", nlp_val, nlp_sub, tone="emerald")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    c_l, c_r = st.columns(2)
    with c_l:
        with st.container(border=True):
            card_header("MASE theo 5 ngành hàng (Tập Test)", "So sánh Baseline MA4 với GRU không NLP (F2) và có NLP (F4)")
            by_cat = results.get("mase_by_category", pd.DataFrame())
            if not by_cat.empty:
                cc = by_cat.columns[0]
                pcols = [c for c in ["MA4", "GRU F2", "GRU F4", "LightGBM F4"] if c in by_cat.columns]
                m_df = by_cat.melt(id_vars=cc, value_vars=pcols, var_name="Mô hình", value_name="MASE")
                m_df[cc] = m_df[cc].astype(str).str.replace("_", " ")
                fig = px.bar(
                    m_df, x=cc, y="MASE", color="Mô hình", barmode="group",
                    color_discrete_sequence=["#CBD5E1", "#6366F1", "#059669", "#0891B2"],
                )
                st.plotly_chart(style_fig(fig, 330, "", "MASE ↓"), use_container_width=True, config=PLOT_CFG)

    with c_r:
        with st.container(border=True):
            card_header("Xếp hạng Top 10 mô hình (Ensemble)", "Đánh giá trên tập test theo chỉ số MASE")
            if not ens_df.empty:
                top = ens_df.sort_values("MASE").head(10).copy()
                top["Tên"] = top["model"] + " " + top["feature_set"].replace("-", "(Base)")
                top["Loại"] = np.where(
                    top["feature_set"] == "-", "Baseline",
                    np.where(top["feature_set"].isin(["F3", "F4"]), "Có NLP", "Không NLP"),
                )
                top = top.iloc[::-1]
                fig = px.bar(
                    top, x="MASE", y="Tên", orientation="h", color="Loại",
                    text=top["MASE"].map(lambda x: f"{x:.3f}"),
                    color_discrete_map={"Không NLP": "#4F46E5", "Có NLP": "#059669", "Baseline": "#D97706"},
                )
                fig.update_traces(textposition="outside", cliponaxis=False)
                fig.update_xaxes(range=[0, float(top["MASE"].max()) * 1.15])
                st.plotly_chart(style_fig(fig, 330, "Test MASE", ""), use_container_width=True, config=PLOT_CFG)

    with st.container(border=True):
        card_header("Thống kê mô tả theo ngành hàng", "Tính trực tiếp từ weekly_demand.parquet trên các tuần đã ra mắt")
        eda = (
            weekly[weekly["launched"] == True]
            .groupby("category")
            .agg(
                chuoi=("series_id", "nunique"),
                rev=("n_reviews", "sum"),
                ver=("n_verified", "sum"),
                tb=("n_verified", "mean"),
                z0=("n_verified", lambda s: (s == 0).mean() * 100),
            )
            .reset_index()
        )
        eda["category"] = eda["category"].str.replace("_", " ")
        eda.columns = ["Ngành hàng", "Số chuỗi", "Tổng review", "Verified", "TB/tuần", "Tuần = 0 (%)"]
        st.dataframe(
            eda.style.format({"Tổng review": "{:,.0f}", "Verified": "{:,.0f}", "TB/tuần": "{:.2f}", "Tuần = 0 (%)": "{:.1f}%"}),
            use_container_width=True,
            hide_index=True,
        )

# -----------------------------------------------------------------------------
# TAB 2: DỰ BÁO THEO SẢN PHẨM
# -----------------------------------------------------------------------------
with tab2:
    with st.container(border=True):
        f1, f2, f3, f4 = st.columns([1.2, 1.0, 2.2, 0.8])
        with f1:
            sel_cat = st.selectbox("Ngành hàng", ["Tất cả"] + sorted(products["category"].dropna().unique().tolist()))
        with f2:
            sel_vol = st.selectbox("Mật độ review", ["Tất cả", "Nhiều review", "Ít review"])

        pf = products.copy()
        if sel_cat != "Tất cả":
            pf = pf[pf["category"] == sel_cat]
        if sel_vol != "Tất cả":
            pf = pf[pf["volume_group"] == sel_vol]
        pf = pf.sort_values("total", ascending=False)

        if pf.empty:
            st.warning("Không có sản phẩm phù hợp bộ lọc.")
            st.stop()

        with f3:
            opts = pf["series_id"].tolist()
            lmap = {r["series_id"]: f"[{r['parent_asin']}] {str(r.get('title', r['series_id']))[:62]}" for _, r in pf.iterrows()}
            sel_sid = st.selectbox("Sản phẩm", opts, format_func=lambda x: lmap.get(x, x))
        with f4:
            sel_h = st.selectbox("Horizon", sorted(pred["horizon"].unique().tolist()), index=0)

        m1, m2 = st.columns([1.1, 2.5])
        avail = sorted([m for m in pred["model"].unique() if m not in ("Naive", "SeasonalNaive", "MA4")])
        with m1:
            sel_model = st.selectbox("Mô hình chính", avail, index=avail.index("GRU") if "GRU" in avail else 0)
        with m2:
            sel_lines = st.multiselect("Đường dự báo hiển thị", ["F0", "F2", "F4", "MA4"], default=["F0", "F4", "MA4"])

    pr = products[products["series_id"] == sel_sid].iloc[0]
    price_str = f"${float(pr['price']):.2f}" if pd.notna(pr.get("price")) else "N/A"
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        kpi_card("Mã sản phẩm (ASIN)", str(pr["parent_asin"]), str(pr["category"]).replace("_", " "))
    with p2:
        kpi_card("Giá & Đánh giá", price_str, f"Rating TB: {pr.get('average_rating', 'N/A')} ★")
    with p3:
        kpi_card("Tổng Verified", f"{int(pr.get('total', 0)):,}", f"TB tập train: {pr['reviews_per_week']:.1f} / tuần")
    with p4:
        kpi_card("Nhóm mật độ", pr["volume_group"], f"Trung vị: {median_vol:.1f} rev/tuần", tone="emerald" if pr["volume_group"] == "Nhiều review" else "")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    g_l, g_r = st.columns([2.1, 1.0])
    with g_l:
        with st.container(border=True):
            card_header(f"Dự báo {sel_model} so với thực tế (h = {sel_h})", "Đường nét đứt dọc đánh dấu mốc kết thúc Validation → bắt đầu Test")
            w_s = weekly[weekly["series_id"] == sel_sid].sort_values("week")
            p_s = pred[(pred["series_id"] == sel_sid) & (pred["horizon"] == sel_h)].sort_values("week")
            val_end = pd.Timestamp(meta.get("val_end", "2022-06-27"))
            w_show = w_s[w_s["week"] >= val_end - pd.Timedelta(weeks=36)]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=w_show["week"], y=w_show["n_verified"], mode="lines+markers",
                    name="Thực tế", line=dict(color="#0F172A", width=2.2), marker=dict(size=4),
                    fill="tozeroy", fillcolor="rgba(15, 23, 42, 0.04)",
                )
            )
            cmap = {"F0": "#4F46E5", "F2": "#0891B2", "F4": "#059669", "MA4": "#D97706"}
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
            st.plotly_chart(style_fig(fig, 360, "", "Review verified / tuần"), use_container_width=True, config=PLOT_CFG)

    with g_r:
        with st.container(border=True):
            card_header("Khía cạnh & Cảm xúc NLP", "Trung bình tỉ lệ nhắc tới theo tuần")
            n_s = nlp[nlp["series_id"] == sel_sid]
            acols = [c for c in n_s.columns if c.startswith("asp_")]
            if not n_s.empty and acols:
                am = (n_s[acols].mean() * 100).sort_values()
                adf = pd.DataFrame({"Khía cạnh": [c.replace("asp_", "").capitalize() for c in am.index], "Tỉ lệ": am.values})
                fig_a = px.bar(
                    adf, x="Tỉ lệ", y="Khía cạnh", orientation="h",
                    text=adf["Tỉ lệ"].map(lambda v: f"{v:.1f}%"),
                    color_discrete_sequence=["#0891B2"],
                )
                fig_a.update_traces(textposition="outside", cliponaxis=False)
                fig_a.update_xaxes(range=[0, max(float(adf["Tỉ lệ"].max()) * 1.25, 5)])
                st.plotly_chart(style_fig(fig_a, 275, "% Review", ""), use_container_width=True, config=PLOT_CFG)

                st.caption(
                    f"Sentiment TB: **{float(n_s['sent_mean'].mean()):+.2f}** • "
                    f"Tích cực: **{float(n_s['pos_ratio'].mean())*100:.1f}%** • "
                    f"Tiêu cực: **{float(n_s['neg_ratio'].mean())*100:.1f}%**"
                )

    with st.container(border=True):
        card_header("Review tiêu biểu (RoBERTa Sentiment)", "Trích từ review_samples.parquet (tối đa 8 pos + 8 neg hữu ích nhất)")
        r_s = reviews[reviews["series_id"] == sel_sid].copy()
        if r_s.empty:
            st.info("Không có mẫu review cho sản phẩm này.")
        else:
            rf = st.radio("Lọc cảm xúc", ["Tất cả", "Tích cực", "Tiêu cực"], horizontal=True, label_visibility="collapsed")
            if rf == "Tích cực":
                r_s = r_s[r_s["label"] == "pos"]
            elif rf == "Tiêu cực":
                r_s = r_s[r_s["label"] == "neg"]

            arcols = [c for c in r_s.columns if c.startswith("asp_")]
            rc1, rc2 = st.columns(2)
            for idx_r, (_, rr) in enumerate(r_s.head(8).iterrows()):
                lbl = str(rr.get("label", "pos"))
                pill = f"{'Positive' if lbl == 'pos' else 'Negative'} • {float(rr.get('sent', 0)):+.2f}"
                stars = "★" * int(round(float(rr.get("rating", 5))))
                chips = "".join(f'<span class="chip">#{c.replace("asp_", "")}</span>' for c in arcols if rr.get(c, 0) == 1)
                html = f"""
                <div class="rev-item {lbl}">
                    <div class="rev-top">
                        <span class="rev-pill {lbl}">{pill}</span>
                        <span>{stars} • {int(rr.get('helpful_vote', 0))} helpful</span>
                    </div>
                    <div class="rev-h">{str(rr.get('title', ''))[:120]}</div>
                    <div class="rev-p">{str(rr.get('text', ''))[:320]}</div>
                    <div>{chips}</div>
                </div>
                """
                (rc1 if idx_r % 2 == 0 else rc2).markdown(html, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 3: ABLATION & V1 -> V2
# -----------------------------------------------------------------------------
with tab3:
    abl = results.get("ablation_mase", pd.DataFrame())
    if not abl.empty:
        with st.container(border=True):
            card_header("Thí nghiệm Ablation (F0 → F4)", "Test MASE khi bổ sung dần lịch, metadata review, cảm xúc và khía cạnh")
            mc = abl.columns[0]
            fsc = [c for c in ["F0", "F1", "F2", "F3", "F4"] if c in abl.columns]
            am = abl.melt(id_vars=mc, value_vars=fsc, var_name="Bộ ĐT", value_name="MASE")
            fig = px.bar(
                am, x=mc, y="MASE", color="Bộ ĐT", barmode="group",
                color_discrete_sequence=["#4F46E5", "#818CF8", "#0891B2", "#14B8A6", "#059669"],
            )
            ens_df = results.get("metrics_ensemble", pd.DataFrame())
            if not ens_df.empty and "MA4" in ens_df["model"].values:
                m4 = float(ens_df.loc[ens_df["model"] == "MA4", "MASE"].iloc[0])
                fig.add_hline(y=m4, line_dash="dash", line_color="#D97706", annotation_text=f"MA4 ({m4:.3f})")
            ymin = float(am["MASE"].min()) * 0.92
            fig.update_yaxes(range=[ymin, float(am["MASE"].max()) * 1.06])
            st.plotly_chart(style_fig(fig, 360, "", "Test MASE ↓"), use_container_width=True, config=PLOT_CFG)

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            card_header("So sánh cải thiện v1 → v2", "Hiệu quả của chuẩn hoá theo cửa sổ 8 tuần (v1_vs_v2.csv)")
            v12 = results.get("v1_vs_v2", pd.DataFrame())
            if not v12.empty:
                st.dataframe(v12, use_container_width=True, hide_index=True, height=340)
            else:
                st.info("Chưa có file v1_vs_v2.csv.")
    with c2:
        with st.container(border=True):
            card_header("Kiểm định Wilcoxon Signed-Rank", "Đánh giá ý nghĩa thống kê trên 150 chuỗi (significance.csv)")
            sig = results.get("significance", pd.DataFrame())
            if not sig.empty:
                st.dataframe(sig, use_container_width=True, hide_index=True, height=340)
            else:
                st.info("Chưa có file significance.csv.")

    comp = results.get("model_comparison", pd.DataFrame())
    if not comp.empty:
        with st.container(border=True):
            h1, h2 = st.columns([3.5, 1])
            with h1:
                card_header("Bảng tổng hợp toàn bộ mô hình", "Trung bình ± độ lệch chuẩn qua 3 seed (model_comparison.csv)")
            with h2:
                st.download_button(
                    "⬇ Tải CSV bảng kết quả",
                    data=comp.to_csv(index=False).encode("utf-8-sig"),
                    file_name="model_comparison.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            st.dataframe(comp, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# TAB 4: NLP & MẬT ĐỘ REVIEW
# -----------------------------------------------------------------------------
with tab4:
    vol_df = results.get("volume_split", pd.DataFrame())
    nl, nr = st.columns([1.4, 1.0])

    with nl:
        with st.container(border=True):
            card_header("Tác động NLP: Nhiều review vs Ít review", "Δ MASE (%) khi đi từ F2 → F4 (giá trị âm = NLP giảm sai số)")
            if not vol_df.empty:
                dcols = [c for c in vol_df.columns if "MASE %" in c]
                if dcols:
                    fig = px.bar(
                        vol_df, x="model", y=dcols[0], color="group", barmode="group",
                        color_discrete_sequence=["#059669", "#E11D48"],
                    )
                    fig.add_hline(y=0, line_color="#64748B", line_width=1)
                    st.plotly_chart(style_fig(fig, 340, "", "Δ MASE F2→F4 (%)"), use_container_width=True, config=PLOT_CFG)

    with nr:
        with st.container(border=True):
            card_header("Tần suất 7 khía cạnh NLP toàn tập", "Tính trung bình trên weekly_nlp_features.parquet")
            acols = [c for c in nlp.columns if c.startswith("asp_")]
            if acols:
                gm = (nlp[acols].mean() * 100).sort_values()
                gdf = pd.DataFrame({"Khía cạnh": [c.replace("asp_", "").capitalize() for c in gm.index], "Tỉ lệ": gm.values})
                fig = px.bar(
                    gdf, x="Tỉ lệ", y="Khía cạnh", orientation="h",
                    text=gdf["Tỉ lệ"].map(lambda v: f"{v:.1f}%"),
                    color_discrete_sequence=["#4F46E5"],
                )
                fig.update_traces(textposition="outside", cliponaxis=False)
                fig.update_xaxes(range=[0, float(gdf["Tỉ lệ"].max()) * 1.25])
                st.plotly_chart(style_fig(fig, 340, "% Trung bình/tuần", ""), use_container_width=True, config=PLOT_CFG)

    if not vol_df.empty:
        with st.container(border=True):
            card_header("Bảng chi tiết phân tích theo lượng review", "Trích từ volume_split.csv (Notebook 4)")
            st.dataframe(vol_df, use_container_width=True, hide_index=True)
