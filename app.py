import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# 1. CẤU HÌNH TRANG & DESIGN SYSTEM (CUSTOM CSS)
# =============================================================================
st.set_page_config(
    page_title="DemandPulse AI — Dự báo nhu cầu tiêu dùng (NLP & Deep Learning)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background-color: #F8FAFC;
    color: #0F172A;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
    max-width: 1320px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}

/* Hero Header */
.hero-box {
    background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 60%, #312E81 100%);
    color: #FFFFFF;
    padding: 1.5rem 1.75rem;
    border-radius: 16px;
    margin-bottom: 1.25rem;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);
}
.hero-badge {
    display: inline-block;
    background: rgba(129, 140, 248, 0.22);
    color: #C7D2FE;
    border: 1px solid rgba(165, 180, 252, 0.35);
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.hero-title {
    font-size: 1.45rem;
    font-weight: 700;
    margin: 0 0 6px 0;
    line-height: 1.3;
}
.hero-sub {
    font-size: 0.875rem;
    color: #CBD5E1;
    margin: 0;
    line-height: 1.55;
}

/* KPI Cards */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1rem 1.15rem;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    height: 100%;
}
.kpi-card.accent {
    border-color: #C7D2FE;
    background: linear-gradient(180deg, #FFFFFF 0%, #EEF2FF 100%);
}
.kpi-card.success {
    border-color: #A7F3D0;
    background: linear-gradient(180deg, #FFFFFF 0%, #ECFDF5 100%);
}
.kpi-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.65rem;
    font-weight: 700;
    color: #0F172A;
    margin: 6px 0 4px 0;
    line-height: 1.15;
}
.kpi-sub {
    font-size: 0.78rem;
    color: #475569;
}

/* Section Card */
.section-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #FFFFFF;
    padding: 6px;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 600;
    font-size: 0.85rem;
    color: #475569;
}
.stTabs [aria-selected="true"] {
    background-color: #4F46E5 !important;
    color: #FFFFFF !important;
}

/* Review Card */
.rev-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.75rem;
}
.rev-card.pos { border-left: 4px solid #059669; }
.rev-card.neg { border-left: 4px solid #E11D48; }
.rev-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.75rem;
    color: #64748B;
    margin-bottom: 6px;
}
.rev-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 999px;
}
.rev-badge.pos { background: #ECFDF5; color: #047857; }
.rev-badge.neg { background: #FFF1F2; color: #BE123C; }
.rev-title {
    font-weight: 700;
    font-size: 0.85rem;
    color: #0F172A;
    margin-bottom: 4px;
}
.rev-text {
    font-size: 0.82rem;
    color: #334155;
    line-height: 1.5;
    margin-bottom: 8px;
}
.asp-chip {
    display: inline-block;
    background: #F1F5F9;
    color: #334155;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 7px;
    border-radius: 6px;
    margin-right: 4px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, sans-serif", size=12, color="#334155"),
    margin=dict(l=16, r=16, t=42, b=16),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
)

# =============================================================================
# 2. NẠP DỮ LIỆU TRỰC TIẾP TỪ app_data/ (KHÔNG THAY ĐỔI DỮ LIỆU)
# =============================================================================
DATA_DIR = Path(__file__).parent / "app_data"
RES_DIR = DATA_DIR / "results"


@st.cache_data(show_spinner="Đang tải dữ liệu từ app_data/...")
def load_all_data():
    meta = json.loads((DATA_DIR / "meta.json").read_text(encoding="utf-8"))
    products = pd.read_csv(DATA_DIR / "products.csv")
    weekly = pd.read_parquet(DATA_DIR / "weekly_demand.parquet")
    nlp = pd.read_parquet(DATA_DIR / "weekly_nlp_features.parquet")
    pred = pd.read_parquet(DATA_DIR / "predictions.parquet")
    reviews = pd.read_parquet(DATA_DIR / "review_samples.parquet")

    weekly["week"] = pd.to_datetime(weekly["week"])
    nlp["week"] = pd.to_datetime(nlp["week"])
    pred["week"] = pd.to_datetime(pred["week"])
    if "origin_week" in pred.columns:
        pred["origin_week"] = pd.to_datetime(pred["origin_week"])

    results = {}
    if RES_DIR.exists():
        for csv_path in sorted(RES_DIR.glob("*.csv")):
            results[csv_path.stem] = pd.read_csv(csv_path)

    # Phân nhóm lượng review dựa trên tập train giống Notebook 4
    train_end = pd.Timestamp(meta.get("train_end", "2021-12-27"))
    tr_mask = (weekly["week"] <= train_end) & (weekly["launched"] == True)
    vol = weekly.loc[tr_mask].groupby("series_id")["n_verified"].mean()
    median_vol = float(vol.median()) if len(vol) else 0.0

    products["reviews_per_week"] = products["series_id"].map(vol).fillna(0.0)
    products["volume_group"] = np.where(
        products["reviews_per_week"] >= median_vol, "Nhiều review", "Ít review"
    )
    return meta, products, weekly, nlp, pred, reviews, results, median_vol


if not DATA_DIR.exists():
    st.error("Không tìm thấy thư mục `app_data/`. Vui lòng đảm bảo thư mục dữ liệu nằm cạnh `app.py`.")
    st.stop()

meta, products, weekly, nlp, pred, reviews, results, median_vol = load_all_data()


def kpi_card(label: str, value: str, sub: str, variant: str = ""):
    st.markdown(
        f"""
        <div class="kpi-card {variant}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# 3. SIDEBAR THÔNG TIN DỰ ÁN
# =============================================================================
with st.sidebar:
    st.markdown("### 📈 DemandPulse AI")
    st.caption("Dự báo nhu cầu tiêu dùng bằng NLP theo thời gian & Học sâu")
    st.divider()

    st.markdown("**Cấu hình mô hình (`meta.json`)**")
    best_m = meta.get("best_model", {})
    st.markdown(
        f"""
        - **Phiên bản:** `{meta.get("version", "v2")}` (`{meta.get("norm", "window")}`)
        - **Cửa sổ trượt:** `L = {meta.get("L", 26)}` tuần → `H = {meta.get("H", 4)}` tuần
        - **Train đến:** `{meta.get("train_end", "-")}`
        - **Val đến:** `{meta.get("val_end", "-")}`
        - **Best (Val):** `{best_m.get("model", "-")} {best_m.get("feature_set", "-")}`
        - **Trung vị review:** `{median_vol:.1f}` / tuần
        """
    )
    st.divider()
    st.markdown("**Bộ đặc trưng (Ablation)**")
    st.caption(
        "- **F0**: Lịch sử nhu cầu\n"
        "- **F1**: F0 + Lịch & mùa lễ\n"
        "- **F2**: F1 + Rating & Helpful\n"
        "- **F3**: F2 + Sentiment (RoBERTa)\n"
        "- **F4**: F3 + 7 khía cạnh (Aspects)"
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
    <div class="hero-box">
        <div class="hero-badge">Capstone Project • Amazon Reviews 2023 • Model {meta.get("version", "v2").upper()}</div>
        <div class="hero-title">Dự báo nhu cầu tiêu dùng dựa trên dữ liệu mạng xã hội bằng NLP & Học sâu</div>
        <p class="hero-sub">
            Hệ thống đánh giá trực quan trên <b>{n_series}</b> chuỗi sản phẩm thuộc <b>{n_cats}</b> ngành hàng
            ({n_weeks} tuần, {tot_reviews:,} reviews) — đối chiếu các mô hình học sâu (GRU, LSTM, Transformer),
            ML (LightGBM, Ridge) và kiểm chứng tác động của đặc trưng NLP theo mật độ review.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# 5. ĐIỀU HƯỚNG 4 TAB CHÍNH
# =============================================================================
tab_overview, tab_explorer, tab_ablation, tab_nlp = st.tabs(
    [
        "📊 Tổng quan kết quả",
        "🔎 Dự báo theo sản phẩm",
        "🧪 Ablation & v1 → v2",
        "✨ NLP & Mật độ review",
    ]
)

# -----------------------------------------------------------------------------
# TAB 1: TỔNG QUAN KẾT QUẢ
# -----------------------------------------------------------------------------
with tab_overview:
    ens_df = results.get("metrics_ensemble", pd.DataFrame())
    vol_df = results.get("volume_split", pd.DataFrame())

    best_mase_str, best_sub = "-", "Chưa có metrics_ensemble.csv"
    if not ens_df.empty and best_m:
        row_best = ens_df[
            (ens_df["model"] == best_m.get("model"))
            & (ens_df["feature_set"] == best_m.get("feature_set"))
        ]
        row_ma4 = ens_df[ens_df["model"] == "MA4"]
        if not row_best.empty:
            b_mase = float(row_best["MASE"].iloc[0])
            b_wape = float(row_best["WAPE"].iloc[0])
            best_mase_str = f"{b_mase:.3f}"
            if not row_ma4.empty:
                m4 = float(row_ma4["MASE"].iloc[0])
                diff_pct = (b_mase / m4 - 1) * 100
                best_sub = f"{best_m.get('model')} {best_m.get('feature_set')} • WAPE {b_wape:.3f} ({diff_pct:+.1f}% vs MA4)"
            else:
                best_sub = f"{best_m.get('model')} {best_m.get('feature_set')} • WAPE {b_wape:.3f}"

    nlp_val_str, nlp_sub = "-", "Xem tab NLP & Mật độ review"
    if not vol_df.empty:
        hv_gru = vol_df[
            vol_df["group"].astype(str).str.contains("nhiều", case=False, na=False)
            & (vol_df["model"] == "GRU")
        ]
        if not hv_gru.empty:
            r = hv_gru.iloc[0]
            nlp_val_str = f"{float(r['MASE F4']):.4f}"
            delta_col = [c for c in vol_df.columns if "MASE %" in c]
            p_col = [c for c in vol_df.columns if "Wilcoxon" in c]
            d_txt = f"Δ {float(r[delta_col[0]]):+.2f}%" if delta_col else ""
            p_txt = f"p = {float(r[p_col[0]]):.4f}" if p_col else ""
            nlp_sub = f"GRU F4 nhóm nhiều review • {d_txt} ({p_txt})"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Chuỗi thời gian", f"{n_series}", f"{n_cats} ngành hàng × {n_weeks} tuần")
    with c2:
        kpi_card("Dữ liệu đánh giá", f"{tot_reviews:,}", f"{tot_verified:,} review verified")
    with c3:
        kpi_card("Best Model (Test MASE)", best_mase_str, best_sub, variant="accent")
    with c4:
        kpi_card("Hiệu ứng NLP (Nhiều review)", nlp_val_str, nlp_sub, variant="success")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("MASE theo ngành hàng (Test)")
        by_cat = results.get("mase_by_category", pd.DataFrame())
        if not by_cat.empty:
            cat_col = by_cat.columns[0]
            plot_cols = [c for c in ["MA4", "GRU F2", "GRU F4", "LightGBM F4"] if c in by_cat.columns]
            if plot_cols:
                m_cat = by_cat.melt(id_vars=cat_col, value_vars=plot_cols, var_name="Mô hình", value_name="MASE")
                fig_cat = px.bar(
                    m_cat,
                    x=cat_col,
                    y="MASE",
                    color="Mô hình",
                    barmode="group",
                    color_discrete_sequence=["#94A3B8", "#6366F1", "#059669", "#0891B2"],
                )
                fig_cat.update_layout(**PLOTLY_LAYOUT, xaxis_title="", yaxis_title="MASE (thấp hơn tốt hơn)", height=360)
                st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("Chưa có file `mase_by_category.csv`.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Top mô hình tốt nhất (Ensemble Test MASE)")
        if not ens_df.empty:
            top_ens = ens_df.sort_values("MASE").head(10).copy()
            top_ens["label"] = top_ens["model"] + " " + top_ens["feature_set"].replace("-", "(Baseline)")
            top_ens["Nhóm"] = np.where(
                top_ens["feature_set"] == "-",
                "Baseline",
                np.where(top_ens["feature_set"].isin(["F3", "F4"]), "Có NLP", "Không NLP"),
            )
            fig_rank = px.bar(
                top_ens.iloc[::-1],
                x="MASE",
                y="label",
                orientation="h",
                color="Nhóm",
                text=top_ens.iloc[::-1]["MASE"].map(lambda x: f"{x:.3f}"),
                color_discrete_map={"Không NLP": "#4F46E5", "Có NLP": "#059669", "Baseline": "#D97706"},
            )
            fig_rank.update_traces(textposition="outside")
            fig_rank.update_layout(**PLOTLY_LAYOUT, xaxis_title="Test MASE", yaxis_title="", height=360)
            st.plotly_chart(fig_rank, use_container_width=True)
        else:
            st.info("Chưa có file `metrics_ensemble.csv`.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Thống kê dữ liệu theo từng ngành (từ `weekly_demand.parquet`)")
    eda_tbl = (
        weekly[weekly["launched"] == True]
        .groupby("category")
        .agg(
            so_chuoi=("series_id", "nunique"),
            tong_review=("n_reviews", "sum"),
            tong_verified=("n_verified", "sum"),
            tb_verified_tuan=("n_verified", "mean"),
            ti_le_tuan_0=("n_verified", lambda s: (s == 0).mean() * 100),
        )
        .reset_index()
    )
    eda_tbl.columns = ["Ngành hàng", "Số chuỗi", "Tổng review", "Tổng verified", "TB verified/tuần", "Tỉ lệ tuần = 0 (%)"]
    st.dataframe(
        eda_tbl.style.format(
            {"Tổng review": "{:,.0f}", "Tổng verified": "{:,.0f}", "TB verified/tuần": "{:.2f}", "Tỉ lệ tuần = 0 (%)": "{:.1f}%"}
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 2: DỰ BÁO THEO SẢN PHẨM
# -----------------------------------------------------------------------------
with tab_explorer:
    f1, f2, f3, f4 = st.columns([1.2, 1.0, 2.2, 0.8])
    with f1:
        cats = ["Tất cả"] + sorted(products["category"].dropna().unique().tolist())
        sel_cat = st.selectbox("Ngành hàng", cats)
    with f2:
        sel_vol = st.selectbox("Nhóm lượng review", ["Tất cả", "Nhiều review", "Ít review"])

    prod_filt = products.copy()
    if sel_cat != "Tất cả":
        prod_filt = prod_filt[prod_filt["category"] == sel_cat]
    if sel_vol != "Tất cả":
        prod_filt = prod_filt[prod_filt["volume_group"] == sel_vol]
    prod_filt = prod_filt.sort_values("total", ascending=False)

    if prod_filt.empty:
        st.warning("Không có sản phẩm nào phù hợp bộ lọc hiện tại.")
    else:
        with f3:
            prod_options = prod_filt["series_id"].tolist()
            label_map = {
                r["series_id"]: f"[{r['parent_asin']}] {str(r.get('title', r['series_id']))[:65]}"
                for _, r in prod_filt.iterrows()
            }
            sel_sid = st.selectbox("Sản phẩm", prod_options, format_func=lambda x: label_map.get(x, x))
        with f4:
            horizons = sorted(pred["horizon"].unique().tolist())
            sel_h = st.selectbox("Horizon (tuần)", horizons, index=0)

        p_row = products[products["series_id"] == sel_sid].iloc[0]

        # Chọn mô hình hiển thị
        avail_models = sorted([m for m in pred["model"].unique() if m not in ("Naive", "SeasonalNaive", "MA4")])
        default_idx = avail_models.index("GRU") if "GRU" in avail_models else 0
        m_col1, m_col2 = st.columns([1, 2.5])
        with m_col1:
            sel_model = st.selectbox("Mô hình chính", avail_models, index=default_idx)
        with m_col2:
            show_lines = st.multiselect(
                "Đường dự báo đối chiếu",
                options=["F0", "F2", "F4", "MA4"],
                default=["F0", "F4", "MA4"],
            )

        # Thẻ thông tin sản phẩm
        pm1, pm2, pm3, pm4 = st.columns(4)
        price_val = p_row.get("price")
        price_str = f"${float(price_val):.2f}" if pd.notna(price_val) else "N/A"
        with pm1:
            kpi_card("ASIN / Ngành", str(p_row["parent_asin"]), str(p_row["category"]).replace("_", " "))
        with pm2:
            kpi_card("Giá bán & Rating", price_str, f"Rating TB: {p_row.get('average_rating', 'N/A')} ★")
        with pm3:
            kpi_card("Tổng review verified", f"{int(p_row.get('total', 0)):,}", f"TB tập train: {p_row['reviews_per_week']:.1f} / tuần")
        with pm4:
            v_variant = "success" if p_row["volume_group"] == "Nhiều review" else ""
            kpi_card("Phân nhóm mật độ", p_row["volume_group"], f"Ngưỡng trung vị: {median_vol:.1f} rev/tuần", variant=v_variant)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Biểu đồ dự báo & Khía cạnh NLP
        ch_left, ch_right = st.columns([2.1, 1.0])
        with ch_left:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader(f"Chuỗi nhu cầu thực tế & Dự báo {sel_model} (h = {sel_h})")
            w_s = weekly[weekly["series_id"] == sel_sid].sort_values("week")
            p_s = pred[(pred["series_id"] == sel_sid) & (pred["horizon"] == sel_h)].sort_values("week")

            # Chỉ hiển thị từ 1 năm trước val_end cho dễ quan sát
            val_end = pd.Timestamp(meta.get("val_end", "2022-06-27"))
            w_show = w_s[w_s["week"] >= val_end - pd.Timedelta(weeks=40)]

            fig_ts = go.Figure()
            fig_ts.add_trace(
                go.Scatter(
                    x=w_show["week"],
                    y=w_show["n_verified"],
                    mode="lines+markers",
                    name="Thực tế (Verified)",
                    line=dict(color="#0F172A", width=2.2),
                    marker=dict(size=4),
                )
            )
            color_map = {"F0": "#4F46E5", "F2": "#0891B2", "F4": "#059669", "MA4": "#D97706"}
            for line_key in show_lines:
                if line_key == "MA4":
                    sub = p_s[p_s["model"] == "MA4"]
                    name = "Baseline MA4"
                    dash = "dot"
                else:
                    sub = p_s[(p_s["model"] == sel_model) & (p_s["feature_set"] == line_key)]
                    name = f"{sel_model} {line_key}"
                    dash = "solid"
                if not sub.empty:
                    fig_ts.add_trace(
                        go.Scatter(
                            x=sub["week"],
                            y=sub["y_pred"],
                            mode="lines",
                            name=name,
                            line=dict(color=color_map[line_key], width=2, dash=dash),
                        )
                    )
            fig_ts.add_vline(x=val_end, line_dash="dash", line_color="#94A3B8")
            fig_ts.update_layout(**PLOTLY_LAYOUT, xaxis_title="", yaxis_title="Review verified / tuần", height=390)
            st.plotly_chart(fig_ts, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with ch_right:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Tín hiệu NLP của sản phẩm")
            n_s = nlp[nlp["series_id"] == sel_sid]
            asp_cols = [c for c in n_s.columns if c.startswith("asp_")]
            if not n_s.empty and asp_cols:
                asp_means = (n_s[asp_cols].mean() * 100).sort_values()
                asp_df = pd.DataFrame(
                    {"Khía cạnh": [c.replace("asp_", "").capitalize() for c in asp_means.index], "Tỉ lệ (%)": asp_means.values}
                )
                fig_asp = px.bar(
                    asp_df,
                    x="Tỉ lệ (%)",
                    y="Khía cạnh",
                    orientation="h",
                    text=asp_df["Tỉ lệ (%)"].map(lambda v: f"{v:.1f}%"),
                    color_discrete_sequence=["#0891B2"],
                )
                fig_asp.update_traces(textposition="outside")
                fig_asp.update_layout(**PLOTLY_LAYOUT, xaxis_title="% Review nhắc tới", yaxis_title="", height=270)
                st.plotly_chart(fig_asp, use_container_width=True)

                pos_m = float(n_s["pos_ratio"].mean()) * 100
                neg_m = float(n_s["neg_ratio"].mean()) * 100
                sent_m = float(n_s["sent_mean"].mean())
                st.caption(f"Cảm xúc TB: **{sent_m:+.2f}** • Tích cực: **{pos_m:.1f}%** • Tiêu cực: **{neg_m:.1f}%**")
            else:
                st.info("Không có dữ liệu NLP cho sản phẩm này.")
            st.markdown("</div>", unsafe_allow_html=True)

        # Review tiêu biểu
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Review tiêu biểu (trích từ `review_samples.parquet`)")
        r_s = reviews[reviews["series_id"] == sel_sid].copy()
        if r_s.empty:
            st.info("Không có mẫu review cho sản phẩm này.")
        else:
            rev_filter = st.radio("Lọc cảm xúc", ["Tất cả", "Tích cực (pos)", "Tiêu cực (neg)"], horizontal=True)
            if rev_filter == "Tích cực (pos)":
                r_s = r_s[r_s["label"] == "pos"]
            elif rev_filter == "Tiêu cực (neg)":
                r_s = r_s[r_s["label"] == "neg"]

            asp_r_cols = [c for c in r_s.columns if c.startswith("asp_")]
            rc1, rc2 = st.columns(2)
            for idx_r, (_, rr) in enumerate(r_s.head(8).iterrows()):
                lbl = str(rr.get("label", "pos"))
                badge_txt = f"{'Pos' if lbl == 'pos' else 'Neg'} • sent {float(rr.get('sent', 0)):+.2f}"
                stars = "★" * int(round(float(rr.get("rating", 5))))
                chips = "".join(
                    f'<span class="asp-chip">#{c.replace("asp_", "")}</span>'
                    for c in asp_r_cols
                    if rr.get(c, 0) == 1
                )
                card_html = f"""
                <div class="rev-card {lbl}">
                    <div class="rev-meta">
                        <span class="rev-badge {lbl}">{badge_txt}</span>
                        <span>{stars} • {int(rr.get('helpful_vote', 0))} helpful</span>
                    </div>
                    <div class="rev-title">{str(rr.get('title', ''))[:120]}</div>
                    <div class="rev-text">{str(rr.get('text', ''))[:320]}</div>
                    <div>{chips}</div>
                </div>
                """
                (rc1 if idx_r % 2 == 0 else rc2).markdown(card_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 3: THÍ NGHIỆM ABLATION & V1 -> V2
# -----------------------------------------------------------------------------
with tab_ablation:
    abl_df = results.get("ablation_mase", pd.DataFrame())
    if not abl_df.empty:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Biểu đồ Ablation (Test MASE theo bộ đặc trưng F0 → F4)")
        m_col = abl_df.columns[0]
        fs_cols = [c for c in ["F0", "F1", "F2", "F3", "F4"] if c in abl_df.columns]
        abl_m = abl_df.melt(id_vars=m_col, value_vars=fs_cols, var_name="Bộ đặc trưng", value_name="MASE")
        fig_abl = px.bar(
            abl_m,
            x=m_col,
            y="MASE",
            color="Bộ đặc trưng",
            barmode="group",
            color_discrete_sequence=["#4F46E5", "#818CF8", "#0891B2", "#14B8A6", "#059669"],
        )
        ens_df = results.get("metrics_ensemble", pd.DataFrame())
        if not ens_df.empty and "MA4" in ens_df["model"].values:
            ma4_val = float(ens_df.loc[ens_df["model"] == "MA4", "MASE"].iloc[0])
            fig_abl.add_hline(y=ma4_val, line_dash="dash", line_color="#D97706", annotation_text=f"MA4 ({ma4_val:.3f})")
        fig_abl.update_layout(**PLOTLY_LAYOUT, xaxis_title="", yaxis_title="Test MASE", height=380)
        st.plotly_chart(fig_abl, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    t1, t2 = st.columns(2)
    with t1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("So sánh cải thiện v1 → v2 (`v1_vs_v2.csv`)")
        v12 = results.get("v1_vs_v2", pd.DataFrame())
        if not v12.empty:
            st.dataframe(v12, use_container_width=True, hide_index=True)
        else:
            st.info("Chưa có file `v1_vs_v2.csv`.")
        st.markdown("</div>", unsafe_allow_html=True)

    with t2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Kiểm định Wilcoxon (`significance.csv`)")
        sig_df = results.get("significance", pd.DataFrame())
        if not sig_df.empty:
            st.dataframe(sig_df, use_container_width=True, hide_index=True)
        else:
            st.info("Chưa có file `significance.csv`.")
        st.markdown("</div>", unsafe_allow_html=True)

    comp_df = results.get("model_comparison", pd.DataFrame())
    if not comp_df.empty:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        c_head, c_dl = st.columns([3, 1])
        with c_head:
            st.subheader("Bảng tổng hợp toàn bộ mô hình (`model_comparison.csv`)")
        with c_dl:
            st.download_button(
                "Tải CSV kết quả",
                data=comp_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="model_comparison.csv",
                mime="text/csv",
                use_container_width=True,
            )
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 4: PHÂN TÍCH NLP & MẬT ĐỘ REVIEW
# -----------------------------------------------------------------------------
with tab_nlp:
    vol_df = results.get("volume_split", pd.DataFrame())
    n_left, n_right = st.columns([1.4, 1.0])

    with n_left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Tác động NLP theo mật độ review (Nhiều vs Ít review)")
        if not vol_df.empty:
            delta_cols = [c for c in vol_df.columns if "MASE %" in c]
            if delta_cols:
                fig_vol = px.bar(
                    vol_df,
                    x="model",
                    y=delta_cols[0],
                    color="group",
                    barmode="group",
                    color_discrete_sequence=["#059669", "#E11D48"],
                )
                fig_vol.add_hline(y=0, line_color="#64748B", line_width=1)
                fig_vol.update_layout(
                    **PLOTLY_LAYOUT,
                    xaxis_title="",
                    yaxis_title="Δ MASE F2→F4 (% — âm là NLP cải thiện)",
                    height=350,
                )
                st.plotly_chart(fig_vol, use_container_width=True)
        else:
            st.info("Chưa có file `volume_split.csv`.")
        st.markdown("</div>", unsafe_allow_html=True)

    with n_right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Tần suất 7 khía cạnh NLP toàn tập")
        asp_cols = [c for c in nlp.columns if c.startswith("asp_")]
        if asp_cols:
            g_asp = (nlp[asp_cols].mean() * 100).sort_values()
            g_df = pd.DataFrame(
                {"Khía cạnh": [c.replace("asp_", "").capitalize() for c in g_asp.index], "Tỉ lệ (%)": g_asp.values}
            )
            fig_g = px.bar(
                g_df,
                x="Tỉ lệ (%)",
                y="Khía cạnh",
                orientation="h",
                text=g_df["Tỉ lệ (%)"].map(lambda v: f"{v:.1f}%"),
                color_discrete_sequence=["#4F46E5"],
            )
            fig_g.update_traces(textposition="outside")
            fig_g.update_layout(**PLOTLY_LAYOUT, xaxis_title="% Trung bình theo tuần", yaxis_title="", height=350)
            st.plotly_chart(fig_g, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if not vol_df.empty:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Bảng chi tiết phân tích theo lượng review (`volume_split.csv`)")
        st.dataframe(vol_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
