"""
streamlit_app.py
----------------
Streamlit web dashboard for the Box Office Revenue Predictor.
Run with: streamlit run streamlit_app.py
"""

import sys
import os
import json
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from src.data_loader   import load_data
from src.preprocessing import preprocess
from src.model         import train_and_evaluate

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Box Office Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0a0c14; }
  [data-testid="stSidebar"]          { background: #0f1117; border-right: 1px solid #1e2235; }
  [data-testid="stSidebar"] h1       { color: #e8ecf5; font-size: 16px; }
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

  .metric-card {
    background: #151822; border: 1px solid #1e2235;
    border-radius: 12px; padding: 18px 20px; margin-bottom: 4px;
  }
  .metric-label { font-size: 12px; color: #6b7494; margin-bottom: 6px; }
  .metric-value { font-size: 28px; font-weight: 600; color: #e8ecf5; line-height: 1; }
  .metric-sub   { font-size: 11px; color: #6b7494; margin-top: 5px; }

  .section-title {
    font-size: 20px; font-weight: 600; color: #e8ecf5;
    margin-bottom: 4px; margin-top: 8px;
  }
  .section-sub { font-size: 13px; color: #6b7494; margin-bottom: 20px; }

  .result-box {
    background: linear-gradient(135deg, rgba(79,142,247,0.08), rgba(123,110,247,0.08));
    border: 1px solid rgba(79,142,247,0.25); border-radius: 12px;
    padding: 24px; text-align: center; margin: 16px 0;
  }
  .result-label  { font-size: 13px; color: #6b7494; margin-bottom: 6px; }
  .result-value  { font-size: 38px; font-weight: 700; color: #4f8ef7; font-family: monospace; }
  .result-sub    { font-size: 12px; color: #6b7494; margin-top: 6px; }

  .rev-row {
    display: flex; justify-content: space-between;
    background: #0f1117; border-radius: 8px;
    padding: 10px 16px; margin-bottom: 8px; font-size: 14px;
  }
  .rev-row-label { color: #a0a8c0; }
  .rev-lr  { color: #4f8ef7; font-weight: 600; font-family: monospace; }
  .rev-rf  { color: #f7934f; font-weight: 600; font-family: monospace; }

  .shap-row {
    display: flex; align-items: center; gap: 12px;
    padding: 8px 0; border-bottom: 1px solid #1e2235;
  }
  .shap-rank { font-size: 11px; color: #6b7494; width: 22px; text-align: right; font-family: monospace; }
  .shap-name { font-size: 12px; color: #a0a8c0; width: 200px; font-family: monospace; }
  .shap-track { flex: 1; height: 10px; background: #1a1d27; border-radius: 5px; overflow: hidden; }
  .shap-fill  { height: 100%; border-radius: 5px; background: linear-gradient(90deg, #4f8ef7, #7b6ef7); }
  .shap-val   { font-size: 11px; color: #6b7494; width: 55px; text-align: right; font-family: monospace; }

  .tag {
    display: inline-block; padding: 3px 11px; border-radius: 20px;
    font-size: 12px; border: 1px solid #2a304a; color: #a0a8c0; margin: 3px;
  }
  div[data-testid="stMetric"] {
    background: #151822; border: 1px solid #1e2235;
    border-radius: 12px; padding: 16px;
  }
  div[data-testid="stMetric"] label { color: #6b7494 !important; }
  div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #e8ecf5 !important; }
</style>
""", unsafe_allow_html=True)


# ── Plot style ────────────────────────────────────────────────────────────────
def set_plot_style():
    plt.rcParams.update({
        "figure.facecolor":  "#0f1117",
        "axes.facecolor":    "#1a1d27",
        "axes.edgecolor":    "#2e3347",
        "axes.labelcolor":   "#a0a8c0",
        "axes.titlecolor":   "#e8ecf5",
        "xtick.color":       "#6b7494",
        "ytick.color":       "#6b7494",
        "text.color":        "#a0a8c0",
        "grid.color":        "#2e3347",
        "grid.linewidth":    0.8,
    })


# ── Load & cache pipeline ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    raw_df             = load_data("data")
    clean_df, model_df = preprocess(raw_df)
    results            = train_and_evaluate(model_df)
    return clean_df, model_df, results


# ── Format money ─────────────────────────────────────────────────────────────
def fmt(v):
    if v >= 1e9: return f"${v/1e9:.2f}B"
    if v >= 1e6: return f"${v/1e6:.0f}M"
    return f"${v:,.0f}"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 Box Office Predictor")
    st.markdown("<div style='font-size:12px;color:#6b7494;margin-bottom:20px;'>TMDB 5000 Dataset · ML Dashboard</div>", unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["📊 Overview", "🔍 Exploration", "🤖 Models", "💡 Explainability", "🎯 Predict", "📋 About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("<div style='font-size:11px;color:#6b7494;line-height:1.8;'>pandas · numpy · scikit-learn<br>matplotlib · SHAP · Streamlit<br><br>Linear Regression · Random Forest</div>", unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("🎬 Loading data and training models — please wait..."):
    try:
        clean_df, model_df, results = load_pipeline()
        pipeline_ok = True
    except FileNotFoundError as e:
        pipeline_ok = False
        err_msg = str(e)

if not pipeline_ok:
    st.error(f"**Data files not found!**\n\n{err_msg}")
    st.info("Place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` in the `data/` folder, then refresh.")
    st.stop()

lr_metrics = results["Linear Regression"]["metrics"]
rf_metrics = results["Random Forest"]["metrics"]


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown('<div class="section-title">Dashboard Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Key metrics from the trained models and TMDB dataset</div>', unsafe_allow_html=True)

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset size",         f"{len(model_df):,}",        "films after cleaning")
    c2.metric("Linear Regression R²", f"{lr_metrics['r2']:.4f}",   f"CV: {results['Linear Regression']['cv_r2_mean']:.4f}")
    c3.metric("Random Forest R²",     f"{rf_metrics['r2']:.4f}",   f"CV: {results['Random Forest']['cv_r2_mean']:.4f}")
    c4.metric("Features engineered",  "14",                         "from raw TMDB data")

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Revenue distribution**")
        set_plot_style()
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        fig.patch.set_facecolor("#0f1117")
        for ax in axes:
            ax.set_facecolor("#1a1d27")
            ax.grid(True, color="#2e3347", linewidth=0.7)
            for s in ax.spines.values(): s.set_edgecolor("#2e3347")
        axes[0].hist(clean_df["revenue"]/1e6, bins=50, color="#4f8ef7", edgecolor="none", alpha=0.85)
        axes[0].set_title("Raw revenue", fontsize=11)
        axes[0].set_xlabel("USD millions")
        axes[1].hist(clean_df["log_revenue"], bins=50, color="#f7934f", edgecolor="none", alpha=0.85)
        axes[1].set_title("Log-transformed", fontsize=11)
        axes[1].set_xlabel("log(1 + revenue)")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("**Budget vs revenue** *(coloured by popularity)*")
        set_plot_style()
        fig, ax = plt.subplots(figsize=(6, 3.8))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#1a1d27")
        ax.grid(True, color="#2e3347", linewidth=0.7)
        for s in ax.spines.values(): s.set_edgecolor("#2e3347")
        sc = ax.scatter(clean_df["log_budget"], clean_df["log_revenue"],
                        alpha=0.3, s=12, c=clean_df["popularity"],
                        cmap="plasma", edgecolors="none")
        cb = fig.colorbar(sc, ax=ax, shrink=0.8)
        cb.set_label("Popularity", color="#a0a8c0")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="#6b7494")
        m, b = np.polyfit(clean_df["log_budget"], clean_df["log_revenue"], 1)
        xr = np.linspace(clean_df["log_budget"].min(), clean_df["log_budget"].max(), 200)
        ax.plot(xr, m*xr+b, color="#f74f4f", linewidth=1.6, label=f"Trend (slope={m:.2f})")
        ax.legend(fontsize=9, facecolor="#1a1d27", edgecolor="#2e3347", labelcolor="#a0a8c0")
        ax.set_xlabel("log(1 + budget)")
        ax.set_ylabel("log(1 + revenue)")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPLORATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Exploration":
    st.markdown('<div class="section-title">Data Exploration</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Visual analysis of the TMDB 5000 dataset</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Average revenue by genre**")
        rows = []
        for _, row in clean_df.iterrows():
            try:
                for g in json.loads(row["genres"]):
                    rows.append({"genre": g["name"], "revenue": row["revenue"]})
            except: pass

        if rows:
            gdf = pd.DataFrame(rows)
            top = (gdf.groupby("genre")["revenue"]
                      .agg(["mean","count"])
                      .query("count >= 20")
                      .sort_values("mean", ascending=True)
                      .tail(12))
            set_plot_style()
            fig, ax = plt.subplots(figsize=(6, 4.5))
            fig.patch.set_facecolor("#0f1117")
            ax.set_facecolor("#1a1d27")
            ax.grid(True, color="#2e3347", linewidth=0.7, axis="x")
            for s in ax.spines.values(): s.set_edgecolor("#2e3347")
            colors = plt.cm.plasma(np.linspace(0.2, 0.85, len(top)))
            bars = ax.barh(top.index, top["mean"]/1e6, color=colors, edgecolor="none", height=0.65)
            ax.set_xlabel("Average revenue (USD millions)")
            for bar, val in zip(bars, top["mean"]/1e6):
                ax.text(val+1, bar.get_y()+bar.get_height()/2, f"${val:,.0f}M",
                        va="center", fontsize=8, color="#a0a8c0")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

    with col2:
        st.markdown("**Median revenue by release month**")
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly = clean_df.groupby("release_month")["revenue"].median().reindex(range(1,13))
        vals = monthly.values / 1e6
        set_plot_style()
        fig, ax = plt.subplots(figsize=(6, 4.5))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#1a1d27")
        ax.grid(True, color="#2e3347", linewidth=0.7, axis="y")
        for s in ax.spines.values(): s.set_edgecolor("#2e3347")
        bar_colors = ["#f7934f" if v == np.nanmax(vals) else "#4f8ef7" for v in vals]
        ax.bar(range(1,13), vals, color=bar_colors, edgecolor="none", width=0.65)
        ax.set_xticks(range(1,13))
        ax.set_xticklabels(months)
        ax.set_ylabel("Median revenue (USD millions)")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")
    st.markdown("**Correlation heatmap**")
    import seaborn as sns
    num_cols = ["log_revenue","log_budget","runtime","popularity","vote_average",
                "vote_count","genres_count","cast_size","crew_size","release_year","release_month"]
    corr = clean_df[[c for c in num_cols if c in clean_df.columns]].corr()
    set_plot_style()
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax,
                annot_kws={"size": 9, "color": "#e8ecf5"},
                cbar_kws={"shrink": 0.8})
    ax.tick_params(colors="#a0a8c0")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODELS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Models":
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Linear regression vs random forest — evaluation metrics and predictions</div>', unsafe_allow_html=True)

    # Metrics table
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Linear Regression** *(primary)*")
        st.metric("R²",   f"{lr_metrics['r2']:.4f}")
        st.metric("MAE",  f"{lr_metrics['mae']:.4f}",  "log-revenue space")
        st.metric("RMSE", f"{lr_metrics['rmse']:.4f}", "log-revenue space")
        st.metric("CV R² (5-fold)", f"{results['Linear Regression']['cv_r2_mean']:.4f} ± {results['Linear Regression']['cv_r2_std']:.4f}")
    with col2:
        st.markdown("**Random Forest** *(comparison)*")
        st.metric("R²",   f"{rf_metrics['r2']:.4f}")
        st.metric("MAE",  f"{rf_metrics['mae']:.4f}",  "log-revenue space")
        st.metric("RMSE", f"{rf_metrics['rmse']:.4f}", "log-revenue space")
        st.metric("CV R² (5-fold)", f"{results['Random Forest']['cv_r2_mean']:.4f} ± {results['Random Forest']['cv_r2_std']:.4f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Actual vs predicted (log revenue)**")
        set_plot_style()
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.patch.set_facecolor("#0f1117")
        for ax, key, color in zip(axes, ["Linear Regression","Random Forest"], ["#4f8ef7","#f7934f"]):
            ax.set_facecolor("#1a1d27")
            ax.grid(True, color="#2e3347", linewidth=0.7)
            for s in ax.spines.values(): s.set_edgecolor("#2e3347")
            y_test = results[key]["y_test"]
            y_pred = results[key]["y_pred"]
            ax.scatter(y_test, y_pred, alpha=0.3, s=10, color=color, edgecolors="none")
            lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
            ax.plot(lims, lims, "r--", linewidth=1.3, alpha=0.7)
            ax.set_title(key, fontsize=10)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("**Metric comparison**")
        set_plot_style()
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#1a1d27")
        ax.grid(True, color="#2e3347", linewidth=0.7, axis="y")
        for s in ax.spines.values(): s.set_edgecolor("#2e3347")
        metrics = ["R²", "MAE", "RMSE"]
        lr_v = [lr_metrics["r2"], lr_metrics["mae"], lr_metrics["rmse"]]
        rf_v = [rf_metrics["r2"], rf_metrics["mae"], rf_metrics["rmse"]]
        x = np.arange(3)
        w = 0.32
        b1 = ax.bar(x-w/2, lr_v, w, label="Linear Reg.", color="#4f8ef7", edgecolor="none")
        b2 = ax.bar(x+w/2, rf_v, w, label="Random Forest", color="#f7934f", edgecolor="none")
        for bar, v in list(zip(b1,lr_v))+list(zip(b2,rf_v)):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f"{v:.3f}", ha="center", fontsize=9, color="#a0a8c0")
        ax.set_xticks(x); ax.set_xticklabels(metrics)
        ax.legend(fontsize=9, facecolor="#1a1d27", edgecolor="#2e3347", labelcolor="#a0a8c0")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Explainability":
    st.markdown('<div class="section-title">SHAP Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Which features drive box office revenue predictions the most</div>', unsafe_allow_html=True)

    lr      = results["Linear Regression"]
    coefs   = lr["model"].coef_
    names   = lr["feature_names"]
    pairs   = sorted(zip(names, np.abs(coefs)), key=lambda x: x[1], reverse=True)
    max_val = pairs[0][1]

    st.markdown("**Feature importance** *(absolute linear regression coefficients)*")
    bars_html = ""
    for i, (name, val) in enumerate(pairs):
        pct = val / max_val * 100
        bars_html += f"""
        <div class="shap-row">
          <div class="shap-rank">{i+1}</div>
          <div class="shap-name">{name}</div>
          <div class="shap-track"><div class="shap-fill" style="width:{pct:.1f}%"></div></div>
          <div class="shap-val">{val:.4f}</div>
        </div>"""
    st.markdown(bars_html, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**What each feature means**")
        descriptions = {
            "log_budget":                    "Production budget (log-transformed)",
            "popularity":                    "TMDB popularity score at release",
            "vote_count":                    "Number of user votes on TMDB",
            "vote_average":                  "Average user rating (1–10)",
            "runtime":                       "Film length in minutes",
            "release_year":                  "Year of theatrical release",
            "cast_size":                     "Number of credited cast members",
            "crew_size":                     "Number of crew members",
            "genres_count":                  "Number of genres the film belongs to",
            "has_homepage":                  "Whether the film has an official website",
            "release_month":                 "Month of theatrical release",
            "keywords_count":                "Number of associated keywords",
            "spoken_languages_count":        "Number of spoken languages in the film",
            "production_companies_count":    "Number of production companies involved",
        }
        for name, val in pairs:
            desc = descriptions.get(name, "")
            st.markdown(f"**`{name}`** — {desc}")

    with col2:
        st.markdown("**Key insight**")
        st.info("💰 **Budget is the strongest predictor** of box office revenue by far.\n\nPopularity and vote count reflect audience awareness and interest.\n\nRuntime, release timing, and cast size also contribute meaningfully to the prediction.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Predict":
    st.markdown('<div class="section-title">Revenue Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Enter film details below to estimate box office revenue</div>', unsafe_allow_html=True)

    col_form, col_result = st.columns([3, 2])

    with col_form:
        st.markdown("**Film details**")
        c1, c2 = st.columns(2)
        with c1:
            budget    = st.number_input("Budget (USD millions)", min_value=0.1, max_value=500.0, value=80.0, step=1.0)
            runtime   = st.number_input("Runtime (minutes)",     min_value=60,  max_value=240,   value=110)
            pop       = st.number_input("TMDB popularity score", min_value=1.0, max_value=500.0, value=30.0)
            vote_avg  = st.number_input("Vote average (1–10)",   min_value=1.0, max_value=10.0,  value=7.0, step=0.1)
            vote_cnt  = st.number_input("Expected vote count",   min_value=10,  max_value=50000, value=2000, step=100)
        with c2:
            genres    = st.number_input("Number of genres",  min_value=1, max_value=8,   value=2)
            cast      = st.number_input("Cast size",         min_value=1, max_value=200, value=25)
            crew      = st.number_input("Crew size",         min_value=1, max_value=500, value=60)
            month     = st.selectbox("Release month", options=list(range(1,13)),
                                     format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun",
                                                             "Jul","Aug","Sep","Oct","Nov","Dec"][x-1],
                                     index=5)
            year      = st.number_input("Release year", min_value=1990, max_value=2030, value=2024)

        predict_btn = st.button("✦ Predict Box Office Revenue", type="primary", use_container_width=True)

    with col_result:
        st.markdown("**Prediction result**")
        if predict_btn:
            features = np.array([[
                np.log1p(budget * 1e6), float(runtime), float(pop),
                float(vote_avg), float(vote_cnt), float(genres),
                float(cast), float(crew), 1.0,
                float(month), float(year), 1.0, 2.0, 10.0
            ]])

            lr_pipeline = results["Linear Regression"]["pipeline"]
            rf_model    = results["Random Forest"]["model"]

            log_lr = float(lr_pipeline.predict(features)[0])
            log_rf = float(rf_model.predict(features)[0])
            rev_lr = float(np.expm1(log_lr))
            rev_rf = float(np.expm1(log_rf))
            avg    = (rev_lr + rev_rf) / 2

            st.markdown(f"""
            <div class="result-box">
              <div class="result-label">Ensemble average estimate</div>
              <div class="result-value">{fmt(avg)}</div>
              <div class="result-sub">Average of both models</div>
            </div>
            <div class="rev-row">
              <span class="rev-row-label">Linear Regression</span>
              <span class="rev-lr">{fmt(rev_lr)}</span>
            </div>
            <div class="rev-row">
              <span class="rev-row-label">Random Forest</span>
              <span class="rev-rf">{fmt(rev_rf)}</span>
            </div>
            """, unsafe_allow_html=True)

            st.caption("Predictions are based on patterns learned from the TMDB 5000 dataset. Treat as estimates.")
        else:
            st.markdown("""
            <div style='text-align:center;padding:40px 20px;color:#6b7494;'>
              <div style='font-size:40px;margin-bottom:12px;opacity:0.4;'>🎬</div>
              <p style='font-size:13px;'>Fill in the film details and click predict</p>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋 About":
    st.markdown('<div class="section-title">About This Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">A complete machine learning application for box office revenue prediction</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Dataset")
        st.markdown("**TMDB 5000 Movie Dataset** from Kaggle — two CSVs merged on film ID. Contains metadata, cast, crew, genres, budget, revenue, and ratings for ~5000 films.")
        st.markdown('<div>'\
            '<span class="tag">5000 films</span>'\
            '<span class="tag">tmdb_5000_movies.csv</span>'\
            '<span class="tag">tmdb_5000_credits.csv</span>'\
            '</div>', unsafe_allow_html=True)

        st.markdown("### Tech Stack")
        st.markdown('<div>'\
            '<span class="tag">Python</span><span class="tag">pandas</span>'\
            '<span class="tag">numpy</span><span class="tag">scikit-learn</span>'\
            '<span class="tag">matplotlib</span><span class="tag">seaborn</span>'\
            '<span class="tag">SHAP</span><span class="tag">Streamlit</span>'\
            '</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("### Pipeline Steps")
        steps = [
            "Load & merge both CSV datasets",
            "Remove duplicates and zero-revenue films",
            "Parse JSON columns (genres, cast, crew)",
            "Engineer 14 predictive features",
            "Log-transform skewed revenue target",
            "80/20 train/test split",
            "Train Linear Regression + Random Forest",
            "Evaluate with R², MAE, RMSE, 5-fold CV",
            "SHAP explainability analysis",
        ]
        for i, s in enumerate(steps, 1):
            st.markdown(f"**{i}.** {s}")

        st.markdown("### Models")
        st.markdown("""
        | Model | Role |
        |---|---|
        | **Linear Regression** | Primary — interpretable baseline |
        | **Random Forest** | Comparison — non-linear patterns |
        """)
