"""
visualizations.py
-----------------
Exploratory Data Analysis (EDA) plots using matplotlib and seaborn.
All plots are saved to the outputs/ directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter


# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

PALETTE       = "viridis"
ACCENT        = "#4C72B0"
ACCENT2       = "#DD8452"
BG_COLOR      = "#F8F9FA"
GRID_COLOR    = "#E0E0E0"
TITLE_SIZE    = 15
LABEL_SIZE    = 12
TICK_SIZE     = 10
OUTPUT_DIR    = "outputs"

def _save(fig: plt.Figure, filename: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"      Saved → {path}")


def _base_fig(nrows=1, ncols=1, figsize=(10, 5)):
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, facecolor=BG_COLOR)
    if isinstance(ax, np.ndarray):
        for a in ax.flat:
            a.set_facecolor(BG_COLOR)
            a.grid(True, color=GRID_COLOR, linewidth=0.8)
    else:
        ax.set_facecolor(BG_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    return fig, ax


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_revenue_distribution(df: pd.DataFrame) -> None:
    """01 — Distribution of box office revenue (raw + log scale)."""
    fig, axes = _base_fig(1, 2, figsize=(14, 5))
    fig.suptitle("Box Office Revenue Distribution", fontsize=TITLE_SIZE, fontweight="bold", y=1.02)

    # Raw revenue
    axes[0].hist(df["revenue"] / 1e6, bins=60, color=ACCENT, edgecolor="white", linewidth=0.4)
    axes[0].set_title("Raw Revenue", fontsize=LABEL_SIZE)
    axes[0].set_xlabel("Revenue (USD Millions)", fontsize=LABEL_SIZE)
    axes[0].set_ylabel("Count", fontsize=LABEL_SIZE)
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}M"))

    # Log revenue
    axes[1].hist(df["log_revenue"], bins=60, color=ACCENT2, edgecolor="white", linewidth=0.4)
    axes[1].set_title("Log-Transformed Revenue", fontsize=LABEL_SIZE)
    axes[1].set_xlabel("log(1 + Revenue)", fontsize=LABEL_SIZE)
    axes[1].set_ylabel("Count", fontsize=LABEL_SIZE)

    plt.tight_layout()
    _save(fig, "01_revenue_distribution.png")


def plot_budget_vs_revenue(df: pd.DataFrame) -> None:
    """02 — Scatter: budget vs revenue (log scale)."""
    fig, ax = _base_fig(figsize=(10, 6))

    sc = ax.scatter(
        df["log_budget"], df["log_revenue"],
        alpha=0.45, s=18, c=df["popularity"],
        cmap=PALETTE, edgecolors="none"
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label("Popularity Score", fontsize=TICK_SIZE)

    # Trend line
    m, b = np.polyfit(df["log_budget"], df["log_revenue"], 1)
    x_line = np.linspace(df["log_budget"].min(), df["log_budget"].max(), 200)
    ax.plot(x_line, m * x_line + b, color="crimson", linewidth=1.8, label=f"Trend (slope={m:.2f})")
    ax.legend(fontsize=TICK_SIZE)

    ax.set_title("Budget vs Revenue (Log Scale)", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("log(1 + Budget)", fontsize=LABEL_SIZE)
    ax.set_ylabel("log(1 + Revenue)", fontsize=LABEL_SIZE)

    plt.tight_layout()
    _save(fig, "02_budget_vs_revenue.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """03 — Correlation heatmap of numeric features."""
    num_cols = [
        "log_revenue", "log_budget", "runtime", "popularity",
        "vote_average", "vote_count", "genres_count", "cast_size",
        "crew_size", "release_year", "release_month",
        "spoken_languages_count", "production_companies_count",
    ]
    available = [c for c in num_cols if c in df.columns]
    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(13, 10), facecolor=BG_COLOR)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, ax=ax,
        annot_kws={"size": 8}, cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=TITLE_SIZE, fontweight="bold")
    plt.tight_layout()
    _save(fig, "03_correlation_heatmap.png")


def plot_top_genres(df: pd.DataFrame) -> None:
    """04 — Average revenue by top genres."""
    # Explode genres list into individual rows
    genres_data = []
    for _, row in df.iterrows():
        for genre in row.get("genres_list", []):
            genres_data.append({"genre": genre, "revenue": row["revenue"]})

    if not genres_data:
        print("      Skipping genre plot (no genre data parsed).")
        return

    gdf = pd.DataFrame(genres_data)
    genre_rev = (
        gdf.groupby("genre")["revenue"]
        .agg(["mean", "count"])
        .query("count >= 30")
        .sort_values("mean", ascending=False)
        .head(15)
    )

    fig, ax = _base_fig(figsize=(12, 6))
    bars = ax.barh(genre_rev.index[::-1], genre_rev["mean"][::-1] / 1e6, color=ACCENT, edgecolor="white")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}M"))
    ax.set_title("Average Box Office Revenue by Genre (min. 30 films)", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("Average Revenue (USD Millions)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Genre", fontsize=LABEL_SIZE)

    for bar, val in zip(bars, genre_rev["mean"][::-1] / 1e6):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2, f"${val:,.0f}M",
                va="center", fontsize=8, color="black")

    plt.tight_layout()
    _save(fig, "04_top_genres.png")


def plot_revenue_by_month(df: pd.DataFrame) -> None:
    """05 — Median revenue by release month."""
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly = df.groupby("release_month")["revenue"].median().reindex(range(1, 13))

    fig, ax = _base_fig(figsize=(11, 5))
    bars = ax.bar(range(1, 13), monthly / 1e6, color=ACCENT, edgecolor="white", width=0.7)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names, fontsize=TICK_SIZE)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}M"))
    ax.set_title("Median Box Office Revenue by Release Month", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("Release Month", fontsize=LABEL_SIZE)
    ax.set_ylabel("Median Revenue (USD Millions)", fontsize=LABEL_SIZE)

    for bar, val in zip(bars, monthly / 1e6):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5, f"${val:,.0f}M",
                    ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    _save(fig, "05_revenue_by_month.png")


def plot_model_comparison(results: dict) -> None:
    """06 — Bar chart comparing model metrics (R², MAE, RMSE)."""
    models  = list(results.keys())
    metrics = ["R2", "MAE (log)", "RMSE (log)"]
    values  = {
        m: [results[m]["r2"], results[m]["mae"], results[m]["rmse"]]
        for m in models
    }

    x = np.arange(len(metrics))
    width = 0.35
    colors = [ACCENT, ACCENT2]

    fig, ax = _base_fig(figsize=(10, 6))
    for i, (model, color) in enumerate(zip(models, colors)):
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, values[model], width, label=model, color=color, edgecolor="white")
        for bar, v in zip(bars, values[model]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=LABEL_SIZE)
    ax.set_title("Model Performance Comparison", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_ylabel("Score", fontsize=LABEL_SIZE)
    ax.legend(fontsize=TICK_SIZE)

    plt.tight_layout()
    _save(fig, "06_model_comparison.png")


def plot_actual_vs_predicted(y_test, y_pred_lr, y_pred_rf) -> None:
    """07 — Actual vs Predicted (log revenue) for both models."""
    fig, axes = _base_fig(1, 2, figsize=(14, 6))
    fig.suptitle("Actual vs Predicted Log Revenue", fontsize=TITLE_SIZE, fontweight="bold")

    for ax, y_pred, label, color in zip(
        axes,
        [y_pred_lr, y_pred_rf],
        ["Linear Regression", "Random Forest"],
        [ACCENT, ACCENT2]
    ):
        ax.scatter(y_test, y_pred, alpha=0.35, s=14, color=color, edgecolors="none")
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1.4, label="Perfect fit")
        ax.set_title(label, fontsize=LABEL_SIZE)
        ax.set_xlabel("Actual log(Revenue)", fontsize=LABEL_SIZE)
        ax.set_ylabel("Predicted log(Revenue)", fontsize=LABEL_SIZE)
        ax.legend(fontsize=TICK_SIZE)

    plt.tight_layout()
    _save(fig, "07_actual_vs_predicted.png")


def run_eda(df: pd.DataFrame) -> None:
    """Run all EDA plots in sequence."""
    print("[3/5] Generating EDA visualisations...")
    plot_revenue_distribution(df)
    plot_budget_vs_revenue(df)
    plot_correlation_heatmap(df)
    plot_top_genres(df)
    plot_revenue_by_month(df)
