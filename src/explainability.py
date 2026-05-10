"""
explainability.py
-----------------
SHAP-based model explainability for both Linear Regression and Random Forest.
Produces summary plot, feature importance bar chart, and waterfall chart.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("      [WARNING] SHAP not installed. Run: pip install shap")
    print("      SHAP plots will be skipped. All other outputs will still be generated.")

OUTPUT_DIR = "outputs"
BG_COLOR   = "#F8F9FA"


def _save(fig, filename: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"      Saved → {path}")


def run_shap_analysis(results: dict) -> None:
    """
    Generate SHAP explanations for both models.

    Args:
        results: Output dict from model.train_and_evaluate().
    """
    print("[5/5] Running SHAP explainability...")

    if not SHAP_AVAILABLE:
        print("      Skipping SHAP analysis — install with: pip install shap")
        return

    # ------------------------------------------------------------------ #
    # SHAP for Linear Regression
    # ------------------------------------------------------------------ #
    lr_data = results["Linear Regression"]
    lr_model = lr_data["model"]
    feature_names = lr_data["feature_names"]

    X_test_scaled_lr = lr_data["X_test"]

    explainer_lr = shap.LinearExplainer(lr_model, lr_data["X_train"], feature_perturbation="interventional")
    shap_values_lr = explainer_lr.shap_values(X_test_scaled_lr)

    # --- 08: SHAP Summary Plot (Linear Regression) ---
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG_COLOR)
    shap.summary_plot(
        shap_values_lr,
        X_test_scaled_lr,
        feature_names=feature_names,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP Summary Plot — Linear Regression", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    _save(plt.gcf(), "08_shap_summary_lr.png")

    # --- 09: SHAP Bar Chart (mean |SHAP|) for LR ---
    mean_abs_shap = np.abs(shap_values_lr).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.grid(True, color="#E0E0E0", linewidth=0.8)
    bars = ax.barh(
        [feature_names[i] for i in sorted_idx[::-1]],
        mean_abs_shap[sorted_idx[::-1]],
        color="#4C72B0", edgecolor="white"
    )
    ax.set_title("Mean |SHAP Value| — Linear Regression\n(Global Feature Importance)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    for bar, val in zip(bars, mean_abs_shap[sorted_idx[::-1]]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    _save(fig, "09_shap_bar_lr.png")

    # ------------------------------------------------------------------ #
    # SHAP for Random Forest (TreeExplainer)
    # ------------------------------------------------------------------ #
    rf_data   = results["Random Forest"]
    rf_model  = rf_data["model"]
    X_test_rf = rf_data["X_test"]

    # Use a background sample for speed
    background_size = min(100, len(rf_data["X_train"]))
    np.random.seed(42)
    background_idx = np.random.choice(len(rf_data["X_train"]), background_size, replace=False)
    background     = rf_data["X_train"][background_idx]

    explainer_rf   = shap.TreeExplainer(rf_model, data=background, feature_perturbation="interventional")
    shap_values_rf = explainer_rf.shap_values(X_test_rf[:200])   # subset for speed

    # --- 10: SHAP Summary Plot (Random Forest) ---
    shap.summary_plot(
        shap_values_rf,
        X_test_rf[:200],
        feature_names=feature_names,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP Summary Plot — Random Forest", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    _save(plt.gcf(), "10_shap_summary_rf.png")

    # --- 11: SHAP Waterfall for a single prediction (Random Forest) ---
    sample_idx = 0
    shap_exp = shap.Explanation(
        values        = shap_values_rf[sample_idx],
        base_values   = explainer_rf.expected_value,
        data          = X_test_rf[sample_idx],
        feature_names = feature_names,
    )

    fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG_COLOR)
    shap.waterfall_plot(shap_exp, show=False, max_display=14)
    plt.title(f"SHAP Waterfall — Random Forest (Sample #{sample_idx})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(plt.gcf(), "11_shap_waterfall_rf.png")

    print("      SHAP analysis complete.")
