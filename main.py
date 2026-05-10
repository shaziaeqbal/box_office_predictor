"""
main.py
-------
Entry point for the Box Office Revenue Predictor pipeline.

Usage:
    python main.py

Steps:
    1. Load & merge TMDB datasets
    2. Preprocess & engineer features
    3. EDA visualisations
    4. Train Linear Regression + Random Forest
    5. SHAP explainability analysis
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Allow imports from src/
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader      import load_data
from src.preprocessing    import preprocess
from src.visualizations   import run_eda, plot_model_comparison, plot_actual_vs_predicted
from src.model            import train_and_evaluate
from src.explainability   import run_shap_analysis


# ---------------------------------------------------------------------------
# Pretty print helpers
# ---------------------------------------------------------------------------

DIVIDER = "=" * 60

def banner(text: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  🎬  {text}")
    print(DIVIDER)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    banner("Box Office Revenue Predictor — TMDB Dataset")

    # ── Step 1: Load data ──────────────────────────────────────────────
    raw_df = load_data(data_dir="data")

    # ── Step 2: Preprocess ─────────────────────────────────────────────
    clean_df, model_df = preprocess(raw_df)

    # ── Step 3: EDA visualisations ─────────────────────────────────────
    run_eda(clean_df)

    # ── Step 4: Train & evaluate models ───────────────────────────────
    results = train_and_evaluate(model_df)

    # Comparison plots
    metric_summary = {
        model: {
            "r2":  data["metrics"]["r2"],
            "mae": data["metrics"]["mae"],
            "rmse": data["metrics"]["rmse"],
        }
        for model, data in results.items()
    }
    plot_model_comparison(metric_summary)

    lr_data = results["Linear Regression"]
    rf_data = results["Random Forest"]
    plot_actual_vs_predicted(
        lr_data["y_test"],
        lr_data["y_pred"],
        rf_data["y_pred"],
    )

    # ── Step 5: SHAP explainability ────────────────────────────────────
    run_shap_analysis(results)

    # ── Summary ────────────────────────────────────────────────────────
    banner("Pipeline Complete ✅")
    print("\n  📊 Results Summary\n")
    for model_name, data in results.items():
        m = data["metrics"]
        print(f"  {model_name}")
        print(f"    R²   : {m['r2']:.4f}")
        print(f"    MAE  : {m['mae']:.4f}  (in log-revenue space)")
        print(f"    RMSE : {m['rmse']:.4f}  (in log-revenue space)")
        print(f"    CV R²: {data['cv_r2_mean']:.4f} ± {data['cv_r2_std']:.4f}")
        print()

    print(f"  📁 All plots saved to: outputs/")
    print(f"  💾 Model files saved to: models/\n")


if __name__ == "__main__":
    main()
