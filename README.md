# 🎬 Box Office Revenue Predictor

A machine learning web application that predicts box office revenue for films using the TMDB 5000 Movie Dataset. Built with Python and Streamlit, the project covers the complete data science workflow — from raw data cleaning and feature engineering through to model training, evaluation, and explainability — presented in an interactive browser-based dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=flat-square&logo=scikit-learn)
![pandas](https://img.shields.io/badge/pandas-2.2-150458?style=flat-square&logo=pandas)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Overview

Predicting box office revenue is a genuine challenge in the film industry — budgets, cast size, genre, and release timing all play a role. This project builds a regression pipeline on the TMDB 5000 dataset to quantify those relationships, compare two model types, and explain what drives each prediction using SHAP values.

The result is a fully interactive Streamlit dashboard where you can explore the data, review model performance, and estimate revenue for any hypothetical film.


---

## Dashboard Pages

| Page | Description |
|---|---|
| **Overview** | Key metrics, revenue distribution, budget vs revenue scatter |
| **Exploration** | Genre analysis, seasonal trends, correlation heatmap |
| **Models** | Actual vs predicted charts, R², MAE, RMSE comparison |
| **Explainability** | SHAP feature importance rankings with descriptions |
| **Predict** | Enter any film's details and get a live revenue estimate |
| **About** | Project summary, tech stack, pipeline breakdown |

---

## Project Structure

```
box_office_predictor/
│
├── streamlit_app.py                  # Web dashboard — run this
├── main.py                           # CLI pipeline (no UI)
├── requirements.txt
│
├── data/                             # Add your Kaggle CSVs here
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
│
├── src/
│   ├── data_loader.py                # Load and merge datasets
│   ├── preprocessing.py              # Cleaning and feature engineering
│   ├── visualizations.py             # Matplotlib/Seaborn plots
│   ├── model.py                      # Model training and evaluation
│   └── explainability.py             # SHAP analysis
│
├── outputs/                          # Auto-generated chart images
└── models/                           # Saved model .pkl files
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/shaziaeqbal/box-office-predictor.git
cd box-office-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Go to [TMDB 5000 Movie Dataset on Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata), download the files, and place both CSVs inside the `data/` folder:

```
data/
├── tmdb_5000_movies.csv
└── tmdb_5000_credits.csv
```

**4. Launch the dashboard**
```bash
streamlit run streamlit_app.py
```

Open your browser at **http://localhost:8501**

> To run the CLI pipeline instead (saves charts to `outputs/`):
> ```bash
> python main.py
> ```

---

## Features Used

The following 14 features were engineered from the raw TMDB data:

| Feature | Description |
|---|---|
| `log_budget` | Production budget (USD, log-transformed to reduce skew) |
| `runtime` | Film duration in minutes |
| `popularity` | TMDB popularity score at time of release |
| `vote_average` | Average user rating on TMDB (1–10) |
| `vote_count` | Total number of user ratings |
| `genres_count` | Number of genres the film belongs to |
| `cast_size` | Number of credited cast members |
| `crew_size` | Number of crew members listed |
| `has_homepage` | Binary flag — whether the film has an official website |
| `release_month` | Month of theatrical release |
| `release_year` | Year of theatrical release |
| `spoken_languages_count` | Number of spoken languages in the film |
| `production_companies_count` | Number of production companies involved |
| `keywords_count` | Number of associated keywords on TMDB |

---

## Models

| Model | Role | Notes |
|---|---|---|
| Linear Regression | Primary | Interpretable baseline with StandardScaler |
| Random Forest | Comparison | 300 estimators, captures non-linear patterns |

Both models are trained on an 80/20 train-test split and evaluated with 5-fold cross-validation. Predictions are made in log-revenue space and converted back for readability.

**Evaluation metrics:** R², MAE, RMSE, CV R²

---

## Pipeline

```
Raw CSVs  →  Merge  →  Clean  →  Feature Engineering  →  Train/Test Split
                                                                  ↓
                                           Linear Regression  +  Random Forest
                                                                  ↓
                                              R²  |  MAE  |  RMSE  |  CV Score
                                                                  ↓
                                                    SHAP Explainability
                                                                  ↓
                                                  Streamlit Dashboard
```

---

## Tech Stack

| Library | Purpose |
|---|---|
| pandas | Data loading, cleaning, merging |
| numpy | Numerical operations, log transforms |
| scikit-learn | Model training, evaluation, pipelines |
| matplotlib / seaborn | Data visualisation |
| SHAP | Model explainability |
| Streamlit | Interactive web dashboard |
| joblib | Saving and loading model artifacts |

---

## Dataset

**TMDB 5000 Movie Dataset** — published by The Movie Database (TMDB) and available on Kaggle.

- ~5,000 films with metadata, cast, crew, financials, and ratings
- Two files: `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`
- Merged on film ID during the data loading step

[View dataset on Kaggle →](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

---


## Author

**Shazia Eqbal**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Shazia%20Eqbal-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/shazia-eqbal-174400409)
[![Email](https://img.shields.io/badge/Email-shaziaeqbal54@gmail.com-D14836?style=flat-square&logo=gmail)](mailto:shaziaeqbal54@gmail.com)

---

## License

This project is licensed under the MIT License. The TMDB dataset is subject to [Kaggle's terms of use](https://www.kaggle.com/terms).