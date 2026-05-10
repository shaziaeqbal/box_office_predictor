"""
preprocessing.py
----------------
Cleans the merged TMDB dataset and engineers features for modeling.
"""

import pandas as pd
import numpy as np
import json


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _parse_json_col(value: str) -> list:
    """Safely parse a stringified JSON column (genres, cast, crew, etc.)."""
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []


def _extract_names(value: str, limit: int = None) -> list:
    """Extract 'name' fields from a JSON column (genres, companies, etc.)."""
    items = _parse_json_col(value)
    names = [item["name"] for item in items if "name" in item]
    return names[:limit] if limit else names


def _count_items(value: str) -> int:
    """Count items in a JSON-encoded list column."""
    return len(_parse_json_col(value))


def _get_director(crew_str: str) -> str:
    """Extract the director name from the crew JSON column."""
    crew = _parse_json_col(crew_str)
    for member in crew:
        if member.get("job") == "Director":
            return member.get("name", "Unknown")
    return "Unknown"


# ---------------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full cleaning and feature engineering pipeline.

    Args:
        df: Raw merged DataFrame from data_loader.

    Returns:
        (clean_df, model_df)
        - clean_df  : Fully cleaned dataframe (for EDA / visualisation)
        - model_df  : Feature matrix + target ready for modelling
    """
    print("[2/5] Preprocessing data...")

    df = df.copy()

    # ------------------------------------------------------------------ #
    # 1. Drop exact duplicates
    # ------------------------------------------------------------------ #
    before = len(df)
    df.drop_duplicates(subset=["id"], inplace=True)
    print(f"      Dropped {before - len(df)} duplicate rows.")

    # ------------------------------------------------------------------ #
    # 2. Filter: revenue & budget must be > 0 (missing data encoded as 0)
    # ------------------------------------------------------------------ #
    df = df[(df["revenue"] > 0) & (df["budget"] > 0)]
    print(f"      Rows after revenue/budget filter: {len(df)}")

    # ------------------------------------------------------------------ #
    # 3. Parse JSON columns → scalar features
    # ------------------------------------------------------------------ #
    df["genres_list"]     = df["genres"].apply(lambda x: _extract_names(x))
    df["genres_count"]    = df["genres"].apply(_count_items)

    df["keywords_count"]  = df["keywords"].apply(_count_items)

    df["cast_size"]       = df["cast"].apply(_count_items)
    df["crew_size"]       = df["crew"].apply(_count_items)
    df["director"]        = df["crew"].apply(_get_director)

    df["production_companies_count"] = df["production_companies"].apply(_count_items)
    df["spoken_languages_count"]     = df["spoken_languages"].apply(_count_items)

    # ------------------------------------------------------------------ #
    # 4. Date features
    # ------------------------------------------------------------------ #
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"]  = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month

    # ------------------------------------------------------------------ #
    # 5. Binary flag: does the movie have a homepage?
    # ------------------------------------------------------------------ #
    df["has_homepage"] = df["homepage"].notna().astype(int)

    # ------------------------------------------------------------------ #
    # 6. Fill / drop remaining nulls
    # ------------------------------------------------------------------ #
    df["runtime"].fillna(df["runtime"].median(), inplace=True)
    df["release_year"].fillna(df["release_year"].median(), inplace=True)
    df["release_month"].fillna(6, inplace=True)          # default: June

    df.dropna(subset=["revenue", "budget", "popularity", "vote_average"], inplace=True)
    print(f"      Rows after null drops : {len(df)}")

    # ------------------------------------------------------------------ #
    # 7. Log-transform revenue and budget (reduce skew)
    # ------------------------------------------------------------------ #
    df["log_revenue"] = np.log1p(df["revenue"])
    df["log_budget"]  = np.log1p(df["budget"])

    # ------------------------------------------------------------------ #
    # 8. Build model-ready feature matrix
    # ------------------------------------------------------------------ #
    FEATURES = [
        "log_budget",
        "runtime",
        "popularity",
        "vote_average",
        "vote_count",
        "genres_count",
        "cast_size",
        "crew_size",
        "has_homepage",
        "release_month",
        "release_year",
        "spoken_languages_count",
        "production_companies_count",
        "keywords_count",
    ]

    TARGET = "log_revenue"

    model_df = df[FEATURES + [TARGET]].dropna()
    print(f"      Model dataset shape   : {model_df.shape}")
    print(f"      Features              : {FEATURES}")

    return df, model_df
