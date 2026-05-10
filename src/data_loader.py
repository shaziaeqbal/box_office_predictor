"""
data_loader.py
--------------
Loads and merges the TMDB 5000 movies and credits datasets.
"""

import pandas as pd
import os


def load_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Load and merge tmdb_5000_movies.csv and tmdb_5000_credits.csv.

    Args:
        data_dir: Path to the folder containing both CSVs.

    Returns:
        Merged DataFrame on movie id.
    """
    movies_path = os.path.join(data_dir, "tmdb_5000_movies.csv")
    credits_path = os.path.join(data_dir, "tmdb_5000_credits.csv")

    if not os.path.exists(movies_path):
        raise FileNotFoundError(
            f"Missing: {movies_path}\n"
            "Please download 'tmdb_5000_movies.csv' from Kaggle and place it in the data/ folder."
        )
    if not os.path.exists(credits_path):
        raise FileNotFoundError(
            f"Missing: {credits_path}\n"
            "Please download 'tmdb_5000_credits.csv' from Kaggle and place it in the data/ folder."
        )

    print("[1/5] Loading datasets...")
    movies  = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    print(f"      Movies shape  : {movies.shape}")
    print(f"      Credits shape : {credits.shape}")

    # Normalise join key: Kaggle credits CSV may use 'movie_id' instead of 'id'
    if "movie_id" in credits.columns and "id" not in credits.columns:
        credits.rename(columns={"movie_id": "id"}, inplace=True)

    # Drop columns in credits that are already in movies (except the join key 'id')
    overlap = [c for c in credits.columns if c in movies.columns and c != "id"]
    credits = credits.drop(columns=overlap, errors="ignore")

    merged = movies.merge(credits, on="id")

    print(f"      Merged shape  : {merged.shape}")
    return merged
