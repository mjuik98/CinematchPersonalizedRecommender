from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import csv
import io
import json
import math
import random
import re
import urllib.request
import zipfile

import numpy as np
import pandas as pd


MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
MOVIELENS_1M_LICENSE_URL = "https://files.grouplens.org/datasets/movielens/ml-1m-README.txt"

GENRE_ORDER = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

AGE_BUCKET_LABELS = {
    1: "Under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "56+",
}


@dataclass(slots=True)
class DatasetBundle:
    users: pd.DataFrame
    items: pd.DataFrame
    interactions: pd.DataFrame
    source_name: str


def _extract_year(title: str) -> int | None:
    match = re.search(r"\((\d{4})\)\s*$", title)
    if not match:
        return None
    return int(match.group(1))


def _normalize_items(items: pd.DataFrame) -> pd.DataFrame:
    items = items.copy()
    items["genres"] = items["genres"].fillna("").map(lambda x: [g for g in str(x).split("|") if g])
    items["primary_genre"] = items["genres"].map(lambda gs: gs[0] if gs else "Unknown")
    items["year"] = items["title"].map(_extract_year)
    items["year"] = items["year"].fillna(items["year"].median() if items["year"].notna().any() else 2000).astype(int)
    return items


def _normalize_users(users: pd.DataFrame) -> pd.DataFrame:
    users = users.copy()
    users["user_id"] = users["user_id"].astype(str)
    users["gender"] = users["gender"].astype(str)
    def _normalize_age(value):
        if pd.isna(value):
            return "Unknown"
        try:
            return AGE_BUCKET_LABELS.get(int(value), str(value))
        except (TypeError, ValueError):
            return str(value)

    users["age_bucket"] = users["age_bucket"].map(_normalize_age)
    users["occupation"] = users["occupation"].astype(str)
    return users


def _normalize_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    interactions = interactions.copy()
    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["item_id"] = interactions["item_id"].astype(str)
    interactions["rating"] = interactions["rating"].astype(float)
    interactions["timestamp"] = interactions["timestamp"].astype(int)
    return interactions.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)


def download_movielens_1m(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "ml-1m.zip"
    manual_zip = Path(str(Path.cwd() / "ml-1m.zip"))
    if zip_path.exists():
        return zip_path
    if manual_zip.exists():
        zip_path.write_bytes(manual_zip.read_bytes())
        return zip_path

    try:
        with urllib.request.urlopen(MOVIELENS_1M_URL, timeout=60) as response:
            zip_path.write_bytes(response.read())
    except Exception as exc:  # pragma: no cover - network-dependent branch
        raise RuntimeError(
            "Could not download MovieLens 1M automatically. "
            f"Download {MOVIELENS_1M_URL} manually and place it at {zip_path} or project root as ml-1m.zip."
        ) from exc
    return zip_path


def extract_zip(zip_path: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
    return target_dir


def load_movielens_1m(raw_dir: Path) -> DatasetBundle:
    zip_path = download_movielens_1m(raw_dir)
    extract_dir = raw_dir / "ml-1m"
    if not (extract_dir / "ml-1m").exists():
        extract_zip(zip_path, extract_dir)
    dataset_root = extract_dir / "ml-1m"

    ratings = pd.read_csv(
        dataset_root / "ratings.dat",
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    movies = pd.read_csv(
        dataset_root / "movies.dat",
        sep="::",
        engine="python",
        names=["item_id", "title", "genres"],
        encoding="latin-1",
    )
    users = pd.read_csv(
        dataset_root / "users.dat",
        sep="::",
        engine="python",
        names=["user_id", "gender", "age_bucket", "occupation", "zip_code"],
        encoding="latin-1",
    )

    users["occupation"] = users["occupation"].astype(str)
    movies["item_id"] = movies["item_id"].astype(str)
    users["user_id"] = users["user_id"].astype(str)

    normalized_users = _normalize_users(users[["user_id", "gender", "age_bucket", "occupation"]])
    normalized_items = _normalize_items(movies[["item_id", "title", "genres"]])
    normalized_interactions = _normalize_interactions(ratings)
    return DatasetBundle(
        users=normalized_users,
        items=normalized_items,
        interactions=normalized_interactions,
        source_name="movielens_1m",
    )


def load_csv_dataset(dataset_dir: Path, source_name: str = "sample_csv") -> DatasetBundle:
    users = pd.read_csv(dataset_dir / "users.csv")
    items = pd.read_csv(dataset_dir / "items.csv")
    interactions = pd.read_csv(dataset_dir / "interactions.csv")

    users["occupation"] = users["occupation"].astype(str)
    items["item_id"] = items["item_id"].astype(str)
    users["user_id"] = users["user_id"].astype(str)

    users = _normalize_users(users[["user_id", "gender", "age_bucket", "occupation"]])
    items = _normalize_items(items[["item_id", "title", "genres"]])
    interactions = _normalize_interactions(interactions[["user_id", "item_id", "rating", "timestamp"]])

    return DatasetBundle(users=users, items=items, interactions=interactions, source_name=source_name)


def load_dataset(source: str, sample_dir: Path, raw_dir: Path) -> DatasetBundle:
    if source == "sample":
        return load_csv_dataset(sample_dir, source_name="sample")
    if source == "movielens_1m":
        return load_movielens_1m(raw_dir)
    custom_dir = Path(source)
    if custom_dir.exists():
        return load_csv_dataset(custom_dir, source_name=custom_dir.name)
    raise ValueError(f"Unsupported dataset source: {source}")


def build_positive_interactions(interactions: pd.DataFrame, positive_threshold: float = 4.0) -> pd.DataFrame:
    positives = interactions[interactions["rating"] >= positive_threshold].copy()
    positives["weight"] = positives["rating"] - positive_threshold + 1.0
    return positives.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)


def temporal_leave_k_out_split(
    positives: pd.DataFrame,
    min_train_interactions: int = 3,
    val_interactions: int = 1,
    test_interactions: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_rows = []
    val_rows = []
    test_rows = []

    for _, user_df in positives.groupby("user_id", sort=False):
        user_df = user_df.sort_values(["timestamp", "item_id"]).reset_index(drop=True)
        n = len(user_df)
        current_test = min(test_interactions, max(1, n // 6))
        current_val = min(val_interactions, max(1, (n - current_test) // 8))
        if n - current_test - current_val < min_train_interactions:
            continue

        train_rows.append(user_df.iloc[: n - current_test - current_val])
        val_rows.append(user_df.iloc[n - current_test - current_val : n - current_test])
        test_rows.append(user_df.iloc[n - current_test :])

    if not train_rows:
        raise ValueError("No users met the minimum interaction requirement after splitting.")

    return (
        pd.concat(train_rows).reset_index(drop=True),
        pd.concat(val_rows).reset_index(drop=True),
        pd.concat(test_rows).reset_index(drop=True),
    )


def build_genre_matrix(items: pd.DataFrame) -> pd.DataFrame:
    data = []
    for _, row in items.iterrows():
        genre_set = set(row["genres"])
        data.append({genre: float(genre in genre_set) for genre in GENRE_ORDER})
    genre_df = pd.DataFrame(data, index=items["item_id"].astype(str))
    return genre_df.sort_index()


def save_dataset_bundle(dataset: DatasetBundle, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    dataset.users.to_csv(target_dir / "users.csv", index=False)
    dataset.items.assign(
        genres=dataset.items["genres"].map(lambda xs: "|".join(xs))
    ).to_csv(target_dir / "items.csv", index=False)
    dataset.interactions.to_csv(target_dir / "interactions.csv", index=False)
    metadata = {"source_name": dataset.source_name}
    (target_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def create_synthetic_sample_dataset(target_dir: Path, random_seed: int = 42) -> DatasetBundle:
    rng = random.Random(random_seed)
    np_rng = np.random.default_rng(random_seed)
    target_dir.mkdir(parents=True, exist_ok=True)

    genre_blueprints = {
        "Action": ["Action", "Adventure", "Thriller", "Sci-Fi"],
        "Drama": ["Drama", "Romance", "Mystery", "Film-Noir"],
        "Comedy": ["Comedy", "Children's", "Animation", "Musical"],
        "Crime": ["Crime", "Thriller", "Mystery", "Drama"],
        "Fantasy": ["Fantasy", "Adventure", "Animation", "Sci-Fi"],
        "Documentary": ["Documentary", "War", "Drama", "Western"],
    }
    occupations = ["student", "engineer", "designer", "researcher", "manager", "writer"]
    age_buckets = ["18-24", "25-34", "35-44", "45-49", "50-55"]
    genders = ["F", "M"]

    items = []
    item_id = 1
    for anchor, genre_pool in genre_blueprints.items():
        for i in range(15):
            genre_count = 2 if i % 3 else 3
            genres = rng.sample(genre_pool, k=genre_count)
            title = f"{anchor} Story {i+1} ({1990 + (i % 25)})"
            items.append(
                {
                    "item_id": str(item_id),
                    "title": title,
                    "genres": "|".join(sorted(set(genres))),
                }
            )
            item_id += 1

    users = []
    interactions = []
    ts = 1_600_000_000
    item_df = pd.DataFrame(items)
    item_df["genre_list"] = item_df["genres"].str.split("|")

    for u in range(1, 31):
        preferred_anchor = list(genre_blueprints)[u % len(genre_blueprints)]
        secondary_anchor = list(genre_blueprints)[(u + 2) % len(genre_blueprints)]
        users.append(
            {
                "user_id": str(u),
                "gender": genders[u % 2],
                "age_bucket": age_buckets[u % len(age_buckets)],
                "occupation": occupations[u % len(occupations)],
            }
        )
        prefs = set(genre_blueprints[preferred_anchor][:2] + genre_blueprints[secondary_anchor][:1])
        candidate_items = []
        for _, row in item_df.iterrows():
            overlap = len(prefs.intersection(row["genre_list"]))
            base = 0.2 + 0.25 * overlap
            if preferred_anchor in row["title"]:
                base += 0.25
            candidate_items.append((row["item_id"], min(base, 0.95)))
        for item_id_str, prob in candidate_items:
            if rng.random() < prob:
                item_genres = set(item_df.loc[item_df["item_id"] == item_id_str, "genre_list"].iloc[0])
                overlap = len(prefs.intersection(item_genres))
                rating = 3 + overlap + (1 if rng.random() < 0.35 else 0)
                rating = max(1, min(5, rating))
                interactions.append(
                    {
                        "user_id": str(u),
                        "item_id": str(item_id_str),
                        "rating": float(rating),
                        "timestamp": ts,
                    }
                )
                ts += rng.randint(2500, 12000)

    users_df = pd.DataFrame(users)
    items_df = pd.DataFrame(items)
    interactions_df = pd.DataFrame(interactions)

    dataset = DatasetBundle(
        users=_normalize_users(users_df),
        items=_normalize_items(items_df),
        interactions=_normalize_interactions(interactions_df),
        source_name="sample",
    )
    save_dataset_bundle(dataset, target_dir)
    return dataset
