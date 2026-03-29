from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import math
import pickle

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import GENRE_ORDER, build_genre_matrix


def _minmax_scale(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if math.isclose(vmin, vmax):
        return np.ones_like(values)
    return (values - vmin) / (vmax - vmin)


@dataclass
class LatentRetriever:
    n_factors: int = 48
    random_state: int = 42
    user_index_: dict[str, int] | None = None
    item_index_: dict[str, int] | None = None
    inverse_user_index_: list[str] | None = None
    inverse_item_index_: list[str] | None = None
    user_factors_: np.ndarray | None = None
    item_factors_: np.ndarray | None = None
    user_history_: dict[str, set[str]] | None = None
    popularity_: dict[str, float] | None = None

    def fit(self, interactions: pd.DataFrame, all_user_ids: Sequence[str], all_item_ids: Sequence[str]) -> "LatentRetriever":
        user_ids = [str(u) for u in all_user_ids]
        item_ids = [str(i) for i in all_item_ids]
        self.user_index_ = {u: idx for idx, u in enumerate(user_ids)}
        self.item_index_ = {i: idx for idx, i in enumerate(item_ids)}
        self.inverse_user_index_ = user_ids
        self.inverse_item_index_ = item_ids

        rows = interactions["user_id"].map(self.user_index_).to_numpy()
        cols = interactions["item_id"].map(self.item_index_).to_numpy()
        data = interactions.get("weight", interactions["rating"]).to_numpy(dtype=float)
        matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(item_ids)))

        max_components = max(2, min(self.n_factors, min(matrix.shape) - 1))
        svd = TruncatedSVD(n_components=max_components, random_state=self.random_state)
        self.user_factors_ = svd.fit_transform(matrix)
        self.item_factors_ = svd.components_.T

        self.user_history_ = (
            interactions.groupby("user_id")["item_id"].apply(lambda s: set(s.astype(str))).to_dict()
        )
        self.popularity_ = interactions.groupby("item_id").size().astype(float).to_dict()
        return self

    @property
    def all_item_ids(self) -> list[str]:
        return list(self.inverse_item_index_ or [])

    def score_candidates(self, user_id: str, candidate_item_ids: Sequence[str]) -> np.ndarray:
        if self.user_index_ is None or self.item_index_ is None or self.user_factors_ is None or self.item_factors_ is None:
            raise RuntimeError("Retriever has not been fit.")
        if str(user_id) not in self.user_index_:
            popularity_scores = np.array([self.popularity_.get(str(item_id), 0.0) for item_id in candidate_item_ids], dtype=float)
            return _minmax_scale(popularity_scores)

        user_idx = self.user_index_[str(user_id)]
        user_vec = self.user_factors_[user_idx]
        item_indices = [self.item_index_[str(item)] for item in candidate_item_ids if str(item) in self.item_index_]
        if not item_indices:
            return np.zeros(len(candidate_item_ids), dtype=float)
        score_lookup = {}
        raw_scores = self.item_factors_[item_indices] @ user_vec
        for item_id, score in zip([str(i) for i in candidate_item_ids if str(i) in self.item_index_], raw_scores):
            score_lookup[item_id] = float(score)
        return np.array([score_lookup.get(str(item), 0.0) for item in candidate_item_ids], dtype=float)

    def recommend(
        self,
        user_id: str,
        top_n: int = 100,
        exclude_seen: bool = True,
        extra_candidate_ids: Sequence[str] | None = None,
    ) -> tuple[list[str], np.ndarray]:
        if self.item_factors_ is None or self.inverse_item_index_ is None:
            raise RuntimeError("Retriever has not been fit.")
        all_items = self.inverse_item_index_
        if str(user_id) in (self.user_index_ or {}):
            raw_scores = self.score_candidates(str(user_id), all_items)
        else:
            raw_scores = np.array([self.popularity_.get(item_id, 0.0) for item_id in all_items], dtype=float)

        seen = self.user_history_.get(str(user_id), set()) if exclude_seen and self.user_history_ else set()
        candidate_pairs = [(item_id, float(score)) for item_id, score in zip(all_items, raw_scores) if item_id not in seen]

        if extra_candidate_ids:
            score_map = {item_id: score for item_id, score in candidate_pairs}
            for item_id in extra_candidate_ids:
                if item_id in seen:
                    continue
                score_map.setdefault(str(item_id), float(self.popularity_.get(str(item_id), 0.0)))
            candidate_pairs = list(score_map.items())

        candidate_pairs.sort(key=lambda x: x[1], reverse=True)
        selected = candidate_pairs[:top_n]
        item_ids = [item_id for item_id, _ in selected]
        scores = np.array([score for _, score in selected], dtype=float)
        return item_ids, scores

    def similar_items(self, item_id: str, top_n: int = 10) -> list[tuple[str, float]]:
        if self.item_index_ is None or self.item_factors_ is None or self.inverse_item_index_ is None:
            raise RuntimeError("Retriever has not been fit.")
        if str(item_id) not in self.item_index_:
            return []
        idx = self.item_index_[str(item_id)]
        target = self.item_factors_[idx]
        norms = np.linalg.norm(self.item_factors_, axis=1)
        denom = norms * max(np.linalg.norm(target), 1e-8)
        similarities = np.divide(self.item_factors_ @ target, denom, out=np.zeros_like(denom), where=denom != 0)
        pairs = []
        for j, score in enumerate(similarities):
            current_id = self.inverse_item_index_[j]
            if current_id == str(item_id):
                continue
            pairs.append((current_id, float(score)))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_n]


class FeatureBuilder:
    def __init__(
        self,
        users: pd.DataFrame,
        items: pd.DataFrame,
        interactions: pd.DataFrame,
        retriever: LatentRetriever,
    ) -> None:
        self.users = users.copy()
        self.items = items.copy()
        self.interactions = interactions.copy()
        self.retriever = retriever

        self.user_lookup = self.users.set_index("user_id")
        self.items["genres"] = self.items["genres"].map(lambda x: x if isinstance(x, list) else [g for g in str(x).split("|") if g])
        self.item_lookup = self.items.set_index("item_id")
        self.genre_matrix = build_genre_matrix(self.items)
        self.item_popularity = self.interactions.groupby("item_id").size().astype(float)
        self.item_avg_rating = self.interactions.groupby("item_id")["rating"].mean()
        self.user_avg_rating = self.interactions.groupby("user_id")["rating"].mean()
        self.user_interactions = self.interactions.groupby("user_id").size()
        self.user_genre_profiles = self._build_user_genre_profiles()
        self.n_users = max(len(self.user_lookup), 1)

        merged = self.interactions.merge(self.users, on="user_id", how="left")
        cohort_counts = merged.groupby(["gender", "age_bucket", "item_id"]).size().rename("count").reset_index()
        self.cohort_lookup = {}
        for (gender, age_bucket), group in cohort_counts.groupby(["gender", "age_bucket"]):
            series = group.set_index("item_id")["count"].astype(float)
            self.cohort_lookup[(str(gender), str(age_bucket))] = series

    def _build_user_genre_profiles(self) -> pd.DataFrame:
        merged = self.interactions.merge(self.items[["item_id", "genres"]], on="item_id", how="left")
        rows = []
        for user_id, group in merged.groupby("user_id"):
            vector = {genre: 0.0 for genre in GENRE_ORDER}
            for _, row in group.iterrows():
                rating_weight = max(float(row["rating"]) - 3.0, 0.2)
                for genre in row["genres"]:
                    if genre in vector:
                        vector[genre] += rating_weight
            total = sum(vector.values()) or 1.0
            rows.append({"user_id": str(user_id), **{g: vector[g] / total for g in GENRE_ORDER}})
        if not rows:
            return pd.DataFrame(columns=["user_id", *GENRE_ORDER]).set_index("user_id")
        return pd.DataFrame(rows).set_index("user_id")

    def _user_row(self, user_id: str) -> pd.Series:
        if str(user_id) in self.user_lookup.index:
            return self.user_lookup.loc[str(user_id)]
        return pd.Series({"gender": "Unknown", "age_bucket": "Unknown", "occupation": "Unknown"})

    def candidate_frame(
        self,
        user_id: str,
        candidate_item_ids: Sequence[str],
        candidate_scores: Sequence[float] | None = None,
    ) -> pd.DataFrame:
        user_row = self._user_row(str(user_id))
        items = self.item_lookup.loc[[str(item_id) for item_id in candidate_item_ids]].copy()
        items["candidate_score_raw"] = (
            np.array(candidate_scores, dtype=float)
            if candidate_scores is not None
            else self.retriever.score_candidates(str(user_id), [str(item_id) for item_id in candidate_item_ids])
        )
        items["candidate_score"] = _minmax_scale(items["candidate_score_raw"].to_numpy(dtype=float))
        items["popularity"] = items.index.map(lambda item_id: float(self.item_popularity.get(item_id, 0.0)))
        items["popularity_norm"] = _minmax_scale(items["popularity"].to_numpy(dtype=float))
        items["item_avg_rating"] = items.index.map(lambda item_id: float(self.item_avg_rating.get(item_id, 0.0)))
        items["user_avg_rating"] = float(self.user_avg_rating.get(str(user_id), self.interactions["rating"].mean()))
        items["user_interactions"] = float(self.user_interactions.get(str(user_id), 0.0))
        items["novelty"] = items["popularity"].map(lambda p: -math.log2(max(p, 1.0) / max(self.n_users, 1)))
        items["novelty_norm"] = _minmax_scale(items["novelty"].to_numpy(dtype=float))
        items["release_year"] = items["year"].astype(int)
        items["year_norm"] = _minmax_scale(items["release_year"].to_numpy(dtype=float))

        user_genre = self.user_genre_profiles.loc[str(user_id)].to_numpy(dtype=float) if str(user_id) in self.user_genre_profiles.index else np.zeros(len(GENRE_ORDER))
        item_genres = self.genre_matrix.loc[items.index].to_numpy(dtype=float)
        user_norm = np.linalg.norm(user_genre)
        item_norms = np.linalg.norm(item_genres, axis=1)
        if user_norm > 0:
            cosine = (item_genres @ user_genre) / np.maximum(item_norms * user_norm, 1e-8)
        else:
            cosine = np.zeros(len(items))
        items["genre_match"] = cosine
        top_user_genres = set()
        if str(user_id) in self.user_genre_profiles.index:
            sorted_pairs = sorted(zip(GENRE_ORDER, self.user_genre_profiles.loc[str(user_id)].tolist()), key=lambda x: x[1], reverse=True)
            top_user_genres = {genre for genre, score in sorted_pairs[:3] if score > 0}

        items["genre_overlap_count"] = [
            len(top_user_genres.intersection(set(genres)))
            for genres in items["genres"].tolist()
        ]

        cohort_series = self.cohort_lookup.get((str(user_row.get("gender", "Unknown")), str(user_row.get("age_bucket", "Unknown"))))
        if cohort_series is None:
            items["cohort_popularity"] = 0.0
        else:
            items["cohort_popularity"] = items.index.map(lambda item_id: float(cohort_series.get(item_id, 0.0)))
        items["cohort_popularity_norm"] = _minmax_scale(items["cohort_popularity"].to_numpy(dtype=float))

        items["user_gender"] = str(user_row.get("gender", "Unknown"))
        items["age_bucket"] = str(user_row.get("age_bucket", "Unknown"))
        items["occupation"] = str(user_row.get("occupation", "Unknown"))
        items["user_id"] = str(user_id)
        items["item_id"] = items.index.astype(str)
        return items.reset_index(drop=True)

    @staticmethod
    def training_feature_columns() -> tuple[list[str], list[str]]:
        numeric = [
            "candidate_score",
            "candidate_score_raw",
            "popularity",
            "popularity_norm",
            "item_avg_rating",
            "user_avg_rating",
            "user_interactions",
            "genre_match",
            "genre_overlap_count",
            "novelty",
            "novelty_norm",
            "cohort_popularity",
            "cohort_popularity_norm",
            "release_year",
            "year_norm",
        ]
        categorical = ["primary_genre", "user_gender", "age_bucket", "occupation"]
        return numeric, categorical

    def build_ranker_pipeline(self) -> Pipeline:
        numeric_features, categorical_features = self.training_feature_columns()
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ]
        )
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=42)),
            ]
        )


def build_ranker_training_frame(
    retriever: LatentRetriever,
    feature_builder: FeatureBuilder,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    candidate_k: int = 80,
) -> pd.DataFrame:
    training_rows = []
    validation_by_user = validation_df.groupby("user_id")["item_id"].apply(list).to_dict()
    for user_id, relevant_items in validation_by_user.items():
        candidate_ids, candidate_scores = retriever.recommend(
            user_id=str(user_id),
            top_n=candidate_k,
            exclude_seen=True,
            extra_candidate_ids=relevant_items,
        )
        frame = feature_builder.candidate_frame(str(user_id), candidate_ids, candidate_scores)
        relevant_set = {str(item_id) for item_id in relevant_items}
        frame["label"] = frame["item_id"].map(lambda item_id: 1 if item_id in relevant_set else 0)
        training_rows.append(frame)

    if not training_rows:
        raise ValueError("Could not build ranker training frame.")
    ranker_df = pd.concat(training_rows, ignore_index=True)
    return ranker_df


def rerank_mmr(
    candidate_df: pd.DataFrame,
    item_feature_matrix: pd.DataFrame,
    top_k: int,
    diversity_lambda: float,
    max_per_primary_genre: int = 3,
) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df

    chosen_indices: list[int] = []
    remaining = candidate_df.index.tolist()
    genre_counts: dict[str, int] = {}
    feature_lookup = item_feature_matrix

    while remaining and len(chosen_indices) < top_k:
        best_idx = None
        best_score = -1e18
        for idx in remaining:
            row = candidate_df.loc[idx]
            primary_genre = row["primary_genre"]
            if genre_counts.get(primary_genre, 0) >= max_per_primary_genre:
                continue
            relevance = float(row["blend_score"])
            similarity_penalty = 0.0
            if chosen_indices:
                current_item = row["item_id"]
                if current_item in feature_lookup.index:
                    current_vec = feature_lookup.loc[current_item].to_numpy(dtype=float)
                    current_norm = np.linalg.norm(current_vec)
                    similarities = []
                    for selected_idx in chosen_indices:
                        selected_item = candidate_df.loc[selected_idx, "item_id"]
                        if selected_item not in feature_lookup.index:
                            continue
                        selected_vec = feature_lookup.loc[selected_item].to_numpy(dtype=float)
                        denom = current_norm * max(np.linalg.norm(selected_vec), 1e-8)
                        if denom == 0:
                            similarities.append(0.0)
                        else:
                            similarities.append(float(np.dot(current_vec, selected_vec) / denom))
                    similarity_penalty = max(similarities, default=0.0)
            rerank_score = diversity_lambda * relevance - (1 - diversity_lambda) * similarity_penalty
            if rerank_score > best_score:
                best_score = rerank_score
                best_idx = idx
        if best_idx is None:
            best_idx = remaining[0]
        chosen_indices.append(best_idx)
        chosen_genre = candidate_df.loc[best_idx, "primary_genre"]
        genre_counts[chosen_genre] = genre_counts.get(chosen_genre, 0) + 1
        remaining.remove(best_idx)

    return candidate_df.loc[chosen_indices].reset_index(drop=True)


def save_pickle(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fp:
        pickle.dump(obj, fp)


def load_pickle(path: Path) -> object:
    with path.open("rb") as fp:
        return pickle.load(fp)
