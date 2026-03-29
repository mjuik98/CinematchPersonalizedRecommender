from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
import json
import math
import pickle

import numpy as np
import pandas as pd

from .config import settings
from .metrics import (
    average_intra_list_diversity,
    average_precision_at_k,
    catalog_coverage,
    hit_rate_at_k,
    long_tail_share_at_k,
    mrr_at_k,
    ndcg_at_k,
    novelty_at_k,
    personalization_at_k,
    precision_at_k,
    recall_at_k,
)
from .modeling import FeatureBuilder, LatentRetriever, load_pickle, rerank_mmr, save_pickle
from .reporting import create_metric_chart, render_html_report, save_json
from .storage import fetch_feedback_summary, fetch_recommendation_summary, init_db, log_feedback, log_recommendation


@dataclass
class RecommendationResult:
    item_id: str
    title: str
    genres: list[str]
    score: float
    reasons: list[str]
    primary_genre: str
    year: int


class RecommendationService:
    def __init__(self, bundle: dict[str, Any], db_path: Path | None = None) -> None:
        self.bundle = bundle
        self.users = bundle["users"]
        self.items = bundle["items"]
        self.retriever: LatentRetriever = bundle["retriever"]
        self.rank_pipeline = bundle["rank_pipeline"]
        self.feature_builder: FeatureBuilder = bundle["feature_builder"]
        self.item_lookup = self.items.set_index("item_id")
        self.history = bundle["history"]
        self.item_feature_matrix = bundle["item_feature_matrix"]
        self.item_popularity = bundle["item_popularity"]
        self.metadata = bundle["metadata"]
        self.latest_metrics = bundle.get("latest_metrics", {})
        self.db_path = db_path or settings.sqlite_path
        init_db(self.db_path)

    @classmethod
    def from_path(cls, artifact_path: Path) -> "RecommendationService":
        bundle = load_pickle(artifact_path)
        return cls(bundle=bundle)

    def _reason_for_item(self, user_id: str, row: pd.Series, mode: str) -> list[str]:
        reasons = []
        top_genres = []
        if str(user_id) in self.feature_builder.user_genre_profiles.index:
            profile = self.feature_builder.user_genre_profiles.loc[str(user_id)].sort_values(ascending=False)
            top_genres = [genre for genre, score in profile.items() if score > 0][:3]
        overlap = [genre for genre in row["genres"] if genre in top_genres]
        if overlap:
            reasons.append(f"선호 장르 {', '.join(overlap[:2])}와 일치")
        if float(row.get("candidate_score", 0.0)) > 0.65:
            reasons.append("협업 필터링 점수가 높음")
        if float(row.get("cohort_popularity_norm", 0.0)) > 0.45:
            reasons.append("비슷한 사용자 집단에서 인기")
        if float(row.get("novelty_norm", 0.0)) > 0.55:
            reasons.append("롱테일 탐색성 보강")
        if not reasons and mode == "cold_start":
            reasons.append("초기 프로필 기반 추천")
        elif not reasons:
            reasons.append("사용자 행동 패턴과 메타데이터를 함께 반영")
        return reasons[:2]

    def _postprocess(self, user_id: str, ranked_df: pd.DataFrame, top_k: int, mode: str) -> list[dict[str, Any]]:
        results = []
        for _, row in ranked_df.head(top_k).iterrows():
            results.append(
                {
                    "item_id": str(row["item_id"]),
                    "title": str(row["title"]),
                    "genres": list(row["genres"]),
                    "score": round(float(row["blend_score"]), 6),
                    "reasons": self._reason_for_item(str(user_id), row, mode=mode),
                    "primary_genre": str(row["primary_genre"]),
                    "year": int(row["year"]),
                }
            )
        return results

    def recommend_for_user(
        self,
        user_id: str,
        top_k: int = 10,
        candidate_k: int = 120,
        diversity_lambda: float | None = None,
        log: bool = True,
    ) -> list[dict[str, Any]]:
        diversity_lambda = settings.diversity_lambda_default if diversity_lambda is None else diversity_lambda
        user_id = str(user_id)
        seen_items = self.history.get(user_id, set())
        if user_id not in self.history or len(seen_items) < settings.min_history_for_personalization:
            user_row = self.users[self.users["user_id"] == user_id]
            if user_row.empty:
                return self.recommend_cold_start(top_k=top_k, favorite_genres=[], log=log)
            user_row = user_row.iloc[0]
            return self.recommend_cold_start(
                top_k=top_k,
                age_bucket=str(user_row["age_bucket"]),
                gender=str(user_row["gender"]),
                occupation=str(user_row["occupation"]),
                favorite_genres=[],
                log=log,
                user_id=user_id,
            )

        candidate_ids, candidate_scores = self.retriever.recommend(user_id, top_n=candidate_k, exclude_seen=True)
        feature_frame = self.feature_builder.candidate_frame(user_id, candidate_ids, candidate_scores)
        if self.rank_pipeline is not None:
            ranking_scores = self.rank_pipeline.predict_proba(feature_frame)[:, 1]
        else:
            ranking_scores = feature_frame["candidate_score"].to_numpy(dtype=float)

        feature_frame["rank_score"] = ranking_scores
        feature_frame["blend_score"] = (
            0.70 * feature_frame["rank_score"]
            + 0.20 * feature_frame["candidate_score"]
            + 0.06 * feature_frame["novelty_norm"]
            + 0.04 * feature_frame["cohort_popularity_norm"]
        )
        reranked = rerank_mmr(feature_frame.sort_values("blend_score", ascending=False).reset_index(drop=True), self.item_feature_matrix, top_k, diversity_lambda)
        payload = self._postprocess(user_id=user_id, ranked_df=reranked, top_k=top_k, mode="personalized")
        if log:
            log_recommendation(
                self.db_path,
                user_id=user_id,
                mode="personalized",
                request_payload={"top_k": top_k, "candidate_k": candidate_k, "diversity_lambda": diversity_lambda},
                response_payload=payload,
            )
        return payload

    def recommend_cold_start(
        self,
        top_k: int = 10,
        age_bucket: str | None = None,
        gender: str | None = None,
        occupation: str | None = None,
        favorite_genres: Sequence[str] | None = None,
        user_id: str = "cold_start",
        log: bool = True,
    ) -> list[dict[str, Any]]:
        favorite_genres = [g for g in (favorite_genres or []) if g]
        items = self.items.copy()
        items["popularity"] = items["item_id"].map(lambda item_id: float(self.item_popularity.get(item_id, 0.0)))
        pop = items["popularity"].to_numpy(dtype=float)
        items["popularity_norm"] = (pop - pop.min()) / (pop.max() - pop.min() + 1e-8)

        cohort_series = None
        if gender and age_bucket:
            cohort_series = self.feature_builder.cohort_lookup.get((str(gender), str(age_bucket)))
        items["cohort_popularity"] = items["item_id"].map(lambda item_id: float(cohort_series.get(item_id, 0.0)) if cohort_series is not None else 0.0)
        cohort = items["cohort_popularity"].to_numpy(dtype=float)
        items["cohort_popularity_norm"] = (cohort - cohort.min()) / (cohort.max() - cohort.min() + 1e-8) if len(cohort) else cohort

        favorite_set = set(favorite_genres)
        items["genre_match"] = items["genres"].map(lambda genres: len(favorite_set.intersection(set(genres))) / max(len(favorite_set), 1) if favorite_set else 0.0)
        items["novelty"] = items["item_id"].map(lambda item_id: -math.log2(max(self.item_popularity.get(item_id, 1.0), 1.0) / max(len(self.users), 1)))
        novelty = items["novelty"].to_numpy(dtype=float)
        items["novelty_norm"] = (novelty - novelty.min()) / (novelty.max() - novelty.min() + 1e-8) if len(novelty) else novelty
        items["candidate_score"] = 0.0
        items["blend_score"] = 0.50 * items["popularity_norm"] + 0.25 * items["cohort_popularity_norm"] + 0.20 * items["genre_match"] + 0.05 * items["novelty_norm"]
        ranked = rerank_mmr(items.sort_values("blend_score", ascending=False).reset_index(drop=True), self.item_feature_matrix, top_k, 0.80)
        payload = self._postprocess(user_id=user_id, ranked_df=ranked, top_k=top_k, mode="cold_start")
        if log:
            log_recommendation(
                self.db_path,
                user_id=user_id,
                mode="cold_start",
                request_payload={"top_k": top_k, "age_bucket": age_bucket, "gender": gender, "occupation": occupation, "favorite_genres": list(favorite_genres)},
                response_payload=payload,
            )
        return payload

    def similar_items(self, item_id: str, top_k: int = 10) -> list[dict[str, Any]]:
        results = []
        for similar_item_id, score in self.retriever.similar_items(str(item_id), top_n=top_k):
            if similar_item_id not in self.item_lookup.index:
                continue
            row = self.item_lookup.loc[similar_item_id]
            results.append(
                {
                    "item_id": similar_item_id,
                    "title": str(row["title"]),
                    "genres": list(row["genres"]),
                    "score": round(float(score), 6),
                    "primary_genre": str(row["primary_genre"]),
                    "year": int(row["year"]),
                }
            )
        return results

    def item_details(self, item_id: str) -> dict[str, Any]:
        row = self.item_lookup.loc[str(item_id)]
        return {
            "item_id": str(item_id),
            "title": str(row["title"]),
            "genres": list(row["genres"]),
            "primary_genre": str(row["primary_genre"]),
            "year": int(row["year"]),
            "popularity": float(self.item_popularity.get(str(item_id), 0.0)),
        }

    def save_feedback(self, user_id: str, item_id: str, event_type: str, value: float | None, context: dict[str, Any] | None) -> None:
        log_feedback(self.db_path, user_id=user_id, item_id=item_id, event_type=event_type, value=value, context=context)

    def analytics_summary(self) -> dict[str, Any]:
        return {
            "recommendations": fetch_recommendation_summary(self.db_path),
            "feedback": fetch_feedback_summary(self.db_path),
        }

    def metadata_payload(self) -> dict[str, Any]:
        return self.metadata

    def latest_metrics_payload(self) -> dict[str, Any]:
        return self.latest_metrics


def evaluate_recommendation_models(
    service: RecommendationService,
    test_df: pd.DataFrame,
    top_k: int = 10,
) -> dict[str, Any]:
    relevant_by_user = test_df.groupby("user_id")["item_id"].apply(lambda s: set(s.astype(str))).to_dict()
    eval_users = list(relevant_by_user)

    def popularity_recs(user_id: str) -> list[str]:
        seen = service.history.get(str(user_id), set())
        items = [
            item_id
            for item_id, _ in sorted(service.item_popularity.items(), key=lambda kv: kv[1], reverse=True)
            if item_id not in seen
        ]
        return items[:top_k]

    def retriever_recs(user_id: str) -> list[str]:
        item_ids, _ = service.retriever.recommend(str(user_id), top_n=top_k, exclude_seen=True)
        return item_ids[:top_k]

    def final_recs(user_id: str) -> list[str]:
        return [item["item_id"] for item in service.recommend_for_user(str(user_id), top_k=top_k, log=False)]

    recommenders = {
        "Popularity": popularity_recs,
        "Latent Retriever": retriever_recs,
        "Final Multi-Stage": final_recs,
    }

    metrics_by_model = []
    recommendations_by_model = {}
    total_items = len(service.items)
    item_popularity = service.item_popularity
    for model_name, recommender in recommenders.items():
        rec_lookup: dict[str, list[str]] = {}
        per_user_rows = []
        for user_id in eval_users:
            recs = recommender(user_id)
            rec_lookup[str(user_id)] = recs
            relevant = relevant_by_user[str(user_id)]
            per_user_rows.append(
                {
                    "user_id": str(user_id),
                    "precision@k": precision_at_k(recs, relevant, top_k),
                    "recall@k": recall_at_k(recs, relevant, top_k),
                    "hit_rate@k": hit_rate_at_k(recs, relevant, top_k),
                    "map@k": average_precision_at_k(recs, relevant, top_k),
                    "mrr@k": mrr_at_k(recs, relevant, top_k),
                    "ndcg@k": ndcg_at_k(recs, relevant, top_k),
                }
            )
        per_user_df = pd.DataFrame(per_user_rows)
        metrics = {
            "model": model_name,
            "precision@k": float(per_user_df["precision@k"].mean()),
            "recall@k": float(per_user_df["recall@k"].mean()),
            "hit_rate@k": float(per_user_df["hit_rate@k"].mean()),
            "map@k": float(per_user_df["map@k"].mean()),
            "mrr@k": float(per_user_df["mrr@k"].mean()),
            "ndcg@k": float(per_user_df["ndcg@k"].mean()),
            "coverage": catalog_coverage(rec_lookup, total_items, top_k),
            "novelty": novelty_at_k(rec_lookup, item_popularity, len(service.users), top_k),
            "diversity": average_intra_list_diversity(rec_lookup, service.item_feature_matrix, top_k),
            "personalization": personalization_at_k(rec_lookup, top_k),
            "long_tail_share": long_tail_share_at_k(rec_lookup, item_popularity, top_k),
        }
        metrics_by_model.append(metrics)
        recommendations_by_model[model_name] = rec_lookup

    metrics_df = pd.DataFrame(metrics_by_model)
    relevance_chart = create_metric_chart(metrics_df, ["hit_rate@k", "ndcg@k", "mrr@k"], "Relevance metrics")
    business_chart = create_metric_chart(metrics_df, ["coverage", "diversity", "personalization"], "Catalog and diversity metrics")
    return {
        "metrics_df": metrics_df,
        "recommendations_by_model": recommendations_by_model,
        "charts": {
            "relevance_chart": relevance_chart,
            "business_chart": business_chart,
        },
    }


def export_evaluation_report(
    service: RecommendationService,
    test_df: pd.DataFrame,
    output_dir: Path,
    dataset_source: str,
    top_k: int = 10,
) -> dict[str, Any]:
    evaluation = evaluate_recommendation_models(service, test_df, top_k=top_k)
    metrics_df: pd.DataFrame = evaluation["metrics_df"]

    generated_at = datetime.now(timezone.utc).isoformat()
    html_path = output_dir / f"evaluation_report_{dataset_source}.html"
    json_path = output_dir / f"evaluation_report_{dataset_source}.json"

    payload = {
        "title": "CineMatch Evaluation Report",
        "generated_at": generated_at,
        "dataset_source": dataset_source,
        "top_k": top_k,
        "metrics_table": metrics_df.round(4).to_dict(orient="records"),
        "relevance_chart": evaluation["charts"]["relevance_chart"],
        "business_chart": evaluation["charts"]["business_chart"],
        "metadata": service.metadata,
    }
    render_html_report(settings.templates_dir, payload, html_path)
    save_json(payload, json_path)

    latest = {
        "generated_at": generated_at,
        "dataset_source": dataset_source,
        "top_k": top_k,
        "html_path": str(html_path),
        "json_path": str(json_path),
        "summary": metrics_df.round(6).to_dict(orient="records"),
    }
    settings.latest_report_path.write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding="utf-8")
    return latest


def build_training_bundle(
    users: pd.DataFrame,
    items: pd.DataFrame,
    interactions_for_service: pd.DataFrame,
    retriever: LatentRetriever,
    rank_pipeline: Any,
    feature_builder: FeatureBuilder,
    metadata: dict[str, Any],
    latest_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    history = interactions_for_service.groupby("user_id")["item_id"].apply(lambda s: set(s.astype(str))).to_dict()
    item_feature_matrix = feature_builder.genre_matrix
    item_popularity = feature_builder.item_popularity.to_dict()
    return {
        "users": users,
        "items": items,
        "retriever": retriever,
        "rank_pipeline": rank_pipeline,
        "feature_builder": feature_builder,
        "history": history,
        "item_feature_matrix": item_feature_matrix,
        "item_popularity": item_popularity,
        "metadata": metadata,
        "latest_metrics": latest_metrics or {},
    }


def save_service_bundle(bundle: dict[str, Any], artifact_path: Path) -> None:
    save_pickle(bundle, artifact_path)


def load_service_bundle(artifact_path: Path) -> dict[str, Any]:
    return load_pickle(artifact_path)
