from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse
import json
from pathlib import Path

import pandas as pd

from app.config import settings
from app.data import (
    build_positive_interactions,
    create_synthetic_sample_dataset,
    load_dataset,
    temporal_leave_k_out_split,
)
from app.modeling import FeatureBuilder, LatentRetriever, build_ranker_training_frame
from app.service import (
    RecommendationService,
    build_training_bundle,
    export_evaluation_report,
    save_service_bundle,
)


def run_training(dataset_source: str, positive_threshold: float, artifact_path: Path | None = None) -> dict:
    if dataset_source == "sample" and not (settings.sample_data_dir / "users.csv").exists():
        create_synthetic_sample_dataset(settings.sample_data_dir)

    dataset = load_dataset(dataset_source, sample_dir=settings.sample_data_dir, raw_dir=settings.raw_dir)
    positives = build_positive_interactions(dataset.interactions, positive_threshold=positive_threshold)
    train_df, val_df, test_df = temporal_leave_k_out_split(positives)

    user_ids = dataset.users["user_id"].astype(str).tolist()
    item_ids = dataset.items["item_id"].astype(str).tolist()

    retriever = LatentRetriever(n_factors=48)
    retriever.fit(train_df, all_user_ids=user_ids, all_item_ids=item_ids)

    feature_builder = FeatureBuilder(
        users=dataset.users,
        items=dataset.items,
        interactions=train_df,
        retriever=retriever,
    )
    ranker_df = build_ranker_training_frame(retriever, feature_builder, train_df=train_df, validation_df=val_df)
    numeric_features, categorical_features = feature_builder.training_feature_columns()
    training_columns = numeric_features + categorical_features

    rank_pipeline = feature_builder.build_ranker_pipeline()
    rank_pipeline.fit(ranker_df[training_columns], ranker_df["label"])

    metadata = {
        "dataset_source": dataset.source_name,
        "user_count": int(dataset.users["user_id"].nunique()),
        "item_count": int(dataset.items["item_id"].nunique()),
        "interaction_count": int(len(dataset.interactions)),
        "positive_interaction_count": int(len(positives)),
        "train_positive_count": int(len(train_df)),
        "validation_positive_count": int(len(val_df)),
        "test_positive_count": int(len(test_df)),
        "ranker_training_rows": int(len(ranker_df)),
        "sample_user_id": str(dataset.users["user_id"].iloc[0]),
        "positive_threshold": float(positive_threshold),
    }

    # Production bundle uses train + validation interactions to improve personalization,
    # while the offline report remains based on the held-out test split.
    service_interactions = pd.concat([train_df, val_df], ignore_index=True)
    production_retriever = LatentRetriever(n_factors=48)
    production_retriever.fit(service_interactions, all_user_ids=user_ids, all_item_ids=item_ids)
    production_feature_builder = FeatureBuilder(
        users=dataset.users,
        items=dataset.items,
        interactions=service_interactions,
        retriever=production_retriever,
    )

    bundle = build_training_bundle(
        users=dataset.users,
        items=dataset.items,
        interactions_for_service=service_interactions,
        retriever=production_retriever,
        rank_pipeline=rank_pipeline,
        feature_builder=production_feature_builder,
        metadata=metadata,
    )
    artifact_path = artifact_path or settings.artifact_path
    save_service_bundle(bundle, artifact_path)

    service = RecommendationService.from_path(artifact_path)
    latest_report = export_evaluation_report(service, test_df=test_df, output_dir=settings.reports_dir, dataset_source=dataset.source_name, top_k=10)
    bundle["latest_metrics"] = latest_report
    save_service_bundle(bundle, artifact_path)
    return {
        "artifact_path": str(artifact_path),
        "latest_report": latest_report,
        "metadata": metadata,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the CineMatch personalized recommender.")
    parser.add_argument("--dataset-source", default=settings.dataset_source, help="sample | movielens_1m | path/to/csv_folder")
    parser.add_argument("--positive-threshold", type=float, default=settings.positive_threshold)
    parser.add_argument("--artifact-path", default=str(settings.artifact_path))
    args = parser.parse_args()

    result = run_training(dataset_source=args.dataset_source, positive_threshold=args.positive_threshold, artifact_path=Path(args.artifact_path))
    print(json.dumps(result, ensure_ascii=False, indent=2))