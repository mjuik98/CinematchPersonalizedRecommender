from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse
import json
from pathlib import Path

from app.config import settings
from app.data import build_positive_interactions, load_dataset, temporal_leave_k_out_split
from app.service import RecommendationService, export_evaluation_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an existing artifact with the held-out split.")
    parser.add_argument("--dataset-source", default=settings.dataset_source)
    parser.add_argument("--artifact-path", default=str(settings.artifact_path))
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_source, sample_dir=settings.sample_data_dir, raw_dir=settings.raw_dir)
    positives = build_positive_interactions(dataset.interactions, positive_threshold=settings.positive_threshold)
    _, _, test_df = temporal_leave_k_out_split(positives)
    service = RecommendationService.from_path(Path(args.artifact_path))
    report = export_evaluation_report(service, test_df=test_df, output_dir=settings.reports_dir, dataset_source=dataset.source_name, top_k=args.top_k)
    print(json.dumps(report, ensure_ascii=False, indent=2))