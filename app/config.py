from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass(slots=True)
class Settings:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    storage_dir: Path = field(init=False)
    raw_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    templates_dir: Path = field(init=False)
    data_dir: Path = field(init=False)
    sample_data_dir: Path = field(init=False)
    artifact_path: Path = field(init=False)
    sqlite_path: Path = field(init=False)
    latest_report_path: Path = field(init=False)

    dataset_source: str = field(default_factory=lambda: os.getenv("DATASET_SOURCE", "sample"))
    positive_threshold: float = field(default_factory=lambda: float(os.getenv("POSITIVE_THRESHOLD", "4.0")))
    top_k_default: int = field(default_factory=lambda: int(os.getenv("TOP_K_DEFAULT", "10")))
    candidate_k_default: int = field(default_factory=lambda: int(os.getenv("CANDIDATE_K_DEFAULT", "120")))
    diversity_lambda_default: float = field(default_factory=lambda: float(os.getenv("DIVERSITY_LAMBDA_DEFAULT", "0.88")))
    min_history_for_personalization: int = field(default_factory=lambda: int(os.getenv("MIN_HISTORY_FOR_PERSONALIZATION", "5")))

    def __post_init__(self) -> None:
        self.storage_dir = self.project_root / "storage"
        self.raw_dir = self.storage_dir / "raw"
        self.processed_dir = self.storage_dir / "processed"
        self.artifacts_dir = self.storage_dir / "artifacts"
        self.reports_dir = self.storage_dir / "reports"
        self.logs_dir = self.storage_dir / "logs"
        self.templates_dir = self.project_root / "app" / "templates"
        self.data_dir = self.project_root / "data"
        self.sample_data_dir = self.data_dir / "sample"
        self.artifact_path = self.artifacts_dir / "service_bundle.pkl"
        self.sqlite_path = self.logs_dir / "recommendation_app.db"
        self.latest_report_path = self.reports_dir / "latest_report.json"

        for path in [
            self.storage_dir,
            self.raw_dir,
            self.processed_dir,
            self.artifacts_dir,
            self.reports_dir,
            self.logs_dir,
            self.sample_data_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
