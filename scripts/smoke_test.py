from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from pathlib import Path

from fastapi.testclient import TestClient

from app.api import create_app
from app.config import settings
from app.data import create_synthetic_sample_dataset
from app.service import RecommendationService
from scripts.train_pipeline import run_training


def main() -> None:
    create_synthetic_sample_dataset(settings.sample_data_dir)
    result = run_training(dataset_source="sample", positive_threshold=4.0, artifact_path=settings.artifact_path)
    service = RecommendationService.from_path(settings.artifact_path)

    user_id = service.users["user_id"].iloc[0]
    recs = service.recommend_for_user(str(user_id), top_k=5, log=False)
    assert len(recs) == 5
    assert all("item_id" in row for row in recs)

    cold = service.recommend_cold_start(top_k=5, favorite_genres=["Drama", "Romance"], log=False)
    assert len(cold) == 5

    app = create_app(settings.artifact_path)
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    response = client.get(f"/users/{user_id}/recommendations?top_k=5")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["recommendations"]) == 5

    item_id = recs[0]["item_id"]
    assert client.get(f"/items/{item_id}").status_code == 200
    assert client.get(f"/items/{item_id}/similar?top_k=3").status_code == 200
    assert client.get("/analytics/summary").status_code == 200

    print("Smoke test passed.")


if __name__ == "__main__":
    main()