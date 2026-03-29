import unittest

from fastapi.testclient import TestClient

from app.api import create_app
from app.config import settings
from app.data import create_synthetic_sample_dataset
from scripts.train_pipeline import run_training


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        create_synthetic_sample_dataset(settings.sample_data_dir)
        run_training("sample", 4.0, artifact_path=settings.artifact_path)
        cls.client = TestClient(create_app(settings.artifact_path))

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

    def test_recommend_and_feedback(self):
        response = self.client.get("/users/1/recommendations?top_k=5")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["recommendations"]), 5)
        item_id = payload["recommendations"][0]["item_id"]

        feedback = self.client.post(
            "/feedback",
            json={"user_id": "1", "item_id": item_id, "event_type": "click", "value": 1},
        )
        self.assertEqual(feedback.status_code, 200)


if __name__ == "__main__":
    unittest.main()
