import unittest
from pathlib import Path

from app.config import settings
from app.data import create_synthetic_sample_dataset
from app.service import RecommendationService
from scripts.train_pipeline import run_training


class ServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        create_synthetic_sample_dataset(settings.sample_data_dir)
        run_training("sample", 4.0, artifact_path=settings.artifact_path)
        cls.service = RecommendationService.from_path(settings.artifact_path)

    def test_personalized_recommendations(self):
        user_id = self.service.users["user_id"].iloc[0]
        recs = self.service.recommend_for_user(str(user_id), top_k=5, log=False)
        self.assertEqual(len(recs), 5)
        self.assertTrue(all("reasons" in row for row in recs))

    def test_cold_start_recommendations(self):
        recs = self.service.recommend_cold_start(top_k=5, favorite_genres=["Comedy"], log=False)
        self.assertEqual(len(recs), 5)


if __name__ == "__main__":
    unittest.main()
