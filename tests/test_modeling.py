import unittest

from app.config import settings
from app.data import build_positive_interactions, create_synthetic_sample_dataset, temporal_leave_k_out_split
from app.modeling import FeatureBuilder, LatentRetriever, build_ranker_training_frame


class ModelingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = create_synthetic_sample_dataset(settings.sample_data_dir)
        positives = build_positive_interactions(cls.dataset.interactions)
        cls.train_df, cls.val_df, cls.test_df = temporal_leave_k_out_split(positives)
        cls.retriever = LatentRetriever(n_factors=16).fit(
            cls.train_df,
            all_user_ids=cls.dataset.users["user_id"].tolist(),
            all_item_ids=cls.dataset.items["item_id"].tolist(),
        )
        cls.feature_builder = FeatureBuilder(
            users=cls.dataset.users,
            items=cls.dataset.items,
            interactions=cls.train_df,
            retriever=cls.retriever,
        )

    def test_retriever_returns_unseen_items(self):
        user_id = self.dataset.users["user_id"].iloc[0]
        seen = set(self.train_df[self.train_df["user_id"] == user_id]["item_id"].astype(str))
        recs, _ = self.retriever.recommend(user_id, top_n=10, exclude_seen=True)
        self.assertTrue(recs)
        self.assertTrue(all(item not in seen for item in recs))

    def test_ranker_training_frame_has_positive_label(self):
        ranker_df = build_ranker_training_frame(self.retriever, self.feature_builder, self.train_df, self.val_df, candidate_k=30)
        self.assertIn("label", ranker_df.columns)
        self.assertGreater(ranker_df["label"].sum(), 0)


if __name__ == "__main__":
    unittest.main()
