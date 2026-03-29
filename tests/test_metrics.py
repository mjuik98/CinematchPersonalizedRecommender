import unittest

from app.metrics import (
    average_precision_at_k,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class MetricsTest(unittest.TestCase):
    def test_basic_metrics(self):
        recs = ["a", "b", "c", "d"]
        relevant = {"b", "d"}
        self.assertAlmostEqual(precision_at_k(recs, relevant, 2), 0.5)
        self.assertAlmostEqual(recall_at_k(recs, relevant, 4), 1.0)
        self.assertEqual(hit_rate_at_k(recs, relevant, 1), 0.0)
        self.assertAlmostEqual(mrr_at_k(recs, relevant, 4), 0.5)
        self.assertGreater(ndcg_at_k(recs, relevant, 4), 0.0)
        self.assertGreater(average_precision_at_k(recs, relevant, 4), 0.0)


if __name__ == "__main__":
    unittest.main()
