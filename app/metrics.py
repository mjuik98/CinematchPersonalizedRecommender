from __future__ import annotations

from itertools import combinations
from math import log2
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def precision_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    recs = list(recommended)[:k]
    if not recs:
        return 0.0
    hits = sum(1 for item in recs if item in relevant)
    return hits / k


def recall_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    recs = list(recommended)[:k]
    hits = sum(1 for item in recs if item in relevant)
    return hits / len(relevant)


def hit_rate_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(item in relevant for item in list(recommended)[:k]) else 0.0


def mrr_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    for rank, item in enumerate(list(recommended)[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    score = 0.0
    hits = 0
    for rank, item in enumerate(list(recommended)[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / rank
    denominator = min(len(relevant), k)
    return score / denominator if denominator else 0.0


def ndcg_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    recs = list(recommended)[:k]
    dcg = 0.0
    for i, item in enumerate(recs, start=1):
        rel = 1.0 if item in relevant else 0.0
        dcg += rel / log2(i + 1)
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    ideal_dcg = sum(1.0 / log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def average_intra_list_diversity(recommendations: Mapping[str, Sequence[str]], item_feature_matrix: pd.DataFrame, k: int) -> float:
    scores = []
    feature_lookup = item_feature_matrix
    for recs in recommendations.values():
        items = [item for item in list(recs)[:k] if item in feature_lookup.index]
        if len(items) < 2:
            continue
        vectors = feature_lookup.loc[items].to_numpy(dtype=float)
        norms = np.linalg.norm(vectors, axis=1)
        pair_scores = []
        for i, j in combinations(range(len(items)), 2):
            denom = norms[i] * norms[j]
            if denom == 0:
                pair_scores.append(0.0)
                continue
            cosine = float(np.dot(vectors[i], vectors[j]) / denom)
            pair_scores.append(1.0 - cosine)
        if pair_scores:
            scores.append(float(np.mean(pair_scores)))
    return float(np.mean(scores)) if scores else 0.0


def catalog_coverage(recommendations: Mapping[str, Sequence[str]], total_items: int, k: int) -> float:
    if total_items <= 0:
        return 0.0
    unique_items = {item for recs in recommendations.values() for item in list(recs)[:k]}
    return len(unique_items) / total_items


def novelty_at_k(recommendations: Mapping[str, Sequence[str]], item_popularity: Mapping[str, float], user_count: int, k: int) -> float:
    scores = []
    denom = max(user_count, 1)
    for recs in recommendations.values():
        for item in list(recs)[:k]:
            popularity = max(float(item_popularity.get(item, 0.0)), 1.0)
            prob = popularity / denom
            scores.append(-log2(prob))
    return float(np.mean(scores)) if scores else 0.0


def personalization_at_k(recommendations: Mapping[str, Sequence[str]], k: int, max_users: int = 300) -> float:
    user_ids = list(recommendations)
    if len(user_ids) < 2:
        return 0.0
    if len(user_ids) > max_users:
        user_ids = user_ids[:max_users]
    overlaps = []
    for u1, u2 in combinations(user_ids, 2):
        s1 = set(list(recommendations[u1])[:k])
        s2 = set(list(recommendations[u2])[:k])
        overlaps.append(len(s1.intersection(s2)) / max(k, 1))
    if not overlaps:
        return 0.0
    return 1.0 - float(np.mean(overlaps))


def long_tail_share_at_k(recommendations: Mapping[str, Sequence[str]], item_popularity: Mapping[str, float], k: int, quantile: float = 0.8) -> float:
    if not item_popularity:
        return 0.0
    threshold = np.quantile(list(item_popularity.values()), quantile)
    items = [item for recs in recommendations.values() for item in list(recs)[:k]]
    if not items:
        return 0.0
    hits = sum(1 for item in items if item_popularity.get(item, 0.0) <= threshold)
    return hits / len(items)
