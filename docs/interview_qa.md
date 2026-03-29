# 예상 면접 질문

## 왜 MovieLens 1M을 선택했나요?
재현성이 좋고 안정적인 공개 벤치마크라서 학습/평가/포트폴리오 설명이 쉽습니다. 또 사용자 demographic 정보가 있어 콜드스타트와 cohort popularity를 실험하기 좋습니다.

## 왜 retrieval + ranking 구조로 설계했나요?
실서비스 추천 시스템은 전체 아이템을 매번 정교하게 점수화하기보다 먼저 후보를 좁히고, 그다음 더 비싼 랭킹 모델을 적용하는 구조가 일반적이기 때문입니다.

## ranker에 어떤 특징을 넣었나요?
candidate score, popularity, item 평균 평점, user 평균 평점, user interaction 수, genre match, novelty, cohort popularity, release year, primary genre, gender, age bucket, occupation 등을 사용했습니다.

## diversity reranking은 왜 필요했나요?
추천 정확도만 최적화하면 한 장르로 쏠리기 쉬워서 사용자 경험이 단조로워질 수 있습니다. 그래서 MMR을 적용해 relevance와 diversity를 균형 있게 맞췄습니다.

## offline evaluation에서 본 지표는?
Precision@K, Recall@K, HitRate@K, MAP@K, MRR@K, NDCG@K를 relevance용으로, Coverage, Novelty, Diversity, Personalization, Long-tail share를 catalog/behavior 지표로 봤습니다.

## 콜드스타트는 어떻게 처리했나요?
신규 사용자는 인기 추천, cohort popularity, favorite genre match를 혼합한 규칙 기반 스코어링으로 대응했습니다.
