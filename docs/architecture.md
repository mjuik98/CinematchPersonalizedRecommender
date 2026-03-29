# 시스템 설계 요약

1. 데이터 레이어
- MovieLens 1M 다운로드 또는 synthetic sample 로드
- users / items / interactions 통합 스키마로 정규화
- rating >= 4를 implicit positive로 변환

2. 모델 레이어
- Latent Retriever: sparse user-item matrix + TruncatedSVD
- Ranker: collaborative score + metadata feature + logistic regression
- Reranker: MMR 기반 diversity 조정

3. 서빙 레이어
- FastAPI
- personalized / cold-start / similar-items / feedback / analytics endpoints
- HTML dashboard

4. 평가 레이어
- train / validation / test temporal split
- baseline 대비 최종 모델 비교
- HTML report 자동 생성
