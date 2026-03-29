# CineMatch: 다단계 개인화 추천 시스템 최종본

CineMatch는 **개인화 추천 시스템 포트폴리오 최종 제출형 프로젝트**입니다.  
MovieLens 1M 또는 내장 synthetic sample 데이터를 기준으로 아래 전체 흐름을 재현합니다.

- 데이터 다운로드/전처리
- **협업 필터링 기반 후보 생성**
- **콘텐츠·사용자 메타데이터 기반 재랭킹**
- **다양성(MMR) 리랭킹**
- 오프라인 평가 리포트 생성
- FastAPI 서빙 + HTML 대시보드
- SQLite 기반 추천/피드백 로그
- Docker / unittest / GitHub Actions

## 왜 이 구조인가
추천 시스템은 보통 **retrieval → ranking**의 2단계 또는 다단계 구조로 설계됩니다. TensorFlow Recommenders 공식 문서는 retrieval 단계가 수많은 후보 중 소수의 후보를 효율적으로 뽑고, ranking 단계가 그 후보를 더 정교하게 재정렬한다고 설명합니다. MovieLens 1M은 GroupLens의 안정 벤치마크이며 약 1백만 건의 평점, 6,040명의 사용자, 약 4천 개 영화로 구성되어 있고 demographic data를 포함하는 대표 공개 버전 중 하나입니다. 평가도 relevance만 보지 않고 NDCG, MAP, MRR, diversity, novelty, personalization 같은 ranking/recommender 지표를 함께 보는 것이 일반적입니다.

참고:
- GroupLens MovieLens 1M: https://grouplens.org/datasets/movielens/1m/
- TensorFlow Recommenders retrieval/ranking: https://www.tensorflow.org/recommenders/examples/basic_retrieval / https://www.tensorflow.org/recommenders/examples/basic_ranking
- Ranking metrics explainer: https://docs.evidentlyai.com/metrics/explainer_recsys

## 프로젝트 핵심
- **데이터셋**: MovieLens 1M 다운로드 스크립트 제공, synthetic sample 내장
- **후보 생성기**: sparse interaction matrix + TruncatedSVD latent retrieval
- **랭커**: candidate score + popularity + novelty + genre match + cohort popularity + demographic/meta features
- **리랭커**: MMR 기반 diversity reranking + 장르 쏠림 제한
- **콜드스타트**: 연령대/성별/선호 장르 기반 인기 추천
- **평가 지표**
  - Precision@K / Recall@K / HitRate@K
  - MAP@K / MRR@K / NDCG@K
  - Catalog Coverage / Novelty / Diversity / Personalization / Long-tail share
- **서빙**
  - `/users/{user_id}/recommendations`
  - `/cold-start/recommendations`
  - `/items/{item_id}/similar`
  - `/feedback`
  - `/analytics/summary`

## 아키텍처
![Architecture](docs/assets/architecture_overview.png)

## 폴더 구조
```text
cinematch_personalized_recommender/
├── app/
│   ├── api.py
│   ├── config.py
│   ├── data.py
│   ├── metrics.py
│   ├── modeling.py
│   ├── reporting.py
│   ├── schemas.py
│   ├── service.py
│   ├── storage.py
│   └── templates/
├── data/
│   └── sample/
├── docs/
├── scripts/
├── tests/
├── storage/
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

## 빠른 실행
### 1) 가상환경
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) synthetic sample로 바로 실행
```bash
make seed-sample
make train-sample
make smoke
make run
```

브라우저에서:
```text
http://127.0.0.1:8000
```

### 3) MovieLens 1M로 실행
```bash
make download-ml1m
make train-ml1m
make export-report
make run
```

## Make 명령어
```bash
make seed-sample        # synthetic sample 생성
make train-sample       # sample 데이터 학습 + 평가
make download-ml1m      # MovieLens 1M 다운로드
make train-ml1m         # MovieLens 1M 학습 + 평가
make export-report      # latest report -> docs/sample_eval_report.html
make run                # FastAPI 실행
make smoke              # end-to-end smoke test
make test               # unittest 실행
make clean              # 산출물/로그 초기화
```

## API 예시
### 기존 사용자 추천
```bash
curl "http://127.0.0.1:8000/users/1/recommendations?top_k=10&diversity_lambda=0.88"
```

### 콜드스타트 추천
```bash
curl -X POST "http://127.0.0.1:8000/cold-start/recommendations" \
  -H "Content-Type: application/json" \
  -d '{"age_bucket":"25-34","gender":"F","favorite_genres":["Drama","Romance"],"top_k":10}'
```

### 피드백 저장
```bash
curl -X POST "http://127.0.0.1:8000/feedback" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"1","item_id":"50","event_type":"click","value":1}'
```

## 학습/평가 설계
### 데이터 분할
- explicit rating 중 **rating >= 4**를 positive interaction으로 사용
- 사용자별 시간 순 정렬 후 **train / validation / test** leave-k-out split
- ranker는 validation candidate set으로 학습
- offline report는 held-out test로 계산

### 모델 비교
리포트는 최소 3개 모델을 함께 비교합니다.
1. **Popularity**
2. **Latent Retriever**
3. **Final Multi-Stage**

즉, 단순 baseline과 최종 시스템을 함께 보여줘서 포트폴리오 설명이 쉽습니다.

## 보고서
- HTML 보고서: `storage/reports/evaluation_report_<dataset>.html`
- JSON 요약: `storage/reports/evaluation_report_<dataset>.json`
- 제출용 샘플: `docs/sample_eval_report.html`

## 검증
이 저장소는 로컬에서 아래 기준으로 검증했습니다.
- `python -m unittest discover -s tests -v`
- `python scripts/smoke_test.py`

실제 검증 기록은 [`docs/validation_report.md`](docs/validation_report.md)에 정리되어 있습니다.

## 문서
- [포트폴리오 소개글](docs/portfolio_intro_ko.md)
- [이력서 bullet](docs/resume_bullets.md)
- [데모 스크립트](docs/demo_script.md)
- [면접 Q&A](docs/interview_qa.md)
- [GitHub 업로드 가이드](docs/github_publish_guide.md)
- [샘플 평가 리포트](docs/sample_eval_report.html)
- [최종 제출 체크리스트](docs/final_submission_checklist.md)

## 라이선스
- **코드**: MIT License
- **MovieLens 데이터셋**: GroupLens 사용 조건을 따르며, 이 저장소에는 데이터셋 본문을 재배포하지 않습니다. GroupLens README는 MovieLens 1M을 연구 목적으로 사용할 수 있지만 재배포는 별도 허가가 필요하다고 명시합니다.
