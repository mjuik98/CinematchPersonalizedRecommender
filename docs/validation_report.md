# 로컬 검증 기록

## 실행 환경에서 직접 확인한 항목
### 1) synthetic sample 데이터셋 생성
```bash
python scripts/create_synthetic_demo_data.py
```

생성 결과:
- users: 30
- items: 90
- interactions: 916

### 2) sample 기준 학습 + 평가
```bash
python scripts/train_pipeline.py --dataset-source sample
```

검증 결과:
- artifact 생성: `storage/artifacts/service_bundle.pkl`
- HTML report 생성: `storage/reports/evaluation_report_sample.html`
- JSON report 생성: `storage/reports/evaluation_report_sample.json`

오프라인 지표(sample):
| model | hit_rate@10 | ndcg@10 | mrr@10 | coverage | diversity | personalization |
|---|---:|---:|---:|---:|---:|---:|
| Popularity | 0.0333 | 0.0079 | 0.0067 | 0.2333 | 0.7386 | 0.4425 |
| Latent Retriever | 0.3000 | 0.0679 | 0.0544 | 0.3667 | 0.8170 | 0.6572 |
| Final Multi-Stage | 0.7667 | 0.4444 | 0.4761 | 0.6444 | 0.8076 | 0.6975 |

### 3) smoke test
```bash
python scripts/smoke_test.py
```

결과:
- personalized recommendation 통과
- cold-start recommendation 통과
- `/health`, `/users/{id}/recommendations`, `/items/{id}`, `/items/{id}/similar`, `/analytics/summary` 통과

### 4) unit test 개별 실행
```bash
python -m unittest tests.test_metrics -v
python -m unittest tests.test_modeling -v
python -m unittest tests.test_service -v
python -m unittest tests.test_api -v
```

결과:
- metrics 1개 테스트 통과
- modeling 2개 테스트 통과
- service 2개 테스트 통과
- api 2개 테스트 통과

## 참고
MovieLens 1M 학습 경로는 코드와 스크립트로 완성되어 있지만, 이 빌드 환경에서는 외부 zip 다운로드가 제한되어 있어 **실제 GroupLens 원본 다운로드 단계까지는 검증하지 못했습니다.**  
대신 아래 두 가지를 반영했습니다.

1. 자동 다운로드 함수 구현
2. 자동 다운로드가 막히면 `storage/raw/ml-1m.zip` 또는 프로젝트 루트 `ml-1m.zip`에 수동 배치해서 학습 가능하도록 fallback 구현
