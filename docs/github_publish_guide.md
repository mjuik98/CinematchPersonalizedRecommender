# GitHub 업로드 가이드

## 1. 새 저장소 생성
GitHub에서 새 public repository를 생성합니다. 예:
- `cinematch-personalized-recommender`

## 2. 로컬 초기화
```bash
git init
git add .
git commit -m "Initial commit: final personalized recommender portfolio"
git branch -M main
git remote add origin <YOUR_REPO_URL>
git push -u origin main
```

## 3. README 상단에 넣으면 좋은 문장
```text
CineMatch is a final portfolio project for a multi-stage personalized recommendation system.
It covers candidate retrieval, ranking, diversity re-ranking, offline evaluation, API serving, and feedback analytics in one reproducible package.
```

## 4. 추천 repository topic
- recommender-system
- personalization
- machine-learning
- fastapi
- mlops
- ranking
- collaborative-filtering

## 5. 제출 체크
- README 최신화
- docs/sample_eval_report.html 링크 확인
- storage/artifacts, storage/logs 비워진 상태 확인
- `.gitignore`로 raw data 제외 확인
