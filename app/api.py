from __future__ import annotations

from pathlib import Path
import json

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

from .config import settings
from .schemas import ColdStartRequest, FeedbackRequest
from .service import RecommendationService


def create_app(artifact_path: Path | None = None) -> FastAPI:
    artifact_path = artifact_path or settings.artifact_path
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}. Run the training pipeline first.")
    service = RecommendationService.from_path(artifact_path)
    templates = Jinja2Templates(directory=str(settings.templates_dir))

    app = FastAPI(title="CineMatch Personalized Recommender", version="1.0.0")
    app.state.service = service

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request) -> HTMLResponse:
        metrics = service.latest_metrics_payload()
        metadata = service.metadata_payload()
        example_user_id = metadata.get("sample_user_id", service.users["user_id"].iloc[0])
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "metrics": metrics.get("summary", []),
                "metadata": metadata,
                "example_user_id": example_user_id,
            },
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metadata")
    def metadata() -> dict:
        return service.metadata_payload()

    @app.get("/metrics/latest")
    def metrics() -> dict:
        return service.latest_metrics_payload()

    @app.get("/users/{user_id}/recommendations")
    def recommend_user(
        user_id: str,
        top_k: int = Query(default=settings.top_k_default, ge=1, le=50),
        candidate_k: int = Query(default=settings.candidate_k_default, ge=10, le=500),
        diversity_lambda: float = Query(default=settings.diversity_lambda_default, ge=0.0, le=1.0),
    ) -> dict:
        return {
            "user_id": user_id,
            "recommendations": service.recommend_for_user(
                user_id=user_id,
                top_k=top_k,
                candidate_k=candidate_k,
                diversity_lambda=diversity_lambda,
                log=True,
            ),
        }

    @app.post("/cold-start/recommendations")
    def recommend_cold_start(payload: ColdStartRequest) -> dict:
        return {
            "recommendations": service.recommend_cold_start(
                top_k=payload.top_k,
                age_bucket=payload.age_bucket,
                gender=payload.gender,
                occupation=payload.occupation,
                favorite_genres=payload.favorite_genres,
            )
        }

    @app.get("/items/{item_id}")
    def item_details(item_id: str) -> dict:
        try:
            return service.item_details(item_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown item: {item_id}") from exc

    @app.get("/items/{item_id}/similar")
    def similar_items(item_id: str, top_k: int = Query(default=10, ge=1, le=50)) -> dict:
        return {"item_id": item_id, "similar_items": service.similar_items(item_id=item_id, top_k=top_k)}

    @app.post("/feedback")
    def feedback(payload: FeedbackRequest) -> dict:
        service.save_feedback(
            user_id=payload.user_id,
            item_id=payload.item_id,
            event_type=payload.event_type,
            value=payload.value,
            context=payload.context,
        )
        return {"status": "saved"}

    @app.get("/analytics/summary")
    def analytics() -> dict:
        return service.analytics_summary()

    return app


app = None
try:
    app = create_app()
except FileNotFoundError:
    # Allows importing before training.
    app = FastAPI(title="CineMatch Personalized Recommender", version="1.0.0")
