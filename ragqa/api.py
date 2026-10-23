"""Optional FastAPI adapter. The pipeline is injected once, not constructed per request."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ragqa.pipeline import RAGPipeline
from ragqa.types import Answer, InputError


class QuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=2000, strict=True)


def create_app(pipeline: RAGPipeline) -> FastAPI:
    app = FastAPI(title="Chinese Medical RAG Research API", version="0.2.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "model_warmup": "lazy", "clinical_validation": False}

    @app.post("/v1/answers", response_model=Answer)
    def answer(request: QuestionRequest):
        try:
            result = pipeline.run(request.question)
        except InputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if result.status == "error":
            from fastapi.responses import JSONResponse

            headers = {"Retry-After": "1"} if result.reason == "busy" else None
            return JSONResponse(result.to_dict(), status_code=503, headers=headers)
        return result

    return app
