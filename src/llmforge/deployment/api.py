from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from llmforge.config import DeploymentConfig, load_deployment_config
from llmforge.deployment.schemas import GenerateRequest, GenerateResponse, HealthResponse
from llmforge.inference.engine import InferenceEngine

_state: dict[str, InferenceEngine | DeploymentConfig] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path = os.environ.get("LLMFORGE_DEPLOYMENT_CONFIG", "configs/deployment/api.yaml")
    cfg = load_deployment_config(config_path)
    _state["config"] = cfg
    _state["engine"] = InferenceEngine(model_path=cfg.model_path, device=cfg.device)
    yield
    _state.clear()


app = FastAPI(title="llmforge inference API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    cfg: DeploymentConfig = _state["config"]
    return HealthResponse(status="ok", model_path=cfg.model_path)


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    cfg: DeploymentConfig = _state["config"]
    engine: InferenceEngine = _state["engine"]
    completion = engine.generate(
        request.prompt,
        max_new_tokens=request.max_new_tokens or cfg.max_new_tokens,
        temperature=request.temperature if request.temperature is not None else cfg.temperature,
        top_p=request.top_p if request.top_p is not None else cfg.top_p,
    )
    return GenerateResponse(completion=completion)
