#!/usr/bin/env python
"""Launch the FastAPI inference server."""
from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/deployment/api.yaml")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import os

    os.environ["LLMFORGE_DEPLOYMENT_CONFIG"] = args.config

    from llmforge.config import load_deployment_config

    cfg = load_deployment_config(args.config)
    uvicorn.run("llmforge.deployment.api:app", host=cfg.host, port=cfg.port, reload=args.reload)


if __name__ == "__main__":
    main()
