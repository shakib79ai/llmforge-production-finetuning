from __future__ import annotations

from typing import Any


class ExperimentTracker:
    """Thin wrapper around wandb so callers don't import/initialize it
    directly and so tracking can be disabled without touching call sites."""

    def __init__(self, project: str, run_name: str | None = None, enabled: bool = True):
        self.enabled = enabled
        self._run = None
        if self.enabled:
            import wandb

            self._run = wandb.init(project=project, name=run_name)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self.enabled and self._run is not None:
            self._run.log(metrics, step=step)

    def log_artifact(self, path: str, name: str, artifact_type: str = "model") -> None:
        if self.enabled and self._run is not None:
            import wandb

            artifact = wandb.Artifact(name, type=artifact_type)
            artifact.add_dir(path)
            self._run.log_artifact(artifact)

    def finish(self) -> None:
        if self.enabled and self._run is not None:
            self._run.finish()
