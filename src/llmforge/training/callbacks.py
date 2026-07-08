from __future__ import annotations

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class EvalLoggingCallback(TrainerCallback):
    """Prints a compact eval summary line so key metrics are visible in plain
    console logs, not just in the experiment tracker UI."""

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict | None = None,
        **kwargs,
    ):
        if metrics is None:
            return
        summary = ", ".join(
            f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))
        )
        print(f"[eval @ step {state.global_step}] {summary}")
