from typing import Optional, Dict, Any

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from ...core.step import Step
from ...core.state import State


class ClassificationMetrics(Step):
    """Compute metrics for multiclass classification."""

    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_prefix: str = "test",
        in_scope: str = "test",
        out_scope: str = "test",
        name: Optional[str] = None,
    ) -> None:
        self.writer: Optional[SummaryWriter] = (
            SummaryWriter(log_dir) if log_dir else None
        )
        self.log_prefix = log_prefix

        super().__init__(
            name=name or "multiclass_classification_metrics",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    def compute_metrics(self, y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """Compute classification metrics."""
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            confusion_matrix,
        )

        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_micro = f1_score(y_true, y_pred, average="micro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")

        cm = confusion_matrix(y_true, y_pred)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "f1_weighted": f1_weighted,
            "f1_per_class": f1_per_class,
            "confusion_matrix": cm,
        }

        return metrics

    @Step.requires(
        predictions=np.ndarray,
        targets=np.ndarray,
    )
    @Step.provides(metrics=dict)
    def run(self, state: State, predictions: np.ndarray, targets: np.ndarray) -> None:
        metrics = self.compute_metrics(predictions, targets)

        for name, value in metrics.items():
            if isinstance(value, (int, float)) and self.writer is not None:
                self.writer.add_scalar(f"{self.log_prefix}/{name}", value)

        if self.writer is not None:
            self.writer.close()

        return {"metrics": metrics}
