from typing import Optional

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from ...core.step import Step
from ...core.state import State
from ...steps.helpers import validate_instance


class ClassificationMetrics(Step):
    """Compute metrics for multiclass classification."""

    def __init__(
        self,
        pred_in: str = "history.val.preds",
        target_in: str = "data.val.target",
        metric_out: str = "history.val.metrics",
        log_dir: Optional[str] = None,
        log_prefix: str = "Train",
        name: Optional[str] = None,
    ) -> None:
        self.pred_in = pred_in
        self.target_in = target_in
        self.metric_out = metric_out
        self.writer: Optional[SummaryWriter] = (
            SummaryWriter(log_dir) if log_dir else None
        )
        self.log_prefix = log_prefix

        super().__init__(
            requires=[self.pred_in],
            provides=[self.metric_out],
            name=name or "multiclass_classification_metrics",
        )

    def compute_metrics(self, y_pred: np.ndarray, y_true: np.ndarray) -> dict:
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
            "precision": precision_score(y_true, y_pred, average="macro"),
            "recall": recall_score(y_true, y_pred, average="macro"),
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "f1_weighted": f1_weighted,
            "f1_per_class": f1_per_class,
            "confusion_matrix": cm,
        }

        return metrics

    def run(self, state: State) -> None:
        y_pred = state[self.pred_in]
        validate_instance(y_pred, np.ndarray, self.name)

        y_true = state[self.target_in]
        validate_instance(y_true, np.ndarray, self.name)

        # Compute metrics (e.g., accuracy, F1 score) using y_pred and y_true
        metrics = self.compute_metrics(y_pred, y_true)

        for name, value in metrics.items():
            if isinstance(value, (int, float)) and self.writer is not None:
                self.writer.add_scalar(f"{self.log_prefix}/{name}", value)

        if self.writer is not None:
            self.writer.close()

        state[self.metric_out] = metrics
