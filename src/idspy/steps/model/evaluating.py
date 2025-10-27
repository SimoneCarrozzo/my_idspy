from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ...core.state import State
from ...core.step import Step
from ...nn.models.base import BaseModel, ModelOutput
from ...nn.losses.base import BaseLoss
from ...nn.batch import ensure_batch
from ...nn.helpers import run_epoch


"""
# Nel main
model = TabularClassifier(...)  # Tipo concreto
state.set("model", model, TabularClassifier)

# In TrainOneEpoch
return {"model": model}
# → Step cerca di fare: state.set("train.model", model, BaseModel)
# ❌ ERRORE! Già esiste con tipo TabularClassifier!

# SOLUZIONE: Non restituire model (è già nello state e viene modificato in-place), e settalo dove serve come nn.Module
"""

class ValidateOneEpoch(Step):
    """Validate a model for one epoch (no gradient updates)."""

    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_prefix: str = "val",
        save_history: bool = False,
        save_outputs: bool = False,
        in_scope: str = "val",
        out_scope: Optional[str] = "val",
        name: Optional[str] = None,
    ) -> None:
        self.writer: Optional[SummaryWriter] = (
            SummaryWriter(log_dir) if log_dir else None
        )
        self.log_prefix = log_prefix
        self.save_history = save_history
        self.save_outputs = save_outputs

        super().__init__(
            name=name or "validate_one_epoch",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    @Step.requires( #cambiato model=BaseModel -> model=Any
        dataloader=torch.utils.data.DataLoader,
        model=nn.Module,
        loss=BaseLoss,
        device=torch.device,
        history=list,
        outputs=list,
        epoch=int,
    )
    @Step.provides(history=list, outputs=list, epoch=int, loss=float) #rimosso model=BaseModel, aggiunto loss=float che prima non c'era perchè prima era train senza early stop
    def run(
        self,
        state: State,
        dataloader: torch.utils.data.DataLoader,
        #model: BaseModel
        model: nn.Module,
        loss: BaseLoss,
        device: torch.device,
        context: Optional[any] = None,
        history: Optional[list] = None,
        outputs: Optional[list] = None,
        # history: list = [],
        # outputs: list = [],
        epoch: int = 0,
    ) -> Optional[Dict[str, Any]]:
        #creazione di nuove liste se none
        if history is None:
            history = []
        if outputs is None:
            outputs = []
        #se save_outputs è true, sovrascrivo invece di appendere, così da prevenire errori
        #l'accumulo tra le epoche, dato che resetto all'inizio di ogni validazione
        if self.save_outputs:
            outputs = []
        average_loss, outputs_list = run_epoch(
            desc="Validation",
            log_prefix=self.log_prefix,
            is_training=False,
            dataloader=dataloader,
            model=model,
            device=device,
            loss_fn=loss,
            optimizer=None,
            writer=self.writer,
            profiler=context,
            clip_grad_max_norm=None,
            save_outputs=self.save_outputs,
            epoch=epoch,
        )

        if self.writer is not None:
            self.writer.close()

        if self.save_history:
            history.append(average_loss)
        if self.save_outputs:
            outputs.extend(outputs_list)

        return {
            # "model": model,
            "history": history,
            "outputs": outputs,
            "epoch": epoch + 1,
            "loss": average_loss,
        }


class ForwardOnce(Step):
    """Compute a single forward pass: model(input_tensor) -> output."""

    def __init__(
        self,
        to_cpu: bool = False,  # move output to CPU before storing
        in_scope: Optional[str] = "test",
        out_scope: Optional[str] = "test",
        name: Optional[str] = None,
    ) -> None:
        self.to_cpu = to_cpu

        super().__init__(
            name=name or "forward_once",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    @Step.requires(
        batch=torch.Tensor,
        model=BaseModel,
        device=torch.device,
    )
    @Step.provides(output=torch.Tensor)
    def run(
        self, state: State, batch: torch.Tensor, model: BaseModel, device: torch.device
    ) -> Optional[Dict[str, Any]]:
        batch = ensure_batch(batch)

        model.eval()
        batch = batch.to(device, non_blocking=True)

        with torch.no_grad():
            out: ModelOutput = model(batch.features)

        if self.to_cpu and torch.is_tensor(out.logits):
            out = out.detach().cpu().numpy()

        return {"output": out}


class MakePredictions(Step):
    """Make predictions from model outputs."""

    def __init__(
        self,
        pred_fn: Callable,
        in_scope: Optional[str] = "test",
        out_scope: Optional[str] = "test",
        name: Optional[str] = None,
    ) -> None:
        self.pred_fn = pred_fn

        super().__init__(
            name=name or "make_predictions",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    @Step.requires(outputs=list)
    @Step.provides(predictions=np.ndarray)
    def run(self, state: State, outputs: list) -> Optional[Dict[str, Any]]:
        predictions = []
        for output in tqdm(outputs, desc="Making predictions", unit="batch"):
            curr_pred = self.pred_fn(output.logits)
            if not torch.is_tensor(curr_pred):
                raise TypeError(
                    f"Expected tensor predictions from 'pred_fn' for step '{self.name}'."
                )
            predictions.append(curr_pred)
        predictions = torch.cat(predictions, dim=0)

        predictions = predictions.detach().cpu().numpy()
        return {"predictions": predictions}
