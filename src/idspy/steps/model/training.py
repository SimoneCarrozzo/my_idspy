from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from ...nn.losses.classification import ClassificationLoss

from ...core.step import Step
from ...core.state import State
from ...nn.models.base import BaseModel
from ...nn.losses.base import BaseLoss
from ...nn.helpers import run_epoch

import logging
import numpy as np
from pathlib import Path
from src.idspy.core.pipeline import ObservablePipeline


"""
# Nel main
model = TabularClassifier(...)  # Tipo concreto
state.set("model", model, TabularClassifier)

# In TrainOneEpoch
return {"model": model}
# → Step cerca di fare: state.set("train.model", model, BaseModel)
# ❌ ERRORE! Già esiste con tipo TabularClassifier!

# SOLUZIONE: Non restituire model (è già nello state e viene modificato in-place), e settalo dove serve come Any
"""


class TrainOneEpoch(Step):
    """Train model for one epoch."""

    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_prefix: str = "train",
        clip_grad_max_norm: Optional[float] = 1.0,
        save_history: bool = False,
        save_outputs: bool = False,
        in_scope: str = "train",
        out_scope: str = "train",
        name: Optional[str] = None,
    ) -> None:
        self.writer: Optional[SummaryWriter] = (
            SummaryWriter(log_dir) if log_dir else None
        )
        self.log_prefix = log_prefix
        self.clip_grad_max_norm = clip_grad_max_norm
        self.save_history = save_history
        self.save_outputs = save_outputs

        super().__init__(
            name=name or "train_one_epoch",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    @Step.requires(
        dataloader=torch.utils.data.DataLoader,
        # model: BaseModel
        model=nn.Module,
        loss=BaseLoss,
        optimizer=torch.optim.Optimizer,
        device=torch.device,
        history=list,
        outputs=list,
        epoch=int,
    )
    @Step.provides(history=list, outputs=list, epoch=int)#model=BaseModel,
    def run(
        self,
        state: State,
        dataloader: torch.utils.data.DataLoader,
        #model: BaseModel
        model: nn.Module,
        loss: BaseLoss,
        optimizer: torch.optim.Optimizer,
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
        #l'accumulo tra le epoche, dato che resetto all'inizio di ogni training
        if self.save_outputs:
            outputs = []
        average_loss, outputs_list = run_epoch(
            desc="Training",
            log_prefix=self.log_prefix,
            is_training=True,
            dataloader=dataloader,
            model=model,
            device=device,
            loss_fn=loss,
            optimizer=optimizer,
            writer=self.writer,
            profiler=context,
            clip_grad_max_norm=self.clip_grad_max_norm,
            save_outputs=self.save_outputs,
            epoch=epoch,
        )

        if self.writer is not None:
            self.writer.close()
        if self.save_history:
            history.append(average_loss)
        if self.save_outputs:
            outputs.append(outputs_list)

        return {
            #"model": model,
            "history": history,
            "outputs": outputs,
            "epoch": epoch + 1,
        }



"""
Custom Step per Training con Early Stopping
============================================

Questo step sostituisce il ciclo for delle epoche, integrando:
- Training per un numero parametrico di epoche
- Early stopping basato sulla loss di validazione
- Salvataggio automatico del miglior modello
- Pulizia dello state tra le epoche
"""



logger = logging.getLogger(__name__)


class TrainWithEarlyStopping(Step):
    """
    Step personalizzato per il training con early stopping integrato.
    
    Gestisce:
    - Esecuzione di N epoche
    - Monitoraggio della loss di validazione
    - Early stopping con patience configurabile
    - Salvataggio del miglior modello
    - Pulizia dello state tra le epoche
    
    Parametri
    ---------
    epoch_pipeline : ObservablePipeline
        Pipeline da eseguire per ogni epoca (contiene TrainOneEpoch, ValidateOneEpoch, ecc.)
    num_epochs : int
        Numero massimo di epoche di training
    patience : int, default=3
        Numero di epoche consecutive senza miglioramento prima di fermarsi
    min_delta : float, default=0.001
        Miglioramento minimo considerato significativo (evita fluttuazioni casuali)
    checkpoint_dir : str o Path, optional
        Directory dove salvare i checkpoint del modello
    save_best_only : bool, default=True
        Se True, salva solo il modello con la migliore loss di validazione
    verbose : bool, default=True
        Se True, stampa informazioni dettagliate durante il training
    """
    
    def __init__(
        self,
        epoch_pipeline: ObservablePipeline,
        num_epochs: int,
        patience: int = 3,
        min_delta: float = 0.001,
        checkpoint_dir: Optional[str] = None,
        save_best_only: bool = True,
        verbose: bool = True,
        scheduler: Optional[object] = None,
        in_scope: str = "train",      
        out_scope: str = "train",     
        name: Optional[str] = None,   
    ) -> None:
        super().__init__(
        name=name or "train_with_early_stopping",  
        in_scope=in_scope,                          
        out_scope=out_scope,                        
        )
        self.epoch_pipeline = epoch_pipeline
        self.num_epochs = num_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.scheduler = scheduler
        # Variabili per l'early stopping
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_epoch = 0
        self.best_model_state = None
        
        # Crea la directory per i checkpoint se necessario
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_keys_to_clean(self):
        """
        Restituisce la lista delle chiavi dello state da pulire tra le epoche.
        Queste chiavi contengono risultati temporanei che si accumulerebbero
        causando problemi di memoria e inconsistenze.
        """
        return [
            # Output dei modelli (temporanei)
            "train.model", "test.model", "val.model",
            # Output delle predizioni/validazione (temporanei)
            "train.outputs", "test.outputs", "val.outputs",
            # Predizioni (temporanee)
            "train.predictions", "test.predictions", "val.predictions",
            # Loss e metriche (temporanee)
            "train.loss", "test.loss", "val.loss",
            "train.metrics", "test.metrics", "val.metrics",
            # History (si accumula)
            "train.history", "test.history", "val.history",
            # Epoch counter (si accumula)
            "train.epoch", "test.epoch", "val.epoch",
        ]
    
    def _clean_state(self, state: State):
        """
        Pulisce lo state rimuovendo le chiavi temporanee.
        Questo previene l'accumulo di dati tra le epoche, che causerebbe:
        - Crescita incontrollata della memoria
        - Inconsistenze nelle dimensioni dei dati
        - Confusione tra risultati di epoche diverse
        """
        keys_to_clean = self._get_keys_to_clean()
        
        for key in keys_to_clean:
            if state.has(key):
                if self.verbose:
                    logger.debug(f"🧹 Pulizia chiave: {key}")
                state.delete(key)
    
    def _save_checkpoint(self, state: State, epoch: int, is_best: bool = False):
        """
        Salva un checkpoint del modello.
        Parametri
        ---------
        state : State
            State contenente il modello da salvare
        epoch : int
            Numero dell'epoca corrente
        is_best : bool
            Se True, salva come "best_model.pt", altrimenti come "model_epoch_N.pt"
        """
        if not self.checkpoint_dir:
            return
        
        model = state.get("model", nn.Module)
        
        if is_best:
            checkpoint_path = self.checkpoint_dir / "best_model.pt"
            if self.verbose:
                logger.info(f"💾 Salvato MIGLIOR modello (epoca {epoch})")
        else:
            checkpoint_path = self.checkpoint_dir / f"model_epoch_{epoch}.pt"
            if self.verbose:
                logger.info(f"💾 Salvato checkpoint epoca {epoch}")
        
        torch.save(model.state_dict(), checkpoint_path)
    
    def _check_early_stopping(self, current_val_loss: float, epoch: int) -> bool:
        """
        Controlla se bisogna fermare il training (early stopping).
        Parametri
        ---------
        current_val_loss : float
            Loss di validazione dell'epoca corrente
        epoch : int
            Numero dell'epoca corrente
        Returns
        -------
        bool
            True se bisogna fermare il training, False altrimenti
        """
        # Verifica se c'è stato un miglioramento significativo
        if current_val_loss < (self.best_val_loss - self.min_delta):
            self.best_val_loss = current_val_loss
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            
            if self.verbose:
                logger.info(f"✅ Nuovo miglior loss: {current_val_loss:.6f}")
            
            return False  # Continua il training
        
        else:
            self.epochs_without_improvement += 1
            
            if self.verbose:
                logger.warning(
                    f"⚠️ Nessun miglioramento da {self.epochs_without_improvement} epoch(e). "
                    f"Best loss: {self.best_val_loss:.6f}"
                )
            
            # Ferma se abbiamo superato la patience
            if self.epochs_without_improvement >= self.patience:
                if self.verbose:
                    logger.info(
                        f"🛑 EARLY STOPPING! Nessun miglioramento per {self.patience} epoche consecutive.\n"
                        f"   Miglior loss: {self.best_val_loss:.6f} (epoca {self.best_epoch})"
                    )
                return True  
            
            return False  
    @Step.requires(model=nn.Module, optimizer=torch.optim.Optimizer, device=torch.device)
    @Step.provides( best_epoch=int, best_val_loss=float)#model=Any,
    def run(self, state: State, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> Optional[Dict[str, Any]]:
        """
        Esegue il training per N epoche con early stopping.
        Flusso di esecuzione
        --------------------
        Per ogni epoca:
        1. Esegue la epoch_pipeline (train + validation + metrics)
        2. Estrae la loss di validazione
        3. Controlla se c'è stato miglioramento (early stopping)
        4. Salva il modello se necessario
        5. Pulisce lo state per la prossima epoca
        """
        if self.verbose:
            logger.info(f"\n{'='*60}")
            logger.info(f"🚀 Inizio Training: {self.num_epochs} epoche max, patience={self.patience}")
            logger.info(f"{'='*60}\n")
        
        for epoch in range(1, self.num_epochs + 1):
            if self.verbose:
                print(f"\n{'─'*60}")
                print(f"📊 EPOCA {epoch}/{self.num_epochs}")
                print(f"{'─'*60}")
            
            # 1️⃣ Esegui l'epoca (train + validation + metrics)
            self.epoch_pipeline.run(state)
            
            # 2️⃣ Estrai la loss di validazione
            # (la loss viene salvata nello state da ValidateOneEpoch)
            if not state.has("test.loss"):
                logger.error("❌ ERRORE: 'test.loss' non trovato nello state!")
                break
            
            current_val_loss = float(state.get("test.loss", float))
            
            #aggiunto
            if self.scheduler is not None:
                old_lr = optimizer.param_groups[0]['lr']
                self.scheduler.step(current_val_loss)  # ← USA self.scheduler
                new_lr = optimizer.param_groups[0]['lr']
            
            # Log solo se il LR è cambiato
            if new_lr != old_lr and self.verbose:
                logger.info(f"📉 Learning rate ridotto: {old_lr:.6f} → {new_lr:.6f}")
            
            if self.verbose:
                logger.info(f"📉 Loss di validazione: {current_val_loss:.6f}")
            
            
            # possible_loss_keys = [
            #     "test.loss",      # Scoped con "test"
            #     # "loss",           # Senza scope
            #     "val.loss",       # Potrebbe essere "val" invece di "test"
            #     "avarage_loss",
            #     "loss_fn",
            #     "loss_weighted",
            # ]

            # current_val_loss = None
            # for key in possible_loss_keys:
            #     if state.has(key):
            #         current_val_loss = float(state.get(key, float))
            #         if self.verbose:
            #             logger.debug(f"✅ Loss trovata con chiave: '{key}'")
            #         break

            # if current_val_loss is None:
            #     logger.error(
            #         f"❌ ERRORE: Nessuna loss trovata nello state!\n"
            #         f"   Chiavi cercate: {possible_loss_keys}\n"
            #         f"   Chiavi disponibili: {[k for k in dir(state) if 'loss' in k.lower()]}"
            #     )
            #     break
            # 3️⃣ Controlla early stopping
            # Prima controlliamo se è il miglior modello finora
            is_best = current_val_loss < self.best_val_loss
            
            # Poi controlliamo se dobbiamo fermarci
            should_stop = self._check_early_stopping(current_val_loss, epoch)
            
            # 4️⃣ Salva il modello
            if is_best or not self.save_best_only:
                self._save_checkpoint(state, epoch, is_best=is_best)
                
                # Salva lo stato del modello in memoria (per ripristinarlo alla fine)
                if is_best:
                    self.best_model_state = {
                        k: v.cpu().clone() for k, v in model.state_dict().items()
                    }
            
            # 5️⃣ Se early stopping, ferma il training
            if should_stop:
                break
            
            # 6️⃣ Pulisci lo state per la prossima epoca
            self._clean_state(state)
        
        # 7️⃣ Ripristina il miglior modello
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            
            if self.verbose:
                logger.info(f"\n✅ Ripristinato miglior modello (epoca {self.best_epoch})")
        
        if self.verbose:
            logger.info(f"\n{'='*60}")
            logger.info(f"🎉 Training completato!")
            logger.info(f"   • Epoche eseguite: {epoch}")
            logger.info(f"   • Miglior epoca: {self.best_epoch}")
            logger.info(f"   • Miglior loss: {self.best_val_loss:.6f}")
            logger.info(f"{'='*60}\n")
        
        return {
            #"model": model,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
        }