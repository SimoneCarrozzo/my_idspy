import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
from typing import Dict, Any, Optional
import logging

from src.idspy.core.step import Step
from src.idspy.core.state import State
import pandas as pd

from src.idspy.federated_learning.client import FederatedClient
from src.idspy.federated_learning.server import FederatedServer


logger = logging.getLogger(__name__)


class FederatedTrainingRound(Step):
    """
    Esegue un singolo round di Federated Learning.
    
    Flusso:
    1. Seleziona client (opzionale)
    2. Ogni client addestra localmente (parallelo)
    3. Server aggrega i pesi
    4. Valuta su test set globale
    
    Analogia: È come una "riunione mensile" dove ogni ospedale 
              condivide ciò che ha imparato.
    """
    
    def __init__(
        self,
        round_num: int,
        local_epochs: int = 5,
        batch_size: int = 1024,
        client_fraction: float = 1.0,  # Frazione di client da usare
        in_scope: str = "federated",
        out_scope: str = "federated",
        name: Optional[str] = None,
    ):
        self.round_num = round_num
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.client_fraction = client_fraction
        
        super().__init__(
            name=name or f"federated_round_{round_num}",
            in_scope=in_scope,
            out_scope=out_scope,
        )
    
    @Step.requires(
        federated_splits=dict,  # {client_id: {'train': df, 'val': df, 'test': df}}
        global_model=nn.Module,
        loss_fn=nn.Module,
        optimizer_class=type,
        optimizer_kwargs=dict,
        device=torch.device,
    )
    @Step.provides(
        round_results=dict,
    )
    def run(
        self,
        state: State,
        federated_splits: dict,
        global_model: nn.Module,
        loss_fn: nn.Module,
        optimizer_class: type,
        optimizer_kwargs: dict,
        device: torch.device,
    ) -> Optional[Dict[str, Any]]:
        """
        Esegue un round completo di FL.
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"🔄 ROUND {self.round_num}")
        logger.info(f"{'='*70}")
        
        # ─────────────────────────────────────────────────────────────
        # 1️⃣ SELEZIONE CLIENT (opzionale)
        # ─────────────────────────────────────────────────────────────
        
        all_client_ids = list(federated_splits.keys())
        num_selected = max(1, int(len(all_client_ids) * self.client_fraction))
        
        if self.client_fraction < 1.0:
            import random
            selected_clients = random.sample(all_client_ids, num_selected)
            logger.info(f"📋 Client selezionati: {num_selected}/{len(all_client_ids)}")
        else:
            selected_clients = all_client_ids
            logger.info(f"📋 Tutti i {len(all_client_ids)} client partecipano")
        
        # ─────────────────────────────────────────────────────────────
        # 2️⃣ CLIENT TRAINING (per ogni client)
        # ─────────────────────────────────────────────────────────────
        
        client_updates = []
        
        for client_id in selected_clients:
            logger.info(f"\n🔧 Training client: {client_id}")
            
            # Prepara dati del client (SOLO train per ora)
            client_train_data = federated_splits[client_id]['train']
            
            # Crea step di training per questo client
            client_step = FederatedClient(
                client_id=client_id,
                local_epochs=self.local_epochs,
                batch_size=self.batch_size,
                device=device,
            )
            
            # Crea state temporaneo per questo client
            client_state = State({
                'federated.client_data': client_train_data,
                'federated.global_model': global_model,
                'federated.loss_fn': loss_fn,
                'federated.optimizer_class': optimizer_class,
                'federated.optimizer_kwargs': optimizer_kwargs,
            })
            
            # Esegui training client
            client_step.run(client_state)
            
            # Raccogli risultati
            local_weights = client_state.get('federated.local_weights', dict)
            local_metrics = client_state.get('federated.local_metrics', dict)
            num_samples = client_state.get('federated.num_samples', int)
            
            client_updates.append({
                'client_id': client_id,
                'weights': local_weights,
                'num_samples': num_samples,
                'metrics': local_metrics,
            })
        # ─────────────────────────────────────────────────────────────
        # ⭐ FILTRAGGIO AGGIORNAMENTI VALIDI
        # ─────────────────────────────────────────────────────────────
        
        valid_updates = []
        for update in client_updates:
            # Verifichiamo che i pesi esistano e che ci siano campioni addestrati
            if update['weights'] is None or update['num_samples'] == 0:
                logger.warning(f"   skip {update['client_id']}: nessun peso valido o zero campioni.")
                continue
            valid_updates.append(update)

        if len(valid_updates) == 0:
            logger.error("❌ NESSUN CLIENT ha prodotto aggiornamenti validi in questo round!")
            # Invece di un break (che funziona solo nei cicli), 
            # restituiamo None o solleviamo un'eccezione a seconda della logica della pipeline
            return None 

        logger.info(f"⚖️ Partecipano all'aggregazione: {len(valid_updates)}/{len(client_updates)} client")
        # ─────────────────────────────────────────────────────────────
        # 3️⃣ SERVER AGGREGATION
        # ─────────────────────────────────────────────────────────────
        
        logger.info(f"\n🏛️ Aggregazione pesi sul server...")
        
        server_step = FederatedServer(aggregation_method="fedavg")
        
        server_state = State({
            'federated.client_updates': client_updates,
            'federated.global_model': global_model,
        })
        
        server_step.run(server_state)
        
        aggregated_weights = server_state.get('federated.aggregated_weights', dict)
        aggregation_metrics = server_state.get('federated.aggregation_metrics', dict)
        
        # ─────────────────────────────────────────────────────────────
        # 4️⃣ UPDATE GLOBAL MODEL
        # ─────────────────────────────────────────────────────────────
        
        global_model.load_state_dict(aggregated_weights)
        logger.info(f"✅ Modello globale aggiornato")
        
        # ─────────────────────────────────────────────────────────────
        # 5️⃣ RISULTATI ROUND
        # ─────────────────────────────────────────────────────────────
        
        round_results = {
            'round_num': self.round_num,
            'num_clients_participated': len(selected_clients),
            'client_updates': client_updates,
            'aggregation_metrics': aggregation_metrics,
        }
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ ROUND {self.round_num} COMPLETATO")
        logger.info(f"{'='*70}")
        
        return {'round_results': round_results}