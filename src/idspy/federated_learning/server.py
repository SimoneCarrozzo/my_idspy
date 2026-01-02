import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
from typing import Dict, Any, Optional
import logging

from src.idspy.core.step import Step
from src.idspy.core.state import State
import pandas as pd

logger = logging.getLogger(__name__)

class FederatedServer(Step):
    """
    Server centrale che aggrega i pesi dei client.
    
    Responsabilità:
    - Ricevere pesi da tutti i client
    - Calcolare media pesata (FedAvg)
    - Aggiornare modello globale
    
    Analogia: È il coordinatore che combina le conoscenze di tutti gli ospedali.
    """
    
    def __init__(
        self,
        aggregation_method: str = "fedavg",  # "fedavg" o "fedprox" (futuro)
        in_scope: str = "federated",
        out_scope: str = "federated",
        name: Optional[str] = None,
    ):
        self.aggregation_method = aggregation_method
        
        super().__init__(
            name=name or "federated_server",
            in_scope=in_scope,
            out_scope=out_scope,
        )
    
    @Step.requires(
        client_updates=list,  # Lista di dict con {weights, num_samples}
        global_model=nn.Module,
    )
    @Step.provides(
        aggregated_weights=dict,
        aggregation_metrics=dict,
    )
    def run(
        self,
        state: State,
        client_updates: list,  # [{'weights': {...}, 'num_samples': 1000}, ...]
        global_model: nn.Module,
    ) -> Optional[Dict[str, Any]]:
        """
        Aggrega i pesi dei client usando FedAvg.
        """
        logger.info(f"🏛️ Server: Aggregando pesi da {len(client_updates)} client...")
        
        if len(client_updates) == 0:
            raise ValueError("Nessun client update ricevuto!")
        
        # ─────────────────────────────────────────────────────────────
        # 1️⃣ ESTRAI PESI E SAMPLES DA OGNI CLIENT
        # ─────────────────────────────────────────────────────────────
        
        client_weights_list = []
        client_samples_list = []
        
        for update in client_updates:
            client_weights_list.append(update['weights'])
            client_samples_list.append(update['num_samples'])
        
        total_samples = sum(client_samples_list)
        
        logger.info(f"   📊 Totale samples: {total_samples:,}")
        logger.info(f"   📊 Samples per client: {client_samples_list}")
        
        # ─────────────────────────────────────────────────────────────
        # 2️⃣ AGGREGAZIONE: MEDIA PESATA (FedAvg)
        # ─────────────────────────────────────────────────────────────
        
        if self.aggregation_method == "fedavg":
            aggregated_weights = self._fedavg(
                client_weights_list,
                client_samples_list,
                total_samples
            )
        else:
            raise NotImplementedError(
                f"Metodo {self.aggregation_method} non implementato"
            )
        
        # ─────────────────────────────────────────────────────────────
        # 3️⃣ METRICHE AGGREGAZIONE
        # ─────────────────────────────────────────────────────────────
        
        aggregation_metrics = {
            'num_clients': len(client_updates),
            'total_samples': total_samples,
            'aggregation_method': self.aggregation_method,
            'client_samples_distribution': {
                f'client_{i}': samples
                for i, samples in enumerate(client_samples_list)
            }
        }
        
        logger.info(f"✅ Server: Aggregazione completata")
        
        return {
            'aggregated_weights': aggregated_weights,
            'aggregation_metrics': aggregation_metrics,
        }
    
    def _fedavg(
        self,
        client_weights_list: list,
        client_samples_list: list,
        total_samples: int
    ) -> dict:
        """
        Implementa FedAvg: Media Pesata dei pesi.
        
        Formula:
        w_global = Σ(w_client_i * n_i) / Σ(n_i)
        
        Dove:
        - w_client_i = pesi del client i
        - n_i = numero di sample del client i
        """
        logger.info("   🔢 Calcolo FedAvg (media pesata)...")
        
        # Inizializza dizionario per pesi aggregati
        aggregated_weights = {}
        
        # Per ogni layer del modello
        for key in client_weights_list[0].keys():
            # Calcola media pesata per questo layer
            layer_sum = None
            
            for client_weights, num_samples in zip(client_weights_list, client_samples_list):
                weight = client_weights[key]
                
                # Peso ponderato per questo client
                weighted_contribution = weight * (num_samples / total_samples)
                
                if layer_sum is None:
                    layer_sum = weighted_contribution
                else:
                    layer_sum += weighted_contribution
            
            aggregated_weights[key] = layer_sum
        
        logger.info(f"   ✅ FedAvg completato per {len(aggregated_weights)} layers")
        
        return aggregated_weights