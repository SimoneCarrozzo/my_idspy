import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
from typing import Dict, Any, Optional
import logging

from src.idspy.nn.models.classifier import TabularClassifier
from src.idspy.core.step import Step
from src.idspy.core.state import State
import pandas as pd
import numpy as np
from src.idspy.real_fed_learn.client import FederatedClient_C
from src.idspy.real_fed_learn.server import FederatedServer_C


logger = logging.getLogger(__name__)
""" VERSIONE DEL FED-TRAIN-ROUND DI TIPO C: OVVERO QUELLA DELLA TESI INIZIALE """

class FederatedTrainingRound_C(Step):
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
        # loss_fn=nn.Module,
        loss_fn_class=type,
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
        # loss_fn: nn.Module,
        loss_fn_class: type,
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
        
        # Aggiunto: 🆕 Recupera assegnazioni cluster dal round precedente
        client_to_cluster = state.get('federated.client_to_cluster', dict) if state.has('federated.client_to_cluster') else {}
        cluster_weights = state.get('federated.cluster_specific_weights', dict) if state.has('federated.cluster_specific_weights') else {}
        
        
        # ⬇️ AGGIUNGI QUESTI LOG PER DEBUG ⬇️
        logger.debug(f"   🔍 Round {self.round_num}: Trovati {len(client_to_cluster)} client→cluster mappings")
        logger.debug(f"   🔍 Round {self.round_num}: Trovati {len(cluster_weights)} cluster con pesi")
        if len(client_to_cluster) > 0:
            logger.debug(f"   📊 Mapping client→cluster: {len(client_to_cluster)} client assegnati")
            # Mostra distribuzione
            from collections import Counter
            cluster_dist = Counter(client_to_cluster.values())
            for cluster_id, count in sorted(cluster_dist.items()):
                logger.debug(f"      Cluster {cluster_id}: {count} client")
        else:
            logger.warning(f"   ⚠️ PRIMO ROUND: Nessun mapping client→cluster (verrà creato dal server)")
        
        #
        # ─────────────────────────────────────────────────────────────
        # 2️⃣ CLIENT TRAINING (per ogni client)
        # ─────────────────────────────────────────────────────────────
        
        client_updates = []
        
        for client_id in selected_clients:
            logger.info(f"\n🔧 Training client: {client_id}")
            
            
            # ⬇️ AGGIUNGI QUESTO LOG ⬇️
            logger.debug(f"   🔍 Debug: client_to_cluster ha {len(client_to_cluster)} elementi")
            logger.debug(f"   🔍 Debug: cluster_weights ha {len(cluster_weights)} elementi")
               
            
            # Prepara dati del client (SOLO train per ora)
            client_train_data = federated_splits[client_id]['train']
            
            # Crea step di training per questo client
            client_step = FederatedClient_C(
                client_id=client_id,
                local_epochs=self.local_epochs,
                batch_size=self.batch_size,
                device=device,
            )
            
            # ✅ NUOVO: Usa cluster dinamici (calcolati al round precedente)
            if state.has('federated.client_to_cluster'):
                # Round >0: usa mapping dinamico
                client_to_cluster = state.get('federated.client_to_cluster', dict)
                cluster_weights = state.get('federated.cluster_specific_weights', dict)
                
                cluster_id = client_to_cluster.get(client_id)
                
                if cluster_id is not None and cluster_id in cluster_weights:
                    cluster_model = copy.deepcopy(global_model)
                    cluster_model.load_state_dict(cluster_weights[cluster_id])
                    logger.info(f"   🎯 {client_id} usa Cluster {cluster_id} model (similarity-based)")
                    model_to_use = cluster_model
                else:
                    logger.warning(f"   ⚠️ {client_id}: cluster non trovato, uso global model")
                    model_to_use = global_model
            else:
                # Round 0: tutti usano global model
                model_to_use = global_model
                logger.info(f"   🌐 {client_id} usa Global model (primo round)")
            
            # Crea state temporaneo per questo client
            client_state = State({
                'federated.client_data': client_train_data,
                # 'federated.global_model': global_model,
                'federated.global_model': model_to_use,  # ← Usa cluster model
                # 'federated.loss_fn': loss_fn,
                'federated.loss_fn_class': state.get('federated.loss_fn_class', type),  # ✅ NUOVO
                'federated.optimizer_class': optimizer_class,
                'federated.optimizer_kwargs': optimizer_kwargs,
                # 'federated.host_specialization': state.get('federated.host_specialization', dict), # ⬅️ NUOVO
                'federated.target_attack': state.get('federated.target_attack', str),
                #nuovo
                'federated.scheduler_config': state.get('federated.scheduler_config', dict) if state.has('federated.scheduler_config') else None,
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
        
        server_step = FederatedServer_C(aggregation_method="fedavg")
        
        state.set('federated.client_updates', client_updates, list)
        state.set('federated.global_model', global_model, TabularClassifier)      
        
        # server_step.run(server_state)
        server_step.run(state)        
        
        # aggregated_weights = server_state.get('federated.aggregated_weights', dict)
        # aggregation_metrics = server_state.get('federated.aggregation_metrics', dict)
        aggregated_weights = state.get('federated.aggregated_weights', dict)
        aggregation_metrics = state.get('federated.aggregation_metrics', dict)
        
        logger.info(f"✅ Server: Aggregazione Completata!")
        # ─────────────────────────────────────────────────────────────
        # 4️⃣ UPDATE GLOBAL MODEL
        # ─────────────────────────────────────────────────────────────
        
        actual_model_device = next(global_model.parameters()).device

        logger.info(f"💾 Spostamento pesi su {actual_model_device} e aggiornamento modello...")

        try:
            # Creiamo il dizionario spostando ogni tensore solo se necessario
            clean_state_dict = {
                k: v.to(actual_model_device) for k, v in aggregated_weights.items()
            }
            
            # Carichiamo i pesi
            global_model.load_state_dict(clean_state_dict)
            logger.info(f"✅ Modello globale aggiornato con successo su {actual_model_device}")

        except Exception as e:
            logger.error(f"❌ Errore durante il caricamento dei pesi: {str(e)}")
            # Fallback estremo: prova a caricare senza spostare nulla
            global_model.load_state_dict(aggregated_weights, strict=False)
            logger.warning("⚠️ Caricamento pesi completato con modalità Fallback (strict=False)")
        logger.info(f"✅ Modello globale aggiornato")
        
        # ─────────────────────────────────────────────────────────────
        # 5️⃣ RISULTATI ROUND
        # ─────────────────────────────────────────────────────────────
        # 📊 Calcola metriche aggregate statistiche
        client_results = valid_updates  # Usa solo i client validi
        
        # 📊 ESTRAZIONE DATI CORRETTA (Basata sul FederatedClient)
        # La struttura è: update -> 'metrics' -> 'final_loss'
        client_losses = [r['metrics']['final_loss'] for r in client_results]
        client_samples = [r['num_samples'] for r in client_results]

        # Calcolo statistiche con controllo di sicurezza (se lista vuota)
        if len(client_losses) > 0:
            avg_loss = np.mean(client_losses)
            weighted_avg = np.average(client_losses, weights=client_samples)
            min_loss = np.min(client_losses)
            max_loss = np.max(client_losses)
            std_loss = np.std(client_losses)
        else:
            avg_loss = weighted_avg = min_loss = max_loss = std_loss = 0.0
        
        # ✅ Calcola metriche di stabilità (già hai client_updates qui) --> nuovo
        client_accs = [r['metrics']['final_acc'] for r in client_results if 'final_acc' in r['metrics']]
        cluster_mapping = state.get('federated.client_to_cluster', dict)
        
        # Creiamo il dizionario delle metriche dettagliate
        round_metrics = {
            'round_num': self.round_num,
            'num_clients_participated': len(selected_clients),
            'total_samples': sum(client_samples),
            'avg_loss': avg_loss,                         # Media aritmetica
            'weighted_avg_loss': weighted_avg, # Media pesata
            'min_loss': min_loss,                          # Miglior client
            'max_loss': max_loss,                          # Peggior client
            'std_loss': std_loss,                          # Varianza tra i client (importante per Non-IID)
            
            'acc_variance': np.var(client_accs) if client_accs else 0.0,
            'num_clusters': len(set(cluster_mapping.values())) if cluster_mapping else 1,
            'avg_cluster_size': len(client_results) / max(1, len(set(cluster_mapping.values()))) if cluster_mapping else len(client_results)
        }

        logger.info(f"\n📊 Metriche Round {self.round_num}:")
        #nuovo
        logger.info(f"  • Acc Variance:   {round_metrics['acc_variance']:.4f}")
        logger.info(f"  • Num Clusters:   {round_metrics['num_clusters']}")
        ########
        logger.info(f"   • Loss Media (Aritm): {round_metrics['avg_loss']:.4f}")
        logger.info(f"   • Loss Pesata (Reale): {round_metrics['weighted_avg_loss']:.4f}")
        logger.info(f"   • Loss Min/Max:       {round_metrics['min_loss']:.4f} / {round_metrics['max_loss']:.4f}")
        logger.info(f"   • Deviazione Std:     {round_metrics['std_loss']:.4f}")
        # ─────────────────────────────────────────────────────────────
        
        # Costruiamo l'oggetto finale
        final_results = {
            'round_num': self.round_num,
            'num_clients_participated': len(client_results),
            'client_updates': client_results,       
            'aggregation_metrics': aggregation_metrics,
            'round_metrics': round_metrics,
            'clustering_metrics': state.get('federated.clustering_metrics', dict)  # ← AGGIUNGI

        }
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ ROUND {self.round_num} COMPLETATO")
        logger.info(f"{'='*70}")
        
        return {'round_results': final_results}
 