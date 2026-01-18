import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
from typing import Dict, Any, Optional
import logging

from src.idspy.core.step import Step
from src.idspy.core.state import State
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class FederatedServer_C(Step):
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
        
        """ if self.aggregation_method == "fedavg":
            aggregated_weights = self._fedavg(
                client_weights_list,
                client_samples_list,
                total_samples
            )
        else:
            raise NotImplementedError(
                f"Metodo {self.aggregation_method} non implementato"
            ) 
            ############    versione vecchia """
        # AGGREGAZIONE: MEDIA PESATA (FedAvg) CON IMPORTANZA ATTACCHI
         ############ NUOVA VERSIONE ############
        if self.aggregation_method == "fedavg":
            aggregated_weights = self._fedavg(
                client_updates=client_updates,  # ⬅️ Passa la lista completa
                state=state                     # ⬅️ Passa lo state
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
        client_updates: list,
        state: State
    ) -> dict:
        """
        FedAvg con Clustering: aggrega separatamente client simili.
        """        
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        logger.info("   🔢 Calcolo Weighted FedAvg con Clustering...")
        
        # client_attack_info = state.get('federated.client_attack_info', dict)
        
        # ════════════════════════════════════════════════════════════════
        # 🆕 FASE 1: ESTRAZIONE PESI ULTIMO LAYER
        # ════════════════════════════════════════════════════════════════
        last_layer_weights = []
        client_ids = []
        
        for update in client_updates:
            weights_dict = update['weights']
            
            # Cerca classifier_head (ultimo layer)
            last_layer_key = 'classifier_head.weight'  # Nome tipico PyTorch
            
            if last_layer_key in weights_dict:
                w = weights_dict[last_layer_key].cpu().numpy().flatten()
                last_layer_weights.append(w)
                client_ids.append(update['client_id'])
            else:
                logger.warning(f"⚠️ Ultimo layer non trovato per {update.get('client_id')}")
        
        last_layer_weights = np.array(last_layer_weights)
        
        # ════════════════════════════════════════════════════════════════════════
        # 🆕 FASE 2: CLUSTERING UNSUPERVISED (Basato su Similarità Pesi)
        # ════════════════════════════════════════════════════════════════════════

        logger.info("   🔍 Calcolo clustering unsupervised (cosine similarity)...")

        # Step 1: Estrai pesi ultimo layer (già fatto in FASE 1)
        # last_layer_weights = array di shape (num_clients, num_features_ultimo_layer)

        if len(last_layer_weights) < 2:
            logger.warning("⚠️ Meno di 2 client, skip clustering")
            # Fallback: aggrega tutti insieme
            cluster_labels = np.zeros(len(client_ids), dtype=int)
            num_clusters = 1
        else:
            # Step 2: Calcola matrice di similarità (cosine)
            similarity_matrix = cosine_similarity(last_layer_weights)
            
            logger.debug(f"   📊 Similarity matrix shape: {similarity_matrix.shape}")
            logger.debug(f"   📊 Similarity range: [{similarity_matrix.min():.3f}, {similarity_matrix.max():.3f}]")
            
            # Step 3: Clustering basato su threshold
            # Approccio: raggruppa client con similarità > THRESHOLD
            
            SIMILARITY_THRESHOLD = 0.80  # ← Parametro critico da tuning
            
            # Inizializza: ogni client è nel suo cluster
            cluster_labels = np.arange(len(client_ids))
            cluster_map = {i: i for i in range(len(client_ids))}
            
            # Unisci client simili
            for i in range(len(client_ids)):
                for j in range(i+1, len(client_ids)):
                    sim = similarity_matrix[i, j]
                    
                    if sim >= SIMILARITY_THRESHOLD:
                        # Unisci cluster i e j
                        old_cluster_j = cluster_map[j]
                        new_cluster = cluster_map[i]
                        
                        # Aggiorna tutti i membri del cluster j
                        for k, v in cluster_map.items():
                            if v == old_cluster_j:
                                cluster_map[k] = new_cluster
                        
                        logger.debug(f"      🔗 Merge: {client_ids[i]} ↔ {client_ids[j]} (sim={sim:.3f})")
            
            # Converti in array
            cluster_labels = np.array([cluster_map[i] for i in range(len(client_ids))])
            
            # Rinumera cluster da 0 a N-1
            unique_clusters = sorted(set(cluster_labels))
            cluster_remap = {old: new for new, old in enumerate(unique_clusters)}
            cluster_labels = np.array([cluster_remap[c] for c in cluster_labels])
            
            num_clusters = len(unique_clusters)

        # Log distribuzione cluster
        logger.info(f"   🎯 Clustering unsupervised: {num_clusters} cluster identificati")
        for cluster_id in range(num_clusters):
            members = [client_ids[i] for i in range(len(cluster_labels)) if cluster_labels[i] == cluster_id]
            logger.info(f"      Cluster {cluster_id}: {len(members)} client → {members}")
            
            """ ✅ NUOVO: Log similarità intra-cluster
            if len(members) > 1:
                member_indices = [i for i in range(len(cluster_labels)) if cluster_labels[i] == cluster_id]
                intra_sims = []
                for i in member_indices:
                    for j in member_indices:
                        if i < j:
                            intra_sims.append(similarity_matrix[i, j])
                avg_intra_sim = np.mean(intra_sims) if intra_sims else 0
                logger.info(f"         Avg intra-cluster similarity: {avg_intra_sim:.3f}") """
            # Calcola media intra/inter cluster
            intra_sims = []
            inter_sims = []

            for cluster_id in range(num_clusters):
                # Intra-cluster similarity
                cluster_members = [i for i in range(len(cluster_labels)) if cluster_labels[i] == cluster_id]
                if len(cluster_members) > 1:
                    for i in cluster_members:
                        for j in cluster_members:
                            if i < j:
                                intra_sims.append(similarity_matrix[i, j])

            # Inter-cluster similarity
            for i in range(len(cluster_labels)):
                for j in range(i+1, len(cluster_labels)):
                    if cluster_labels[i] != cluster_labels[j]:
                        inter_sims.append(similarity_matrix[i, j])

        # ✅ Salva metriche di clustering
        state.set('federated.clustering_metrics', {
            'num_clusters': num_clusters,
            'similarity_threshold': SIMILARITY_THRESHOLD,
            'avg_intra_similarity': np.mean(intra_sims) if intra_sims else 0.0, #aggiunto
            'avg_inter_similarity': np.mean(inter_sims) if inter_sims else 0.0, #aggiunto
            'similarity_matrix': similarity_matrix.tolist(),
            'cluster_labels': cluster_labels.tolist(),
        }, dict)
        
        
        # ════════════════════════════════════════════════════════════════
        # 🆕 FASE 3: AGGREGAZIONE PER CLUSTER
        # ════════════════════════════════════════════════════════════════
        cluster_aggregations = {}
        
        for cluster_id in range(num_clusters):
            # Filtra client del cluster
            cluster_updates = [
                client_updates[i] for i in range(len(client_updates)) 
                if cluster_labels[i] == cluster_id
            ]
            
            if len(cluster_updates) == 0:
                continue
            
            # Calcola pesi per questo cluster (attack-weighted)
            # aggregation_weights = []
            # for update in cluster_updates:
            #     client_id = update.get('client_id')
            #     num_samples = update['num_samples']
            #     # info = client_attack_info.get(client_id, {})
            #     attack_ratio = info.get('attack_ratio', 0.1)
            #     weight = np.sqrt(attack_ratio) * num_samples
            #     aggregation_weights.append(weight)
            
            # total_weight = sum(aggregation_weights)
            # if total_weight > 0:
            #     aggregation_weights = [w / total_weight for w in aggregation_weights]
            # else:
            #     aggregation_weights = [1.0 / len(cluster_updates)] * len(cluster_updates)
            
            # ✅ NUOVO: Pesi basati SOLO su num_samples (FedAvg standard)
            cluster_samples = [u['num_samples'] for u in cluster_updates]
            total_samples = sum(cluster_samples)

            if total_samples > 0:
                aggregation_weights = [s / total_samples for s in cluster_samples]
            else:
                aggregation_weights = [1.0 / len(cluster_updates)] * len(cluster_updates)
            
            # Aggrega pesi del cluster
            cluster_weights = {}
            first_weights = cluster_updates[0]['weights']
            
            for key in first_weights.keys():
                weighted_sum = sum(
                    aggregation_weights[i] * cluster_updates[i]['weights'][key]
                    for i in range(len(cluster_updates))
                )
                cluster_weights[key] = weighted_sum
            
            cluster_aggregations[cluster_id] = {
                'weights': cluster_weights,
                'client_ids': [u['client_id'] for u in cluster_updates],
                'num_clients': len(cluster_updates)
            }
            
            logger.info(f"   ✅ Cluster {cluster_id}: Aggregati {len(cluster_updates)} client")
        
        # ════════════════════════════════════════════════════════════════
        # FASE 4: AGGREGAZIONE IBRIDA
        # - Globale: media semplice tra cluster (per backward compatibility)
        # - Cluster-specific: salva pesi per ogni cluster (usati nel prossimo round)
        # ════════════════════════════════════════════════════════════════
        global_weights = {}
        first_cluster = list(cluster_aggregations.values())[0]['weights']
        
        for key in first_cluster.keys():
            cluster_values = [c['weights'][key] for c in cluster_aggregations.values()]
            global_weights[key] = sum(cluster_values) / len(cluster_values)
        
        # 🆕 SALVA PESI PER CLUSTER (non solo metadata)
        state.set('federated.cluster_specific_weights', {
            cluster_id: agg['weights']  # ← Pesi effettivi
            for cluster_id, agg in cluster_aggregations.items()
        }, dict)

        state.set('federated.client_to_cluster', 
                dict(zip(client_ids, cluster_labels.tolist())), dict)
        
        # ⬇️ AGGIUNGI QUESTI LOG PER DEBUG ⬇️
        logger.debug(f"   💾 Salvati pesi per {len(cluster_aggregations)} cluster")
        logger.debug(f"   💾 Mapping client→cluster: {dict(zip(client_ids, cluster_labels.tolist()))}")
        
        return global_weights  
 