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

""" VERSIONE DEL FED-SERVER DI TIPO A: OVVERO COLLABORA CON LA VERSIONE DEL FED-TRAIN-ROUND DI TIPO A E CON IL FED-CLIENT DI TIPO A. 
IN QUESTA TIPOLOGIA, I PESI DEI CLIENT CARATTERIZZATI DAL VEDERE LO STESSO ATTACCO SONO RAGGRUPPATI IN CLUSTER, MA A CAUSA DI ALCUNE 
FEATURE IN COMUNE TRA GLI ATTACCHI, SI VERIFICA UN OVERLAP SEMANTICO CHE PORTA A DEI CLUSTER DIVERSI DA QUELLI EFFETTIVI. 
DOVREMMO AVERE 4 CLUSTER, DI CUI UNO FORMATO DA 5 HOST CHE VEDONO DOSS-HOIC COME TOP-ATTACK, 
UN CLUSTER FORMATO DA 2 HOST CHE VEDONO DOSS-HULK, UN CLUSTER FORMATO DA UN CLIENT CHE VEDE BOT, 
E L'ULTIMO CLUSTER FORMATO DA L'ULTIMO HOST CHE VEDE INFILTRATION. ALL'ATTO PRATICO ABBIAMO 4 CLUSTER FORMATI DA 6-1-1-1"""

class FederatedServer_alt(Step):
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
        if self.aggregation_method == "fedAVG":
            aggregated_weights = self._fedAVG(
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
        
        
    def _fedAVG(
        self,
        client_updates: list,
        state: State
    ) -> dict:
        """
        FedAvg con Clustering: aggrega separatamente client simili.
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics.pairwise import cosine_similarity
        
        logger.info("   🔢 Calcolo Weighted FedAvg con Clustering...")
        
        client_attack_info = state.get('federated.client_attack_info', dict)
        
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
        
        # ════════════════════════════════════════════════════════════════
        # 🆕 FASE 2: CLUSTERING (K-Means)
        # ════════════════════════════════════════════════════════════════
        """ KMeans.fit_predict() calcola automaticamente la distanza euclidea tra i vettori last_layer_weights
            Algoritmo K-Means: raggruppa i punti che hanno distanza euclidea minima dal centroide del cluster
            NON serve calcolare manualmente cosine_similarity() o euclidean_distance()
            
            MOMNTANEMENTE CAMBIATO CON QUANTO SEGUE 
        num_clusters = min(4, len(client_updates))  # Max 4 cluster (DDoS, DoS, Bot, Infiltration)
        
        if len(client_updates) < 2:
            logger.warning("⚠️ Meno di 2 client, skip clustering")
            cluster_labels = np.zeros(len(client_updates))
        else:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10) 
            cluster_labels = kmeans.fit_predict(last_layer_weights)
            
            logger.info(f"   🎯 Clustering: {num_clusters} cluster identificati")
            for i in range(num_clusters):
                members = [client_ids[j] for j in range(len(cluster_labels)) if cluster_labels[j] == i]
                logger.info(f"      Cluster {i}: {len(members)} client → {members[:3]}...")"""
        # ════════════════════════════════════════════════════════════════
        # 🆕 FASE 2: CLUSTERING STABILE (Una sola volta al Round 0)
        # ════════════════════════════════════════════════════════════════
        num_clusters = min(4, len(client_updates))

        # ✅ CLUSTERING STABILE: Usa KMeans solo al Round 0
        if not state.has('federated.stable_kmeans_model'):
            # PRIMO ROUND: Calcola cluster e salva il modello
            if len(client_updates) < 2:
                cluster_labels = np.zeros(len(client_updates), dtype=int)
                kmeans = None
            else:
                kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(last_layer_weights)
            
            # Salva il modello KMeans per i round successivi
            state.set('federated.stable_kmeans_model', kmeans, object)
            logger.debug(f" 🆕🆕🆕 Primo clustering: {num_clusters} cluster fissi creati")
        else:
            # ✅ ROUND SUCCESSIVI: Predici usando il modello salvato
            kmeans = state.get('federated.stable_kmeans_model', object)
            if kmeans is not None:
                cluster_labels = kmeans.predict(last_layer_weights)
                logger.debug(f" 🔄🔄🔄🔄 Riutilizzo cluster esistenti (assignment stabile)")
            else:
                cluster_labels = np.zeros(len(client_updates), dtype=int)

        logger.debug(f"  🎯 🎯 Clustering: {num_clusters} cluster identificati")
        for i in range(num_clusters):
            members = [client_ids[j] for j in range(len(cluster_labels)) if cluster_labels[j] == i]
            logger.debug(f"      Cluster {i}: {len(members)} client → {members[:3]}...")
        
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
            aggregation_weights = []
            for update in cluster_updates:
                client_id = update.get('client_id')
                num_samples = update['num_samples']
                info = client_attack_info.get(client_id, {})
                attack_ratio = info.get('attack_ratio', 0.1)
                weight = np.sqrt(attack_ratio) * num_samples
                aggregation_weights.append(weight)
            
            total_weight = sum(aggregation_weights)
            if total_weight > 0:
                aggregation_weights = [w / total_weight for w in aggregation_weights]
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
        """ # ════════════════════════════════════════════════════════════════
        # FASE 4: RITORNA AGGREGAZIONE GLOBALE (media dei cluster)
        # ════════════════════════════════════════════════════════════════
        # Per ora: media semplice tra i cluster (puoi pesare dopo)
        global_weights = {}
        first_cluster = list(cluster_aggregations.values())[0]['weights']
        
        for key in first_cluster.keys():
            cluster_values = [c['weights'][key] for c in cluster_aggregations.values()]
            global_weights[key] = sum(cluster_values) / len(cluster_values)
        
        # 🆕 SALVA INFO CLUSTERING NELLO STATE
        state.set('federated.cluster_info', {
            'cluster_labels': cluster_labels.tolist(),
            'client_to_cluster': dict(zip(client_ids, cluster_labels.tolist())),
            'cluster_aggregations': {
                k: {'client_ids': v['client_ids'], 'num_clients': v['num_clients']}
                for k, v in cluster_aggregations.items()
            }
        }, dict)
        
        return global_weights """
    
    def _fedavg_preCluster(
        self,
        client_updates: list,
        # client_attack_info: dict,
        state: State
    ) -> dict:
        """
        Implementa FedAvg pesata sulla proporzione di attacchi.
        """
        logger.info("   📢 Calcolo Weighted FedAvg (Attack-Weighted)...")
    
        # 1️⃣ Recupera le info sugli Attacks dal state
        client_attack_info = state.get('federated.client_attack_info', dict)
        
        # 1. Calcolo dei pesi di aggregazione per ogni client
        # L'obiettivo è dare più peso a chi ha più attacchi per "istruire" meglio il modello globale
        aggregation_weights = []
        for update in client_updates:
            client_id = update.get('client_id')
            num_samples = update['num_samples']
            
            # Recupera l'attack ratio (default 0.1 se non trovato per evitare divisioni per zero)
            # Se hai salvato le info nel main come 'client_attack_info[host_id]', usale qui
            info = client_attack_info.get(client_id, {})
            attack_ratio = info.get('attack_ratio', 0.1) 
            
            # FORMULA IBRIDA: Peso = sqrt(attack_ratio) * num_samples
            # Usiamo la radice quadrata per non penalizzare TROPPO chi ha pochi attacchi
            weight = np.sqrt(attack_ratio) * num_samples
            aggregation_weights.append(weight)

        
        """ # Normalizzazione pesi (la somma deve fare 1.0)
            # total_importance = sum(aggregation_weights)
            norm_weights = [w / total_importance for w in aggregation_weights]
        logger.info(f"   ⚖️ Pesi di aggregazione calcolati: {[f'{w:.3f}' for w in norm_weights]}") """
        # 3️⃣ Normalizza i pesi (somma = 1.0)
        total_weight = sum(aggregation_weights)
        
        if total_weight == 0:
            logger.warning("⚠️ Peso totale = 0! Uso FedAvg standard.")
            # Fallback: media semplice basata su num_samples
            total_samples = sum(u['num_samples'] for u in client_updates)
            aggregation_weights = [u['num_samples'] / total_samples for u in client_updates]
        else:
            aggregation_weights = [w / total_weight for w in aggregation_weights]
        
        logger.info(f"   ⚖️ Pesi aggregazione: {[f'{w:.3f}' for w in aggregation_weights]}")

        # 4️⃣ Aggregazione effettiva (media pesata dei pesi)
        aggregated_weights = {}
        first_client_weights = client_updates[0]['weights']
        
        """ for key in first_client_weights.keys():
            layer_sum = None
            
            for i, update in enumerate(client_updates):
                client_weight = update['weights'][key]
                # Applichiamo il peso di importanza normalizzato
                contribution = client_weight * norm_weights[i]
                
                if layer_sum is None:
                    layer_sum = contribution
                else:
                    layer_sum += contribution
            
            aggregated_weights[key] = layer_sum
            
        return aggregated_weights """
        for key in first_client_weights.keys():
            # Somma pesata: w₁*θ₁ + w₂*θ₂ + ...
            weighted_sum = sum(
                aggregation_weights[i] * client_updates[i]['weights'][key]
                for i in range(len(client_updates))
            )
            aggregated_weights[key] = weighted_sum
    
        return aggregated_weights
    
    def _fedavg_old_preCluster(
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
    
    
#=============================================================================#
#=============================================================================#
#=============================================================================#
""" VERSIONE DEL FED-TRAIN-ROUND DI TIPO B: OVVERO COLLABORA CON LA VERSIONE DEL FED-SERVER DI TIPO B E CON IL FED-CLIENT DI TIPO B. 
IN QUESTA TIPOLOGIA, NON SI USA PIU' K-MEANS, SI RICORRE AL FILE METADATA DA CUI PRELEVARE IN MODO PRECISO
I CLUESTER COSì CHE RISPECCHINO QUELLI EFFETTIVI E REALI """

class FederatedServer_B_alt(Step):
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
        # from sklearn.cluster import KMeans
        # from sklearn.metrics.pairwise import cosine_similarity
        
        logger.info("   🔢 Calcolo Weighted FedAvg con Clustering...")
        
        client_attack_info = state.get('federated.client_attack_info', dict)
        
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
        
        # ═══════════════════════════════════════════════════════════════════════
        # 🆕 FASE 2: CLUSTERING SUPERVISIONATO (Basato su Metadata)
        # ═══════════════════════════════════════════════════════════════════════

        logger.info("   🎯🎯 Calcolo Supervised Clustering (basato su target_column)...")

        # Step 1: Estrai specializzazione da metadata
        host_specialization = state.get('federated.host_specialization', dict)

        # Step 2: Crea mapping client → target_column
        client_to_target = {}
        for client_id in client_ids:
            # Converti 'host_172_31_0_2' → '172.31.0.2'
            ip_key = client_id.replace('host_', '').replace('_', '.')
            spec = host_specialization.get(ip_key, {})
            target = spec.get('target_column', None)
            client_to_target[client_id] = target

        # Step 3: Crea mapping target → cluster_id
        unique_targets = sorted(list(set(client_to_target.values()) - {None}))
        target_to_cluster = {target: i for i, target in enumerate(unique_targets)}

        logger.debug(f"   📊 Attacchi unici trovati: {len(unique_targets)}")
        for target, cluster_id in target_to_cluster.items():
            logger.debug(f"      {target} → Cluster {cluster_id}")

        # Step 4: Assegna cluster_id basandoti sul target
        cluster_labels = []
        for client_id in client_ids:
            target = client_to_target[client_id]
            if target is None:
                logger.warning(f"   ⚠️ {client_id}: Nessun target definito! Assegno cluster -1")
                cluster_labels.append(-1)
            else:
                cluster_labels.append(target_to_cluster[target])

        cluster_labels = np.array(cluster_labels)
        num_clusters = len(unique_targets)

        # ✅ CLUSTERING STABILE: Salva mapping solo al Round 0
        if not state.has('federated.stable_target_to_cluster'):
            state.set('federated.stable_target_to_cluster', target_to_cluster, dict)
            state.set('federated.stable_client_to_target', client_to_target, dict)
            logger.debug(f" 🆕 Primo round: {num_clusters} cluster supervisionati creati")
        else:
            # Round successivi: riusa mapping esistente
            target_to_cluster = state.get('federated.stable_target_to_cluster', dict)
            client_to_target = state.get('federated.stable_client_to_target', dict)
            logger.debug(f" 🔄 Round successivo: riutilizzo cluster esistenti")

        # Log distribuzione
        logger.info(f"   🎯 Clustering supervisionato: {num_clusters} cluster identificati")
        for i in range(num_clusters):
            members = [client_ids[j] for j in range(len(cluster_labels)) if cluster_labels[j] == i]
            target_name = [k for k, v in target_to_cluster.items() if v == i][0]
            logger.info(f"      Cluster {i} ({target_name}): {len(members)} client → {members[:3]}...")
        
        
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
            aggregation_weights = []
            for update in cluster_updates:
                client_id = update.get('client_id')
                num_samples = update['num_samples']
                info = client_attack_info.get(client_id, {})
                attack_ratio = info.get('attack_ratio', 0.1)
                weight = np.sqrt(attack_ratio) * num_samples
                aggregation_weights.append(weight)
            
            total_weight = sum(aggregation_weights)
            if total_weight > 0:
                aggregation_weights = [w / total_weight for w in aggregation_weights]
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