from src.idspy.core.step import Step, State
from src.idspy.nn.models.base import BaseModel
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, Any, Optional
import copy
import logging

logger = logging.getLogger(__name__)


class EvaluateClusterModels_C(Step):
    """Valuta ogni cluster model sul suo sottoinsieme di attacco specifico."""
    
    def __init__(self, batch_size=1024):
        super().__init__(name="evaluate_cluster_models")
        self.batch_size = batch_size
    
    @Step.requires(
        global_model=BaseModel,
        aggregated_test=pd.DataFrame,
        device=torch.device,
        cluster_specific_weights=dict,  # Pesi per ogni cluster
        # stable_target_to_cluster=dict,  # Mapping attacco → cluster_id
        real_cat_cardinalities=list, # ← NUOVO: cardinalities per clipping
    )
    @Step.provides(cluster_metrics=dict, aggregated_metrics=dict, cluster_results_detailed=dict) # ← NUOVO: contiene predictions/targets
    def run(
        self, 
        state: State, 
        global_model: BaseModel, 
        aggregated_test: pd.DataFrame, 
        device: torch.device,
        cluster_specific_weights: dict,
        # stable_target_to_cluster: dict,
        real_cat_cardinalities: list, # ← NUOVO: cardinalities per clipping
    ) -> Dict[str, Any]:
        
        logger.info("🎯 Valutazione per Cluster...")
        
        # ═══════════════════════════════════════════════════════════════════
        # 1. PREPARAZIONE COLONNE (come client)
        # ═══════════════════════════════════════════════════════════════════
        
        KNOWN_CATEGORICAL_COLS = [
            'L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO',
            'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS',
            'ICMP_TYPE', 'ICMP_IPV4_TYPE', 'DNS_QUERY_ID', 'DNS_QUERY_TYPE'
        ]
        
        # Identifica colonne is_* disponibili (per filtraggio)
        available_is_cols = [c for c in aggregated_test.columns if c.startswith('is_')]
        
        cluster_results = {}  # Per metriche aggregate
        cluster_results_detailed = {}  # ← NUOVO: per predictions/targets
        
        total_samples = 0
        weighted_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        
        # ═══════════════════════════════════════════════════════════════════
        # 2. VALUTAZIONE PER OGNI CLUSTER
        # ═══════════════════════════════════════════════════════════════════
        
        # ✅ NUOVO: Tutti i cluster valutano lo STESSO attacco
        target_attack = state.get('federated.target_attack', str)

        logger.info(f"🎯 Valutazione cluster su attacco comune: {target_attack}")

        # Per ogni cluster (identificato da clustering unsupervised)
        cluster_labels = state.get('federated.client_to_cluster', dict)
        unique_clusters = sorted(set(cluster_labels.values()))
        
        for cluster_id in unique_clusters:
    
            if cluster_id not in cluster_specific_weights:
                logger.warning(f"⚠️ Cluster {cluster_id}: Nessun peso disponibile, skip")
                continue
            
            # ✅ NUOVO: Filtra test set per attacco comune
            if target_attack not in aggregated_test.columns:
                logger.error(f"❌ Colonna {target_attack} non trovata nel test set!")
                continue
            
            # Usa TUTTI i sample (positivi + negativi) per valutazione binaria
            cluster_test = aggregated_test.copy()
            targets = cluster_test[target_attack].values.astype(np.int64)
            
            n_positive = (targets == 1).sum()
            n_negative = (targets == 0).sum()
            
            
            if n_positive == 0:
                logger.warning(f"⚠️ Cluster {cluster_id} ({target_attack}): 0 samples positivi, skip")
                continue
            
            logger.debug(
                f"   📊 Cluster {cluster_id} ({target_attack}): "
                f"{n_positive} positivi, {n_negative} negativi"
            )

            logger.info(f"   📊 Cluster {cluster_id} ({target_attack}): {len(cluster_test)} samples")

            # ───────────────────────────────────────────────────────────────
            # 2.2 PREPARA DATI (stessa logica client)
            # ───────────────────────────────────────────────────────────────
            
            cols_to_drop = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'original_Attack', 'Label']
            cluster_test = cluster_test.drop(columns=[c for c in cols_to_drop if c in cluster_test.columns])
            
            # Escludi is_* (data leakage)
            excluded_cols = {'Attack'}
            excluded_cols.update(available_is_cols)
            
            categorical_cols = [c for c in KNOWN_CATEGORICAL_COLS 
                                if c in cluster_test.columns and c not in excluded_cols]
            numerical_cols = [c for c in cluster_test.columns 
                              if c not in categorical_cols and c not in excluded_cols]
            
            X_num = cluster_test[numerical_cols].values.astype(np.float32)
            
            X_cat_raw = cluster_test[categorical_cols].values
            if X_cat_raw.dtype in [np.float64, np.float32]:
                X_cat = np.round(X_cat_raw).clip(min=0).astype(np.int64)
            else:
                X_cat = X_cat_raw.astype(np.int64)
            
            ###################
            # NUOVOOOO⭐ FIX CRITICO: Clip rispetto alle cardinalities del modello
            for i, cardinality in enumerate(real_cat_cardinalities):
                max_val = X_cat[:, i].max()
                if max_val >= cardinality:
                    logger.warning(f"⚠️ Col {categorical_cols[i]}: max={max_val} >= card={cardinality}, clipping...")
                X_cat[:, i] = np.clip(X_cat[:, i], 0, cardinality - 1)
            ##############
            
            # ✅ Target: usa target_attack condiviso
            targets = cluster_test[target_attack].values.astype(np.int64)
            
            # Sanity check
            X_num = np.clip(X_num, -10, 10)
            X_num = np.nan_to_num(X_num, nan=0.0, posinf=10.0, neginf=-10.0)
            
            # ───────────────────────────────────────────────────────────────
            # 2.3 CARICA CLUSTER MODEL
            # ───────────────────────────────────────────────────────────────
            
            cluster_model = copy.deepcopy(global_model)
            cluster_model.load_state_dict(cluster_specific_weights[cluster_id])
            cluster_model.to(device)
            cluster_model.eval()
            
            # ───────────────────────────────────────────────────────────────
            # 2.4 INFERENZA
            # ───────────────────────────────────────────────────────────────
            
            all_preds = []
            with torch.no_grad():
                for i in range(0, len(X_num), self.batch_size):
                    batch_num = torch.from_numpy(X_num[i:i+self.batch_size]).to(device)
                    batch_cat = torch.from_numpy(X_cat[i:i+self.batch_size]).to(device)
                    
                    features = {'numerical': batch_num, 'categorical': batch_cat}
                    output = cluster_model(features)
                    
                    logits = output.logits if hasattr(output, 'logits') else output
                    batch_preds = (torch.sigmoid(logits) > 0.5).long()
                    
                    all_preds.extend(batch_preds.cpu().numpy())
            
            predictions = np.array(all_preds).flatten() # ← Assicura 1D
            
            # ───────────────────────────────────────────────────────────────
            # 2.5 CALCOLA METRICHE
            # ───────────────────────────────────────────────────────────────
            
            metrics = {
                'cluster_id': cluster_id,
                'target_column': target_attack,
                'num_samples': len(targets),
                'accuracy': accuracy_score(targets, predictions),
                'precision': precision_score(targets, predictions, average='binary', zero_division=0),
                'recall': recall_score(targets, predictions, average='binary', zero_division=0),
                'f1': f1_score(targets, predictions, average='binary', zero_division=0),
                'confusion_matrix': confusion_matrix(targets, predictions).tolist()
            }
            
            # ✅ SALVA per round_history (metriche aggregate)
            cluster_results[cluster_id] = metrics
            
            # ✅ SALVA per ClusterRoundMetrics (con predictions/targets)
            cluster_results_detailed[cluster_id] = {
                'predictions': predictions,
                'targets': targets,
                'target_column': target_attack,
                **metrics  # Include anche metriche calcolate
            }
            
            logger.info(
                f"   ✅ Cluster {cluster_id}: Acc={metrics['accuracy']:.4f}, "
                f"F1={metrics['f1']:.4f}, Samples={metrics['num_samples']}"
            )
            
            # Accumula per media pesata
            total_samples += metrics['num_samples']
            for key in ['accuracy', 'precision', 'recall', 'f1']:
                weighted_metrics[key] += metrics[key] * metrics['num_samples']
        
        # ═══════════════════════════════════════════════════════════════════
        # 3. AGGREGAZIONE FINALE
        # ═══════════════════════════════════════════════════════════════════
        
        if total_samples > 0:
            for key in weighted_metrics:
                weighted_metrics[key] /= total_samples
        
        aggregated_metrics = {
            'total_samples': total_samples,
            'num_clusters_evaluated': len(cluster_results),
            **weighted_metrics
        }
        
        logger.info(f"\n📊 Metriche Aggregate (pesate):")
        logger.info(f"   • Accuracy: {aggregated_metrics['accuracy']:.4f}")
        logger.info(f"   • F1-Score: {aggregated_metrics['f1']:.4f}")
        logger.info(f"   • Total Samples: {total_samples}")
        
        return {
            'cluster_metrics': cluster_results,
            'aggregated_metrics': aggregated_metrics,
            'cluster_results_detailed': cluster_results_detailed  # ← NUOVO: per ClusterRoundMetrics

        }