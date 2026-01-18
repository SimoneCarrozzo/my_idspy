import logging

from sklearn import logger
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.idspy.core.step import Step
from src.idspy.core.state import State
from src.idspy.nn.models.base import BaseModel

logger = logging.getLogger(__name__)

class EvaluateGlobalModel_alt(Step):
    def __init__(self, batch_size=1024):
        super().__init__(name="evaluate_global_model")
        self.batch_size = batch_size
    
    @Step.requires(
        global_model=BaseModel,
        aggregated_test=pd.DataFrame,
        device=torch.device
    )
    @Step.provides(test_metrics=dict)
    def run(self, state: State, global_model, aggregated_test, device):
        # 1. Preparazione Modello
        global_model.eval()
        
        
        # 🔧 FIX: Rimuovi colonne non necessarie (COME NEL MAIN)
        cols_to_drop = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'original_Attack', 'Label']
        existing_cols_to_drop =[c for c in cols_to_drop if c in aggregated_test.columns]
        final_correct_df = aggregated_test.drop(columns=existing_cols_to_drop)
    
        
        """ # 2. Preparazione Dati (stessa logica del client)
        numerical_cols = final_correct_df.select_dtypes(include=['float64', 'float32']).columns.tolist()
        categorical_cols = [c for c in final_correct_df.columns if c not in numerical_cols and c != 'Attack']
        
        X_num = final_correct_df[numerical_cols].values.astype(np.float32)
        X_cat = final_correct_df[categorical_cols].values.astype(np.int32)
        
        targets = final_correct_df['Attack'].values.astype(np.int64)
        
        # 3. Clipping e Pulizia
        X_num = np.clip(X_num, -10, 10)
        X_num = np.nan_to_num(X_num, nan=0.0, posinf=10.0, neginf=-10.0) """
        # ═══════════════════════════════════════════════════════════════════════
        # 2. Preparazione Dati (STESSA LOGICA DEL CLIENT)
        # ═══════════════════════════════════════════════════════════════════════

        # 🔧 Rimuovi colonne non necessarie
        cols_to_drop = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'original_Attack', 'Label']
        existing_cols_to_drop = [c for c in cols_to_drop if c in aggregated_test.columns]
        final_correct_df = aggregated_test.drop(columns=existing_cols_to_drop)

        # ⭐ DEFINIZIONE ESPLICITA (copia da client.py)
        KNOWN_CATEGORICAL_COLS = [
            'L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO',
            'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS',
            'ICMP_TYPE', 'ICMP_IPV4_TYPE', 'DNS_QUERY_ID', 'DNS_QUERY_TYPE'
        ]

        # Escludi colonne is_* (data leakage)
        all_target_cols = [c for c in final_correct_df.columns if c.startswith('is_')]
        excluded_cols = {'Attack'}
        excluded_cols.update(all_target_cols)

        # Categorical: quelle in KNOWN_CATEGORICAL_COLS
        categorical_cols = [c for c in KNOWN_CATEGORICAL_COLS 
                            if c in final_correct_df.columns and c not in excluded_cols]

        # Numerical: tutto il resto
        numerical_cols = [c for c in final_correct_df.columns 
                        if c not in categorical_cols 
                        and c not in excluded_cols]

        logger.debug(f"   📊 Eval: {len(numerical_cols)} numerical, {len(categorical_cols)} categorical")

        # Estrai dati
        X_num = final_correct_df[numerical_cols].values.astype(np.float32)

        # ⭐ Gestisci float normalizzati (come nel client)
        X_cat_raw = final_correct_df[categorical_cols].values
        if X_cat_raw.dtype in [np.float64, np.float32]:
            X_cat = np.round(X_cat_raw).clip(min=0).astype(np.int64)
        else:
            X_cat = X_cat_raw.astype(np.int64)

        targets = final_correct_df['Attack'].values.astype(np.int64)

        # 3. Clipping e Pulizia
        X_num = np.clip(X_num, -10, 10)
        X_num = np.nan_to_num(X_num, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # 4. Inferenza a Batch
        # --- FASE 2: Ciclo di Inferenza (Simile al tuo snippet) ---
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(X_num), self.batch_size):
                
                # Estrazione batch e spostamento su device solo ora (efficienza memoria)
                batch_num = torch.from_numpy(X_num[i:i+self.batch_size]).to(device)
                batch_cat = torch.from_numpy(X_cat[i:i+self.batch_size]).to(device)
                
                batch_y = torch.from_numpy(targets[i:i+self.batch_size]).to(device)
                
                # Costruzione dizionario features
                features = {
                    'numerical': batch_num,
                    'categorical': batch_cat
                }
                
                # Passa il dizionario al modello
                output = global_model(features)
                
                # Gestione output: se è un oggetto custom prendiamo .logits, altrimenti l'output stesso
                logits = output.logits if hasattr(output, 'logits') else output
                # logits = output.logits
                
                # Se vuoi la loss (assumendo state contenga la loss_fn)
                # loss = state.get('federated.loss_fn')(logits, batch_y)
                               
                # Calcolo predizioni (indice della classe con probabilità maggiore)
                # batch_preds = torch.argmax(logits, dim=1)
                
                # Binary classification: apply sigmoid + threshold
                batch_preds = (torch.sigmoid(logits) > 0.5).long()
                
                # Spostiamo su CPU e accumuliamo
                all_preds.extend(batch_preds.cpu().numpy())
        
        # y_pred = np.concatenate(all_preds)
        # Convertiamo la lista in array numpy finale
        predictions = np.array(all_preds)
        
        # --- FASE 3: Calcolo Metriche e Return ---
        
        
        # 5. Calcolo Metriche
        metrics = {
            'accuracy': accuracy_score(targets, predictions),
            'precision': precision_score(targets, predictions, average='binary', zero_division=0),
            'recall': recall_score(targets, predictions, average='binary', zero_division=0),
            'f1': f1_score(targets, predictions, average='binary', zero_division=0),
            'confusion_matrix': confusion_matrix(targets, predictions).tolist() # .tolist() per eventuale serializzazione JSON
        }
        # Log per vedere subito come va
        logger.info(f"📊 Global Evaluation: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
        
        return {"test_metrics": metrics, "predictions": predictions, "targets": targets}
    
#=═══════════════════════════════════════════════════════════════════════#
#=═══════════════════════════════════════════════════════════════════════#
#=═══════════════════════════════════════════════════════════════════════#
#=═══════════════════════════════════════════════════════════════════════#
""" valutazione per cluster piuttosto che sul modello globale """
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

class EvaluateClusterModels_alt(Step):
    """Valuta ogni cluster model sul suo sottoinsieme di attacco specifico."""
    
    def __init__(self, batch_size=1024):
        super().__init__(name="evaluate_cluster_models")
        self.batch_size = batch_size
    
    @Step.requires(
        global_model=BaseModel,
        aggregated_test=pd.DataFrame,
        device=torch.device,
        cluster_specific_weights=dict,  # Pesi per ogni cluster
        stable_target_to_cluster=dict,  # Mapping attacco → cluster_id
    )
    @Step.provides(cluster_metrics=dict, aggregated_metrics=dict, cluster_results_detailed=dict) # ← NUOVO: contiene predictions/targets
    def run(
        self, 
        state: State, 
        global_model: BaseModel, 
        aggregated_test: pd.DataFrame, 
        device: torch.device,
        cluster_specific_weights: dict,
        stable_target_to_cluster: dict,
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
        
        for target_col, cluster_id in stable_target_to_cluster.items():
            
            # Skip se non abbiamo pesi per questo cluster
            if cluster_id not in cluster_specific_weights:
                logger.warning(f"⚠️ Cluster {cluster_id} ({target_col}): Nessun peso disponibile, skip")
                continue
            
            # ───────────────────────────────────────────────────────────────
            # 2.1 FILTRA TEST SET per questo attacco
            # ───────────────────────────────────────────────────────────────
            
            if target_col not in available_is_cols:
                logger.warning(f"⚠️ Cluster {cluster_id}: Colonna {target_col} non trovata nel test set, skip")
                continue
            
            """ # Filtra solo samples di questo attacco
            cluster_test = aggregated_test[aggregated_test[target_col] == 1].copy()
            
            if len(cluster_test) == 0:
                logger.warning(f"⚠️ Cluster {cluster_id} ({target_col}): 0 samples nel test set, skip")
                continue """
             # ✅ NUOVO: Includi TUTTI i samples (positivi E negativi)
            cluster_test = aggregated_test.copy()
            
            # Verifica esistenza samples positivi
            n_positive = (cluster_test[target_col] == 1).sum()
            n_negative = (cluster_test[target_col] == 0).sum()
            
            if n_positive == 0:
                logger.warning(f"⚠️ Cluster {cluster_id} ({target_col}): 0 samples positivi, skip")
                continue
            
            logger.debug(
                f"   📊 Cluster {cluster_id} ({target_col}): "
                f"{n_positive} positivi, {n_negative} negativi"
            )
            
            logger.info(f"   📊 Cluster {cluster_id} ({target_col}): {len(cluster_test)} samples")
            
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
            
            # ✅ Target: usa il target_col (ORA ha sia 0 che 1)
            targets = cluster_test[target_col].values.astype(np.int64)
            
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
                'target_column': target_col,
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
                'target_column': target_col,
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
 