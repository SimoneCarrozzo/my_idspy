import logging
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.idspy.core.step import Step
from src.idspy.core.state import State
from src.idspy.nn.models.base import BaseModel

class EvaluateGlobalModel(Step):
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
        aggregated_test = aggregated_test.drop(columns=[c for c in cols_to_drop if c in aggregated_test.columns])
    
        
        # 2. Preparazione Dati (stessa logica del client)
        numerical_cols = aggregated_test.select_dtypes(include=['float64', 'float32']).columns.tolist()
        categorical_cols = [c for c in aggregated_test.columns if c not in numerical_cols and c != 'Attack']
        
        X_num = aggregated_test[numerical_cols].values.astype(np.float32)
        X_cat = aggregated_test[categorical_cols].values.astype(np.int32)
        y_true = aggregated_test['Attack'].values.astype(np.int64)
        
        # 3. Clipping e Pulizia
        X_num = np.clip(X_num, -10, 10)
        X_num = np.nan_to_num(X_num, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # 4. Inferenza a Batch
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(X_num), self.batch_size):
                # Estrazione batch e spostamento su device solo ora (efficienza memoria)
                batch_num = torch.from_numpy(X_num[i:i+self.batch_size]).to(device)
                batch_cat = torch.from_numpy(X_cat[i:i+self.batch_size]).to(device)
                
                batch_y = torch.from_numpy(y_true[i:i+self.batch_size]).to(device)
                
                features = {
                    'numerical': batch_num,
                    'categorical': batch_cat
                }
                
                # Passa il dizionario al modello
                output = global_model(features)
                # --- MODIFICA QUI: Estrai il tensore dall'oggetto ModelOutput ---
                # Accediamo all'attributo .logits dell'oggetto restituito
                logits = output.logits
                # logits = global_model(batch_num, batch_cat)
                
                # Se vuoi la loss (assumendo state contenga la loss_fn)
                # loss = state.get('federated.loss_fn')(logits, batch_y)
                
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu().numpy())
        
        y_pred = np.concatenate(all_preds)
        
        # 5. Calcolo Metriche
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist() # .tolist() per eventuale serializzazione JSON
        }
        
        return {"test_metrics": metrics}