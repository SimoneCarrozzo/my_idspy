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
class FederatedClient(Step):
    
    def __init__(
        self,
        client_id: str,  # Es. "host_172_31_0_2"
        local_epochs: int = 5,  # Quante epoche addestrare localmente
        batch_size: int = 1024,
        num_workers: int = 0,
        device: Optional[torch.device] = None,
        in_scope: str = "federated",
        out_scope: str = "federated",
        name: Optional[str] = None,
    ):
        self.client_id = client_id
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        
        super().__init__(
            name=name or f"federated_client_{client_id}",
            in_scope=in_scope,
            out_scope=out_scope,
        )
    
    @Step.requires(
        client_data=pd.DataFrame,  # Dati locali del client
        global_model=nn.Module,     # Modello globale ricevuto dal server
        loss_fn=nn.Module,          # Loss function
        optimizer_class=type,       # Classe optimizer (es. torch.optim.Adam)
        optimizer_kwargs=dict,      # Parametri optimizer (es. {'lr': 0.001})
    )
    @Step.provides(
        local_weights=dict,         # Pesi del modello dopo training locale
        local_metrics=dict,         # Metriche locali (loss, accuracy)
        num_samples=int,            # Numero di sample usati
    )
    def run(
        self,
        state: State,
        client_data: pd.DataFrame,
        global_model: nn.Module,
        loss_fn: nn.Module,
        optimizer_class: type,
        optimizer_kwargs: dict,
    ) -> Optional[Dict[str, Any]]:
        """
        Addestra il modello localmente per K epoche.
        """
        logger.info(f"🔧 Client {self.client_id}: Inizio training locale...")
        device = self.device or state.get("device", torch.device)

        # 1️⃣ PREPARAZIONE DATI
        # Separa features e target
        # Assumiamo che client_data abbia già colonne numeriche/categoriche processate
        # e una colonna 'Attack' come target
        
        if 'Attack' not in client_data.columns:
            raise ValueError(f"Client {self.client_id}: colonna 'Attack' non trovata!")
                
        # ⭐ SEPARA NUMERICAL E CATEGORICAL (basandoti sullo schema)
        # Assumiamo che tu abbia già definito quali sono numerical e categorical
        # Identifica colonne numeriche e categoriche

        # Feature numeriche (float64)
        numerical_cols = client_data.select_dtypes(include=['float64', 'float32']).columns.tolist()

        # Feature categoriche (int32, int64 ma NON Attack)
        categorical_cols = [col for col in client_data.columns 
                        if col not in numerical_cols 
                        and col != 'Attack'
                        and col not in ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'original_Attack']]

        logger.info(f"   📊 {len(numerical_cols)} numerical, {len(categorical_cols)} categorical")

        # Estrai dati
        # X_num = client_data[numerical_cols].values.astype(np.float32) ----> poichè causa overflow, sostituito con:
        X_num_raw = client_data[numerical_cols].values
        # Clipping a un range sicuro per float32
        X_num_raw = np.clip(X_num_raw, -1e10, 1e10)
        
        X_num = X_num_raw.astype(np.float32)
        
        X_cat = client_data[categorical_cols].values.astype(np.int64)
        y = client_data['Attack'].values.astype(np.int64)

        ###################################################################
        # ⭐ FIX: Controlla se il dataset è vuoto
        if len(X_num) == 0:
            logger.warning(f"⚠️ Client {self.client_id}: NESSUN CAMPIONE VALIDO! Skip training.")
            return {
                'local_weights': None,  # Segnala al server di ignorare questo client
                'local_metrics': {
                    'client_id': self.client_id,
                    'error': 'no_samples',
                    'num_samples': 0,
                },
                'num_samples': 0,
            }

        ###################################################################
        # ⭐ SANITIZZAZIONE DELLE FEATURE NUMERICHE:
        # Fix 1: Rimuovi inf/nan dalle feature numeriche
        X_num = np.nan_to_num(X_num, nan=0.0, posinf=0.0, neginf=0.0)

        # Fix 2: Clip valori estremi (evita overflow in float32)
        X_num = np.clip(X_num, -1e6, 1e6)

        logger.info(f"   🧹 Sanitizzato X_num: range=[{X_num.min():.2f}, {X_num.max():.2f}]")
  
        # Crea tensori
        X_num_tensor = torch.from_numpy(X_num)
        X_cat_tensor = torch.from_numpy(X_cat)
        y_tensor = torch.from_numpy(y)

        # ⭐ CREA DATASET CON TUPLE (num, cat, label)
        from torch.utils.data import TensorDataset

        dataset = TensorDataset(X_num_tensor, X_cat_tensor, y_tensor)

        # Crea DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )

        num_samples = len(dataset)
        logger.info(f"   📊 Client {self.client_id}: {num_samples} samples")
        
        # 2️⃣ COPIA MODELLO GLOBALE (per non modificare l'originale)
        local_model = copy.deepcopy(global_model)
        local_model.to(device)
        local_model.train()

        # 3️⃣ CREA OPTIMIZER LOCALE
        local_optimizer = optimizer_class(
            local_model.parameters(),
            **optimizer_kwargs
        )

        # 4️⃣ TRAINING LOCALE per K epoche
        epoch_losses = []

        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            num_batches = 0

            # ⭐ ora unpack 3 valori invece di 2
            for batch_idx, (X_num_batch, X_cat_batch, y_batch) in enumerate(dataloader):
                X_num_batch = X_num_batch.to(device)
                X_cat_batch = X_cat_batch.to(device)
                y_batch = y_batch.to(device)
                
                # ⭐ CREAZIONE DEL DIZIONARIO CHE IL MODELLO SI ASPETTA
                features = {
                    'numerical': X_num_batch,
                    'categorical': X_cat_batch
                }
                
                # Forward pass
                local_optimizer.zero_grad()
                outputs = local_model(features)  # ⭐ Passa il dizionario
                
                # ⭐ ESTRAZIONE DEI LOGITS dall'oggetto ModelOutput
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                # ⭐ UTILIZZO DELLA ClassificationLoss correttamente
                loss = loss_fn(logits, y_batch)  # ✅ Ora passa il tensore
                #loss = loss_fn(outputs, y_batch)
                
                # ⭐ AGGIUNTA CONTROLLO NaN:
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"   ⚠️ NaN/Inf loss detected at batch {batch_idx}, skipping")
                    continue  # Salta questo batch
                
                loss.backward()
                
                # ⭐ Gradient Clipping
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
                
                local_optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            epoch_losses.append(avg_loss)

            logger.info(
                f"      Epoch {epoch+1}/{self.local_epochs}: "
                f"Loss = {avg_loss:.4f}"
            )

        # 5️⃣ ESTRAI PESI AGGIORNATI
        local_weights = {
            name: param.cpu().clone().detach()
            for name, param in local_model.state_dict().items()
        }
               
        # 6️⃣ METRICHE LOCALI
        local_metrics = {
            'client_id': self.client_id,
            'final_loss': epoch_losses[-1],
            'avg_loss': sum(epoch_losses) / len(epoch_losses),
            'num_samples': num_samples,
            'local_epochs': self.local_epochs,
        }
        
        logger.info(
            f"✅ Client {self.client_id}: Training completato | "
            f"Final Loss: {local_metrics['final_loss']:.4f}"
        )
        
        return {
            'local_weights': local_weights,
            'local_metrics': local_metrics,
            'num_samples': num_samples,
        }
################################################################ LA CLASSE CHE SEGUE QUà SOTTO è UGUALE A QUELLA SOPRA
class FedClient(Step):
    """
    Rappresenta un singolo client nel sistema federato.
    
    Responsabilità:
    - Caricare i propri dati locali
    - Addestrare il modello per K epoche
    - Restituire i pesi aggiornati
    
    Analogia: È come un ospedale che addestra sul suo dataset privato.
    """
    
    def __init__(
        self,
        client_id: str,  # Es. "host_172_31_0_2"
        local_epochs: int = 5,  # Quante epoche addestrare localmente
        batch_size: int = 1024,
        num_workers: int = 0,
        device: Optional[torch.device] = None,
        in_scope: str = "federated",
        out_scope: str = "federated",
        name: Optional[str] = None,
    ):
        self.client_id = client_id
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        
        super().__init__(
            name=name or f"federated_client_{client_id}",
            in_scope=in_scope,
            out_scope=out_scope,
        )
    
    @Step.requires(
        client_data=pd.DataFrame,  # Dati locali del client
        global_model=nn.Module,     # Modello globale ricevuto dal server
        loss_fn=nn.Module,          # Loss function
        optimizer_class=type,       # Classe optimizer (es. torch.optim.Adam)
        optimizer_kwargs=dict,      # Parametri optimizer (es. {'lr': 0.001})
    )
    @Step.provides(
        local_weights=dict,         # Pesi del modello dopo training locale
        local_metrics=dict,         # Metriche locali (loss, accuracy)
        num_samples=int,            # Numero di sample usati
    )
    def run(
        self,
        state: State,
        client_data: pd.DataFrame,
        global_model: nn.Module,
        loss_fn: nn.Module,
        optimizer_class: type,
        optimizer_kwargs: dict,
    ) -> Optional[Dict[str, Any]]:
        """
        Addestra il modello localmente per K epoche.
        """
        logger.info(f"🔧 Client {self.client_id}: Inizio training locale...")
        
        device = self.device or state.get("device", torch.device)
        
        # ─────────────────────────────────────────────────────────────
        # 1️⃣ PREPARAZIONE DATI
        # ─────────────────────────────────────────────────────────────
        
        # Separa features e target
        # Assumiamo che client_data abbia già colonne numeriche/categoriche processate
        # e una colonna 'Attack' come target
        
        if 'Attack' not in client_data.columns:
            raise ValueError(f"Client {self.client_id}: colonna 'Attack' non trovata!")
        
        # # Estrai features (tutte tranne Attack)
        # feature_cols = [col for col in client_data.columns if col != 'Attack']
        # X = client_data[feature_cols].values
        # y = client_data['Attack'].values
        
        # # Crea Dataset PyTorch
        # from torch.utils.data import TensorDataset
        
        # X_tensor = torch.FloatTensor(X)
        # y_tensor = torch.LongTensor(y)
        
        # dataset = TensorDataset(X_tensor, y_tensor)
        
        # # Crea DataLoader
        # dataloader = DataLoader(
        #     dataset,
        #     batch_size=self.batch_size,
        #     shuffle=True,
        #     num_workers=self.num_workers,
        #     pin_memory=True if device.type == 'cuda' else False
        # )
        
        # num_samples = len(dataset)
        # logger.info(f"   📊 Client {self.client_id}: {num_samples} samples")
        
        #######################################################################################
        
        # ⭐ SEPARA NUMERICAL E CATEGORICAL (basandoti sullo schema)
        # Assumiamo che tu abbia già definito quali sono numerical e categorical
        # Identifica colonne numeriche e categoriche

        # Feature numeriche (float64)
        numerical_cols = client_data.select_dtypes(include=['float64', 'float32']).columns.tolist()

        # Feature categoriche (int32, int64 ma NON Attack)
        categorical_cols = [col for col in client_data.columns 
                        if col not in numerical_cols 
                        and col != 'Attack'
                        and col not in ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'original_Attack']]

        logger.info(f"   📊 {len(numerical_cols)} numerical, {len(categorical_cols)} categorical")

        # Estrai dati
        X_num = client_data[numerical_cols].values.astype(np.float32)
        X_cat = client_data[categorical_cols].values.astype(np.int64)
        y = client_data['Attack'].values.astype(np.int64)

        # ⭐ SANITIZZAZIONE DELLE FEATURE NUMERICHE:
        # Fix 1: Rimuovi inf/nan dalle feature numeriche
        X_num = np.nan_to_num(X_num, nan=0.0, posinf=0.0, neginf=0.0)

        # Fix 2: Clip valori estremi (evita overflow in float32)
        X_num = np.clip(X_num, -1e6, 1e6)

        logger.info(f"   🧹 Sanitizzato X_num: range=[{X_num.min():.2f}, {X_num.max():.2f}]")

                
        # Crea tensori
        X_num_tensor = torch.from_numpy(X_num)
        X_cat_tensor = torch.from_numpy(X_cat)
        y_tensor = torch.from_numpy(y)

        # ⭐ CREA DATASET CON TUPLE (num, cat, label)
        from torch.utils.data import TensorDataset

        dataset = TensorDataset(X_num_tensor, X_cat_tensor, y_tensor)

        # Crea DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )

        num_samples = len(dataset)
        logger.info(f"   📊 Client {self.client_id}: {num_samples} samples")
        
        # ─────────────────────────────────────────────────────────────
        # 2️⃣ COPIA MODELLO GLOBALE (per non modificare l'originale)
        # ─────────────────────────────────────────────────────────────
        
        local_model = copy.deepcopy(global_model)
        local_model.to(device)
        local_model.train()
        
        # ─────────────────────────────────────────────────────────────
        # 3️⃣ CREA OPTIMIZER LOCALE
        # ─────────────────────────────────────────────────────────────
        
        local_optimizer = optimizer_class(
            local_model.parameters(),
            **optimizer_kwargs
        )
        
        # ─────────────────────────────────────────────────────────────
        # 4️⃣ TRAINING LOCALE per K epoche
        # ─────────────────────────────────────────────────────────────
        
        epoch_losses = []
        
        # for epoch in range(self.local_epochs):
        #     epoch_loss = 0.0
        #     num_batches = 0
            
        #     for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
        #         X_batch = X_batch.to(device)
        #         y_batch = y_batch.to(device)
                
        #         # Forward pass
        #         local_optimizer.zero_grad()
        #         outputs = local_model(X_batch)
        #         loss = loss_fn(outputs, y_batch)
                
        #         # Backward pass
        #         loss.backward()
        #         local_optimizer.step()
                
        #         epoch_loss += loss.item()
        #         num_batches += 1
            
        #     avg_loss = epoch_loss / num_batches
        #     epoch_losses.append(avg_loss)
            
        #     logger.info(
        #         f"      Epoch {epoch+1}/{self.local_epochs}: "
        #         f"Loss = {avg_loss:.4f}"
        #     )
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            num_batches = 0

            # ⭐ ora unpack 3 valori invece di 2
            for batch_idx, (X_num_batch, X_cat_batch, y_batch) in enumerate(dataloader):
                X_num_batch = X_num_batch.to(device)
                X_cat_batch = X_cat_batch.to(device)
                y_batch = y_batch.to(device)
                
                # ⭐ CREAZIONE DEL DIZIONARIO CHE IL MODELLO SI ASPETTA
                features = {
                    'numerical': X_num_batch,
                    'categorical': X_cat_batch
                }
                
                # Forward pass
                local_optimizer.zero_grad()
                outputs = local_model(features)  # ⭐ Passa il dizionario
                
                # ⭐ ESTRAZIONE DEI LOGITS dall'oggetto ModelOutput
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                # ⭐ UTILIZZO DELLA ClassificationLoss correttamente
                loss = loss_fn(logits, y_batch)  # ✅ Ora passa il tensore
                #loss = loss_fn(outputs, y_batch)
                
                # ⭐ AGGIUNTA CONTROLLO NaN:
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"   ⚠️ NaN/Inf loss detected at batch {batch_idx}, skipping")
                    continue  # Salta questo batch
                
                loss.backward()
                
                # ⭐ Gradient Clipping
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
                
                local_optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            epoch_losses.append(avg_loss)

            logger.info(
                f"      Epoch {epoch+1}/{self.local_epochs}: "
                f"Loss = {avg_loss:.4f}"
            )
        # ─────────────────────────────────────────────────────────────
        # 5️⃣ ESTRAI PESI AGGIORNATI
        # ─────────────────────────────────────────────────────────────
        
        local_weights = {
            name: param.cpu().clone().detach()
            for name, param in local_model.state_dict().items()
        }
        
        # ─────────────────────────────────────────────────────────────
        # 6️⃣ METRICHE LOCALI
        # ─────────────────────────────────────────────────────────────
        
        local_metrics = {
            'client_id': self.client_id,
            'final_loss': epoch_losses[-1],
            'avg_loss': sum(epoch_losses) / len(epoch_losses),
            'num_samples': num_samples,
            'local_epochs': self.local_epochs,
        }
        
        logger.info(
            f"✅ Client {self.client_id}: Training completato | "
            f"Final Loss: {local_metrics['final_loss']:.4f}"
        )
        
        return {
            'local_weights': local_weights,
            'local_metrics': local_metrics,
            'num_samples': num_samples,
        }
""" class FederatedClient(Step):
    
    # Rappresenta un singolo client nel sistema federato.
    
    # Responsabilità:
    # - Caricare i propri dati locali
    # - Addestrare il modello per K epoche
    # - Restituire i pesi aggiornati
    
    # Analogia: È come un ospedale che addestra sul suo dataset privato.
    
    
    def __init__(
        self,
        client_id: str,  # Es. "host_172_31_0_2"
        local_epochs: int = 5,  # Quante epoche addestrare localmente
        batch_size: int = 1024,
        num_workers: int = 0,
        device: Optional[torch.device] = None,
        in_scope: str = "federated",
        out_scope: str = "federated",
        name: Optional[str] = None,
    ):
        self.client_id = client_id
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        
        super().__init__(
            name=name or f"federated_client_{client_id}",
            in_scope=in_scope,
            out_scope=out_scope,
        )
    
    @Step.requires(
        client_data=pd.DataFrame,  # Dati locali del client
        global_model=nn.Module,     # Modello globale ricevuto dal server
        loss_fn=nn.Module,          # Loss function
        optimizer_class=type,       # Classe optimizer (es. torch.optim.Adam)
        optimizer_kwargs=dict,      # Parametri optimizer (es. {'lr': 0.001})
    )
    @Step.provides(
        local_weights=dict,         # Pesi del modello dopo training locale
        local_metrics=dict,         # Metriche locali (loss, accuracy)
        num_samples=int,            # Numero di sample usati
    )
    def run(
        self,
        state: State,
        client_data: pd.DataFrame,
        global_model: nn.Module,
        loss_fn: nn.Module,
        optimizer_class: type,
        optimizer_kwargs: dict,
    ) -> Optional[Dict[str, Any]]:
        
        ###Addestra il modello localmente per K epoche.
        
        logger.info(f"🔧 Client {self.client_id}: Inizio training locale...")
        
        device = self.device or state.get("device", torch.device)
        
        # ─────────────────────────────────────────────────────────────
        # 1️⃣ PREPARAZIONE DATI
        # ─────────────────────────────────────────────────────────────
        
        # Separa features e target
        # Assumiamo che client_data abbia già colonne numeriche/categoriche processate
        # e una colonna 'Attack' come target
        
        if 'Attack' not in client_data.columns:
            raise ValueError(f"Client {self.client_id}: colonna 'Attack' non trovata!")
        
        # Estrai features (tutte tranne Attack)
        feature_cols = [col for col in client_data.columns if col != 'Attack']
        X = client_data[feature_cols].values
        y = client_data['Attack'].values
        
        # Crea Dataset PyTorch
        from torch.utils.data import TensorDataset
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Crea DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        num_samples = len(dataset)
        logger.info(f"   📊 Client {self.client_id}: {num_samples} samples")
        
        # ─────────────────────────────────────────────────────────────
        # 2️⃣ COPIA MODELLO GLOBALE (per non modificare l'originale)
        # ─────────────────────────────────────────────────────────────
        
        local_model = copy.deepcopy(global_model)
        local_model.to(device)
        local_model.train()
        
        # ─────────────────────────────────────────────────────────────
        # 3️⃣ CREA OPTIMIZER LOCALE
        # ─────────────────────────────────────────────────────────────
        
        local_optimizer = optimizer_class(
            local_model.parameters(),
            **optimizer_kwargs
        )
        
        # ─────────────────────────────────────────────────────────────
        # 4️⃣ TRAINING LOCALE per K epoche
        # ─────────────────────────────────────────────────────────────
        
        epoch_losses = []
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                # Forward pass
                local_optimizer.zero_grad()
                outputs = local_model(X_batch)
                loss = loss_fn(outputs, y_batch)
                
                # Backward pass
                loss.backward()
                local_optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            epoch_losses.append(avg_loss)
            
            logger.info(
                f"      Epoch {epoch+1}/{self.local_epochs}: "
                f"Loss = {avg_loss:.4f}"
            )
        
        # ─────────────────────────────────────────────────────────────
        # 5️⃣ ESTRAI PESI AGGIORNATI
        # ─────────────────────────────────────────────────────────────
        
        local_weights = {
            name: param.cpu().clone().detach()
            for name, param in local_model.state_dict().items()
        }
        
        # ─────────────────────────────────────────────────────────────
        # 6️⃣ METRICHE LOCALI
        # ─────────────────────────────────────────────────────────────
        
        local_metrics = {
            'client_id': self.client_id,
            'final_loss': epoch_losses[-1],
            'avg_loss': sum(epoch_losses) / len(epoch_losses),
            'num_samples': num_samples,
            'local_epochs': self.local_epochs,
        }
        
        logger.info(
            f"✅ Client {self.client_id}: Training completato | "
            f"Final Loss: {local_metrics['final_loss']:.4f}"
        )
        
        return {
            'local_weights': local_weights,
            'local_metrics': local_metrics,
            'num_samples': num_samples,
        } """