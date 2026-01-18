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

# ═══════════════════════════════════════════════════════════════════
# ⚖️ CALCOLO pos_weight con EFFECTIVE NUMBER (Class-Balanced Loss)
# ═══════════════════════════════════════════════════════════════════

def compute_effective_weight(n_pos, n_neg, beta=0.9999):
    """
    Effective Number of Samples (Cui et al., CVPR 2019).
    Gestisce sbilanciamenti estremi meglio di n_neg/n_pos.
    
    Args:
        beta: 0.999 per dataset grandi, 0.99 per piccoli
              Più alto = più enfasi su classi rare
    """
    if n_pos == 0:
        return 50.0  # Fallback
    
    # Effective number per classe
    effective_num_pos = (1 - beta**n_pos) / (1 - beta)
    effective_num_neg = (1 - beta**n_neg) / (1 - beta)
    
    # Peso bilanciato
    weight = effective_num_neg / effective_num_pos
    
    # Clipping conservativo (gestisce già meglio gli estremi)
    return np.clip(weight, 0.1, 200.0)

#=============================================================================#
#=============================================================================#
#=============================================================================#
""" VERSIONE DEL FED-Client DI TIPO C: OVVERO QUELLA CORRETTA DELLA TESI"""

class FederatedClient_C(Step):
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
        # loss_fn=nn.Module,          
        loss_fn_class=type,         # Classe loss function (es. nn.BCEWithLogitsLoss)
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
        # loss_fn: nn.Module,
        loss_fn_class: type,
        optimizer_class: type,
        optimizer_kwargs: dict,
    ) -> Optional[Dict[str, Any]]:
        """
        Addestra il modello localmente per K epoche.
        """

        #Gestione più robusta
        if self.device:
            device = self.device
        else:
            device = state.get("device")
            if not device:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"🔧 Client {self.client_id}: Inizio training locale su {device}...")

        # 1️⃣ PREPARAZIONE DATI

        # ✅ NUOVO: usa target_attack dallo state (uguale per tutti)
        target_attack = state.get('federated.target_attack', str)

        if target_attack not in client_data.columns:
            logger.error(f"❌ {self.client_id}: colonna {target_attack} non trovata!")
            return {
                'local_weights': None,
                'local_metrics': {'error': 'missing_target'},
                'num_samples': 0,
            }

        logger.info(f"   🎯 Client {self.client_id} allena su: {target_attack} (condiviso)")
 
        # ═══════════════════════════════════════════════════════════════════════
        # ⭐ PREVENZIONE DATA LEAKAGE: Esclusione Target + Metadata
        # ═══════════════════════════════════════════════════════════════════════

        #DEFINIZIONE ESPLICITA (copia da main_fed.py)
        KNOWN_CATEGORICAL_COLS = [
            'L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO',
            'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS',
            'ICMP_TYPE', 'ICMP_IPV4_TYPE', 'DNS_QUERY_ID', 'DNS_QUERY_TYPE'
        ]

        # Step 1: Identifica TUTTE le colonne target binarie (is_*)
        all_target_cols = [c for c in client_data.columns if c.startswith('is_')]

        # Step 2: Crea set completo di esclusione
        excluded_cols = {
            'Attack',                    # Target binario generico (se presente)
            target_attack,                  # Target specifico di questo client (es. 'is_infilteration')
            'IPV4_SRC_ADDR',            # Metadata
            'IPV4_DST_ADDR',            # Metadata
            'original_Attack',          # Metadata
            'Label'                     # Metadata
        }

        # Step 3: Aggiungi TUTTE le colonne is_* al set di esclusione
        excluded_cols.update(all_target_cols)

        # ✅ LOG: Verifica esclusioni
        logger.debug(f"   🚫 Escluse {len(excluded_cols)} colonne:")
        logger.debug(f"      • Target: {target_attack}")
        logger.debug(f"      • Altri target (leakage): {[c for c in all_target_cols if c != target_attack]}")
        logger.debug(f"      • Metadata: IPV4_SRC_ADDR, IPV4_DST_ADDR, original_Attack, Label")
        
        
        # Step 4: Separa features numeriche e categoriche
        # Categoriche: quelle in KNOWN_CATEGORICAL_COLS che esistono
        categorical_cols = [c for c in KNOWN_CATEGORICAL_COLS 
                            if c in client_data.columns and c not in excluded_cols]

        # Numeriche: tutto il resto
        numerical_cols = [c for c in client_data.columns 
                        if c not in categorical_cols 
                        and c not in excluded_cols]

        logger.info(f"   📊 {len(numerical_cols)} numerical, {len(categorical_cols)} categorical")

        # Step 5: ✅ SANITY CHECK (verifica che nessuna colonna is_* sia nelle features)
        leaked_cols = [c for c in numerical_cols + categorical_cols if c.startswith('is_')]
        if leaked_cols:
            logger.error(f"   ❌ DATA LEAKAGE RILEVATO: {leaked_cols} sono nelle features!")
            raise ValueError(f"Data leakage: colonne target {leaked_cols} usate come features")

        # Step 6: Estrai dati
        X_num_raw = client_data[numerical_cols].values
        X_num_raw = np.clip(X_num_raw, -1e10, 1e10)
        X_num = X_num_raw.astype(np.float32)
       
        # ✅ NUOVO: Arrotonda float normalizzati e converti a int
        X_cat_raw = client_data[categorical_cols].values

        # Se sono float normalizzati, arrotondali
        if X_cat_raw.dtype == np.float64 or X_cat_raw.dtype == np.float32:
            X_cat = np.round(X_cat_raw).clip(min=0).astype(np.int64)
        else:
            X_cat = X_cat_raw.astype(np.int64)
        
        # ✅ NUOVO: Encoding binario dinamico
        # target_attack = 1, tutto il resto = 0
        y = client_data[target_attack].values.astype(np.int64)

        logger.debug(f"   ✅ Shapes finali: X_num={X_num.shape}, X_cat={X_cat.shape}, y={y.shape}")
        # ═══════════════════════════════════════════════════════════════════════

        # ═══════════════════════════════════════════════════════════════════
        # ⚖️ CALCOLO pos_weight DINAMICO (specifico per questo client)
        # ═══════════════════════════════════════════════════════════════════

        """ n_pos = (y == 1).sum()
        n_neg = (y == 0).sum()
        
        if n_pos > 0:
            pos_weight_value = n_neg / n_pos
            # Limita il range per evitare pesi estremi
            pos_weight_value = np.clip(pos_weight_value, 1.0, 50.0)
        else:
            pos_weight_value = 10.0  # Default se non ci sono positivi

        pos_weight = torch.FloatTensor([pos_weight_value]).to(device)

        # Crea l'istanza della loss con pos_weight specifico
        loss_fn = loss_fn_class(pos_weight=pos_weight).to(device)

        logger.info(
            f"   ⚖️ pos_weight: {pos_weight_value:.4f} "
            f"(positivi={n_pos}, negativi={n_neg})"
        ) """
        
          # Applicazione  CLASS-BALANCED-LOSS
        n_pos = (y == 1).sum()
        n_neg = (y == 0).sum()

        total_samples = n_pos + n_neg

        pos_weight_value = compute_effective_weight(n_pos, n_neg, beta=1-(1/total_samples))
        pos_weight = torch.FloatTensor([pos_weight_value]).to(device)

        loss_fn = loss_fn_class(pos_weight=pos_weight).to(device)

        logger.info(
            f"   ⚖️ pos_weight (effective): {pos_weight_value:.2f} "
            f"(pos={n_pos}, neg={n_neg}, ratio={n_neg/n_pos:.1f})"
        )

        # """ cluster_specific_weights = {
        #     'is_bot': 8.0,              # Era 1.0 → troppo basso (60k pos vs 11k neg)
        #     'is_ddos_attack_hoic': 0.1,  # Era 1.0 → invertito! (457k pos vs 10k neg)
        #     'is_dos_attacks_hulk': 5.0,  # Era 1.12 → aumentato per migliorare recall
        #     'is_infilteration': 80.0,    # Era 50.0 → aumentato (3k pos vs 3M neg)
        #     'is_ddos_attacks_loic_http': 10.0,  # Default se presente
        # }

        # # Usa peso specifico se disponibile, altrimenti calcola dinamico
        # if target_col in cluster_specific_weights:
        #     pos_weight_value = cluster_specific_weights[target_col]
        #     logger.info(f"   🎯 Usando pos_weight ottimizzato per {target_col}: {pos_weight_value}")
        # else:
        #     # Fallback dinamico (SENZA clipping a min=1.0)
        #     if n_pos > 0:
        #         pos_weight_value = n_neg / n_pos
        #         pos_weight_value = np.clip(pos_weight_value, 0.01, 100.0)  # ✅ Permette <1
        #         logger.info(f"   ⚙️ pos_weight calcolato dinamicamente: {pos_weight_value:.2f}")
        #     else:
        #         pos_weight_value = 10.0

        # pos_weight = torch.FloatTensor([pos_weight_value]).to(device)
        # loss_fn = loss_fn_class(pos_weight=pos_weight).to(device)

        # logger.info(
        #     f"   ⚖️ pos_weight finale: {pos_weight_value:.2f} "
        #     f"(positivi={n_pos}, negativi={n_neg})"
        # ) """
        
        
        
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
        logger.debug(f"   ⚙️ Ottimizzatore: {optimizer_class.__name__} con {optimizer_kwargs}")
        
        # 3️⃣.5 CREA SCHEDULER (se configurato)
        scheduler = None
        if state.has('federated.scheduler_config'):
            logger.debug("   ⚙️LO SCHEDUELER ESISTE: Configurazione scheduler trovata.")
            sched_cfg = state.get('federated.scheduler_config', dict)
            
            try:
                # ✅ Usa direttamente la classe (più generico)
                scheduler_class = sched_cfg['type']
                scheduler = scheduler_class(local_optimizer, **sched_cfg['kwargs'])
                
                logger.debug(f"   📉 Scheduler attivato: {scheduler_class.__name__}")
                logger.debug(f"      Parametri: {sched_cfg['kwargs']}")
            except Exception as e:
                logger.error(f"   ❌ Errore creazione scheduler: {e}")
                scheduler = None

        # 4️⃣ TRAINING LOCALE per K epoche
        epoch_losses = []
        epoch_accs = [] # ✨ NEW: Tracciamo l'accuracy
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            num_batches = 0
            correct = 0      # ✨ NEW
            total_preds = 0  # ✨ NEW

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
                #loss = loss_fn(outputs, y_batch) ---> modificato come segue:
                # loss = loss_fn(logits, y_batch)  # ✅ Ora passa il tensore
                
                y_batch_float = y_batch.float().view_as(logits)
                loss = loss_fn(logits, y_batch_float)  # Per BCEWithLogitsLoss, y_batch deve essere float e della stessa shape di logits
                
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
                
                # ✨ NEW: CALCOLO ACCURACY BATCH
                with torch.no_grad():
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    correct += (preds == y_batch_float).sum().item()
                    total_preds += y_batch.size(0)

            avg_loss = epoch_loss / num_batches
            avg_acc = correct / total_preds if total_preds > 0 else 0 # ✨ NEW
            
            epoch_losses.append(avg_loss)
            epoch_accs.append(avg_acc) # ✨ NEW

            logger.info(
                f"      Epoch {epoch+1}/{self.local_epochs}: "
                f"Loss = {avg_loss:.6f} | Acc = {avg_acc*100:.2f}%"
            )
            
            # ✅ Step scheduler (se esiste)
            if scheduler is not None:
                scheduler.step(avg_loss)
                current_lr = local_optimizer.param_groups[0]['lr']
                logger.debug(f" 📉 [DEBUG] Ep {epoch+1}: LR={current_lr:.6f} | "
                            f"Patience Count={scheduler.num_bad_epochs}/{scheduler.patience}")
                if epoch == 0 or current_lr != local_optimizer.param_groups[0]['lr']:
                    logger.debug(f"         📉 LR: {current_lr:.4e}")
            else:
                logger.debug("   ⚙️LO SCHEDUELER NON ESISTE: Nessuna azione scheduler.")
        
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
            'final_acc': epoch_accs[-1] if epoch_accs else 0, # ✨ NEW
            'num_samples': num_samples,
            'local_epochs': self.local_epochs,
        }
        
        logger.info(
            f"✅ Client {self.client_id}: Training completato | "
            f"Final Loss: {local_metrics['final_loss']:.4f} | Acc: {local_metrics['final_acc']*100:.2f}%" # ✨ NEW
        )
        
        return {
            'local_weights': local_weights,
            'local_metrics': local_metrics,
            'num_samples': num_samples,
        }
