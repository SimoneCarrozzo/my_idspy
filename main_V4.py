import logging
import numpy as np
from numpy import ndarray
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import torch

from src.idspy.nn.losses.base import BaseLoss
from src.idspy.nn.models.base import BaseModel
from src.idspy.common.logging import setup_logging
from src.idspy.common.seeds import set_seeds

from src.idspy.core.state import State
from src.idspy.core.pipeline import (
    FitAwareObservablePipeline,
    ObservablePipeline,
    PipelineEvent,
)

from src.idspy.data.schema import Schema, ColumnRole
from src.idspy.data.tab_accessor import TabAccessor

from src.idspy.events.bus import EventBus
from src.idspy.events.events import only_id
from src.idspy.events.handlers.logging import Logger

from src.idspy.steps.io.saver import SaveData
from src.idspy.steps.io.loader import LoadData
from src.idspy.steps.builders.dataloader import BuildDataLoader
from src.idspy.steps.builders.dataset import BuildDataset
from src.idspy.steps.transforms.adjust import DropNulls
from src.idspy.steps.transforms.map import FrequencyMap, LabelMap
from src.idspy.steps.transforms.scale import StandardScale
from src.idspy.steps.transforms.split import (
    AssignSplitPartitions,
    StratifiedSplit,
    AssignSplitTarget,
)
from src.idspy.steps.model.training import TrainOneEpoch, TrainWithEarlyStopping
from src.idspy.steps.model.evaluating import ValidateOneEpoch, MakePredictions
from src.idspy.steps.metrics.classification import ClassificationMetrics, PlotMetrics

from src.idspy.nn.batch import default_collate, Batch
from src.idspy.nn.helpers import get_device
from src.idspy.nn.checkpoints import save_checkpoint, save_weights
from src.idspy.nn.models.classifier import TabularClassifier
from src.idspy.nn.losses.classification import ClassificationLoss


setup_logging() #inizializza il logging: cioè configura il modulo logging di Python
logger = logging.getLogger(__name__)    #crea un logger per questo modulo
set_seeds(42) #imposta il seed per la generazione casuale consentendo la riproducibilità

"""
Pipeline principale per il training di un Network Intrusion Detection System (NIDS)
con approccio federato e gestione di dati non-IID.

Flusso completo AGGIORNATO V4 CON REBUILDING DEL DATASET (11/12/2025):
===============
NUOVO STEP PRIMA DEL PREPROCESSING: 
0.SMART SAMPLING )

1. Definizione dello schema dei dati
2. Setup del sistema di eventi (EventBus)
3. Preprocessing dei dati (fit-aware pipeline)
4. Configurazione del modello e della loss
5. Training con Early Stopping
6. Visualizzazione delle metriche
"""
from imblearn.over_sampling import SMOTENC

import pandas as pd
from typing import Dict, Optional

import os
from pathlib import Path
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_CUDA_ALLOC_MAX_SPLIT_SIZE_MB'] = '128'  # Limita frammentazione

def main():
    # ═══════════════════════════════════════════════════════════════════
    # 0️⃣ CONFIGURAZIONE PATHS
    # ═══════════════════════════════════════════════════════════════════
    
    BASE_DIR = Path("c:/Users/simon/OneDrive/Documenti/TESI_UNI/SetUp")
    CSV_ORIGINAL = BASE_DIR / "dataset_v2/cic_2018_v2.csv"
    CSV_TRAIN_BALANCED = BASE_DIR / "dataset_v2/rebuilt/train_balanced.csv"
    CSV_TEST = BASE_DIR / "dataset_v2/rebuilt/test.csv"
    PARQUET_DIR = BASE_DIR / "dataset_processati/rebuilt"
    PARQUET_TRAIN = PARQUET_DIR / "train_processed.parquet"
    PARQUET_TEST = PARQUET_DIR / "test_processed.parquet"
    
    # ═══════════════════════════════════════════════════════════════════
    # 1️⃣ DEFINIZIONE DELLO SCHEMA
    # ═══════════════════════════════════════════════════════════════════
    schema = Schema()
    
    # Target: colonna che indica se il traffico è un attacco o no
    schema.add(["Attack"], ColumnRole.TARGET)
    
    # Feature numeriche: statistiche del traffico di rete
    schema.add(
        [
            "IN_BYTES",
            "IN_PKTS",
            "OUT_BYTES",
            "OUT_PKTS",
            "FLOW_DURATION_MILLISECONDS",
            "DURATION_IN",
            "DURATION_OUT",
            "MIN_TTL",
            "MAX_TTL",
            "LONGEST_FLOW_PKT",
            "SHORTEST_FLOW_PKT",
            "MIN_IP_PKT_LEN",
            "MAX_IP_PKT_LEN",
            "SRC_TO_DST_SECOND_BYTES",
            "DST_TO_SRC_SECOND_BYTES",
            "RETRANSMITTED_IN_BYTES",
            "RETRANSMITTED_IN_PKTS",
            "RETRANSMITTED_OUT_BYTES",
            "RETRANSMITTED_OUT_PKTS",
            "SRC_TO_DST_AVG_THROUGHPUT",
            "DST_TO_SRC_AVG_THROUGHPUT",
            "NUM_PKTS_UP_TO_128_BYTES",
            "NUM_PKTS_128_TO_256_BYTES",
            "NUM_PKTS_256_TO_512_BYTES",
            "NUM_PKTS_512_TO_1024_BYTES",
            "NUM_PKTS_1024_TO_1514_BYTES",
            "TCP_WIN_MAX_IN",
            "TCP_WIN_MAX_OUT",
            "DNS_TTL_ANSWER",
        ],
        ColumnRole.NUMERICAL,
    )
    
    # Feature categoriche: protocolli, porte, flag TCP/IP
    schema.add(
        [
            "L4_SRC_PORT",
            "L4_DST_PORT",
            "PROTOCOL",
            "L7_PROTO",
            "TCP_FLAGS",
            "CLIENT_TCP_FLAGS",
            "SERVER_TCP_FLAGS",
            "ICMP_TYPE",
            "ICMP_IPV4_TYPE",
            "DNS_QUERY_ID",
            "DNS_QUERY_TYPE",
        ],
        ColumnRole.CATEGORICAL,
    )
    
    # ═══════════════════════════════════════════════════════════════════
    # 2️⃣ SETUP EVENTBUS - Sistema di pubblicazione/sottoscrizione per la gestione degli eventi, che traccia cosa succede ad ogni step della pipeline
     # ═══════════════════════════════════════════════════════════════════
    bus = EventBus()
    bus.subscribe(callback=Logger(), event_type=PipelineEvent.BEFORE_STEP)
    
    # ═══════════════════════════════════════════════════════════════════
    # 3️⃣ PIPELINE FIT-AWARE - Trasformazioni che "imparano" dai dati
    # ═══════════════════════════════════════════════════════════════════
    fit_aware_pipeline = FitAwareObservablePipeline(
        steps=[
            StandardScale(),           
            FrequencyMap(max_levels=20),  
            LabelMap(),                
        ],
        bus=bus,
        name="fit_aware_pipeline",
    )

    # ═══════════════════════════════════════════════════════════════════
    # 5️⃣ PREPARAZIONE DATASET: SPLIT + SMOTE SU TRAIN
    # ═══════════════════════════════════════════════════════════════════

    rebuild_needed = (
        not CSV_TRAIN_BALANCED.exists() or 
        not CSV_TEST.exists()
    )

    if rebuild_needed:
        logger.info("\n" + "="*70)
        logger.info("✂️ FASE 0: SPLIT + SMOTE AUGMENTATION")
        logger.info("="*70)
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STEP 1: Carica base dataset (primi 10M righe)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        logger.info("\n📖 Caricamento base dataset (primi 10M righe)...")
        df_base = pd.read_csv(CSV_ORIGINAL, nrows=10_000_000)
        
        logger.info(f"✅ Caricati {len(df_base):,} samples base")
        logger.info(f"📊 Distribuzione base:")
        for cls, count in df_base["Attack"].value_counts().items():
            logger.info(f"   {cls}: {count:,}")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STEP 2: Split 70/30 PRIMA di SMOTE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        logger.info("\n✂️ SPLIT STRATIFICATO 70/30 (prima di SMOTE)")
        
        from sklearn.model_selection import train_test_split
        
        train_idx, test_idx = train_test_split(
            df_base.index,
            test_size=0.30,
            stratify=df_base["Attack"],
            random_state=42
        )
        
        df_train_raw = df_base.iloc[train_idx].reset_index(drop=True)
        df_test_raw = df_base.iloc[test_idx].reset_index(drop=True)
        
        logger.info(f"   ✅ Train (raw): {len(df_train_raw):,} samples")
        logger.info(f"   ✅ Test (raw): {len(df_test_raw):,} samples")
        
        logger.info("\n📊 Distribuzione train raw:")
        for cls, count in df_train_raw["Attack"].value_counts().items():
            logger.info(f"   {cls}: {count:,}")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STEP 3: Applica SMOTE-NC SOLO su train
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.over_sampling import SMOTE
        from sklearn.preprocessing import LabelEncoder
        import psutil
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024**3  # GB
        logger.info(f"\n🧬 Applicazione SMOTE Moderato (Hybrid Strategy)...")
        logger.info(f"📊 RAM usata prima SMOTE: {mem_before:.2f} GB")
        logger.info("\n🧬 Applicazione SMOTE Standard (solo numeriche)...")
        
        class_counts = df_train_raw["Attack"].value_counts().to_dict()
        # 🎯 STRATEGIA BILANCIATA:
        # - Benign: 7M (mantenuto)
        # - Attacchi totali: 1M (distribuiti equamente tra 14 classi → ~71k per classe)
        # - Minoritarie estreme: portate a 10k (limite minimo statisticamente robusto)
        
        sampling_strategy_selective = {
            "DDOS attack-LOIC-UDP": 71_000,      # 770 → 71k (overlap con Benign, serve più boost)
            "SQL Injection": 5_000,              # 161 → 5k (troppo rara, target minimo)
            # "SSH-Bruteforce": 5_000,             # 35k → 5k (ATTENZIONE: già a 35k, ma tu vuoi ridurre?)
            "FTP-BruteForce": 15_000,            # 9.7k → 10k (quasi sufficiente, piccolo boost)
        }

        logger.info("📊 Strategia SMOTE Selettiva:")
        for cls, target in sampling_strategy_selective.items():
            current = class_counts.get(cls, 0)
            ratio = target / current if current > 0 else 0
            logger.info(f"   {cls}: {current:,} → {target:,} (x{ratio:.1f})")

        # Verifica k_neighbors
        min_class_size = min([class_counts[cls] for cls in sampling_strategy_selective.keys()])
        k_neighbors = 5
        if min_class_size <= k_neighbors:
            k_neighbors = max(1, min_class_size - 1)
            logger.warning(f"⚠️  Ridotto k_neighbors a {k_neighbors}")

        # Separa features e target
        X_train = df_train_raw.drop(columns=["Attack"])
        y_train = df_train_raw["Attack"]

        # Identifica colonne categoriche
        cat_cols = [
            "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO", "TCP_FLAGS",
            "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS", "ICMP_TYPE", "ICMP_IPV4_TYPE",
            "DNS_QUERY_ID", "DNS_QUERY_TYPE"
        ]

        # Salva categoriche
        cat_backup = X_train[cat_cols].copy()

        # Rimuovi categoriche
        X_train_numerical = X_train.drop(columns=cat_cols)

        # Converti a float64
        for col in X_train_numerical.columns:
            X_train_numerical[col] = pd.to_numeric(X_train_numerical[col], errors='coerce')
        X_train_numerical = X_train_numerical.fillna(0.0)

        logger.info(f"✅ Feature numeriche: {X_train_numerical.shape[1]} colonne")

        # SMOTE
        smote = SMOTE(
            sampling_strategy=sampling_strategy_selective,
            k_neighbors=k_neighbors,
            random_state=42,
        )

        logger.info(f"🔄 Esecuzione SMOTE (k_neighbors={k_neighbors})...")
        X_num_resampled, y_resampled = smote.fit_resample(X_train_numerical, y_train)

        logger.info(f"✅ SMOTE completato: {len(X_num_resampled):,} samples")    
            
        """ sampling_strategy_hybrid = {
            "SQL Injection": 3_220,              # x20
            "Brute Force -XSS": 6_860,           # x20  
            "DDOS attack-LOIC-UDP": 15_400,      # x20
            "Brute Force -Web": 16_440,          # x20
            "DoS attacks-Slowloris": 34_980,     # x10
            "DoS attacks-SlowHTTPTest": 52_210,  # x10
            "FTP-BruteForce": 96_830,            # x10
            "DoS attacks-GoldenEye": 103_070,    # x10
        }
        
        logger.info("📊 Strategia SMOTE (solo numeriche):")
        for cls, target in sampling_strategy_hybrid.items():
            current = class_counts.get(cls, 0)
            ratio = target / current if current > 0 else 0
            logger.info(f"   {cls}: {current:,} → {target:,} (x{ratio:.1f})")
        
        # Verifica k_neighbors
        min_class_size = min([class_counts[cls] for cls in sampling_strategy_hybrid.keys()])
        k_neighbors = 5
        if min_class_size <= k_neighbors:
            k_neighbors = max(1, min_class_size - 1)
            logger.warning(f"⚠️  Ridotto k_neighbors a {k_neighbors}")
        
        # Separa features e target
        X_train = df_train_raw.drop(columns=["Attack"])
        y_train = df_train_raw["Attack"]
        
        # Identifica colonne categoriche
        cat_cols = [
            "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO", "TCP_FLAGS",
            "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS", "ICMP_TYPE", "ICMP_IPV4_TYPE",
            "DNS_QUERY_ID", "DNS_QUERY_TYPE"
        ]
        
        # Salva categoriche PRIMA di SMOTE (saranno duplicate, non interpolate)
        cat_backup = X_train[cat_cols].copy()
        
        # Rimuovi temporaneamente categoriche
        X_train_numerical = X_train.drop(columns=cat_cols)
        
        # Converti numeriche a float64 (SMOTE preferisce float64)
        for col in X_train_numerical.columns:
            X_train_numerical[col] = pd.to_numeric(X_train_numerical[col], errors='coerce')
        X_train_numerical = X_train_numerical.fillna(0.0)
        
        logger.info(f"✅ Feature numeriche: {X_train_numerical.shape[1]} colonne")
        logger.info(f"✅ Feature categoriche (backup): {len(cat_cols)} colonne")
        
        # Applica SMOTE SOLO su numeriche
        smote = SMOTE(
            sampling_strategy=sampling_strategy_hybrid,
            k_neighbors=k_neighbors,
            random_state=42,
            #n_jobs=-1  # Usa tutti i core
        )
        
        logger.info(f"🔄 Esecuzione SMOTE (k_neighbors={k_neighbors}, solo numeriche)...")
        X_num_resampled, y_resampled = smote.fit_resample(X_train_numerical, y_train)
        
        logger.info(f"✅ SMOTE completato: {len(X_num_resampled):,} samples") """
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Ricrea categoriche per i nuovi samples
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        logger.info("🔧 Ricostruzione categoriche (duplicate per samples sintetici)...")
        
        # Per ogni nuovo sample, duplica le categoriche del sample originale più vicino
        # SMOTE ci dice quali samples originali ha usato per interpolare
        
        # Strategia semplice: usa le categoriche più comuni per ogni classe
        from scipy.stats import mode
        
        cat_resampled = pd.DataFrame(index=range(len(X_num_resampled)), columns=cat_cols)
        
        for cls in y_resampled.unique():
            # Indici nuovi samples per questa classe
            new_indices = np.where(y_resampled == cls)[0]
            
            # Indici vecchi samples per questa classe
            old_indices = np.where(y_train == cls)[0]
            
            if len(old_indices) == 0:
                continue
            
            # Per ogni colonna categorica, usa la moda (valore più comune)
            for col in cat_cols:
                mode_result = mode(cat_backup.iloc[old_indices][col], keepdims=True)
                most_common_value = mode_result.mode[0]
                cat_resampled.loc[new_indices, col] = most_common_value
        
        logger.info(f"✅ Categoriche ricostruite")
        
        # Ricombina numeriche + categoriche
        df_train_augmented = pd.DataFrame(X_num_resampled, columns=X_train_numerical.columns)
        df_train_augmented[cat_cols] = cat_resampled[cat_cols]
        df_train_augmented["Attack"] = y_resampled
        
        logger.info(f"✅ Train finale con SMOTE: {len(df_train_augmented):,} samples")
        
        mem_after = process.memory_info().rss / 1024**3  # GB
        mem_delta = mem_after - mem_before
        logger.info(f"📊 RAM usata dopo SMOTE: {mem_after:.2f} GB (+{mem_delta:.2f} GB)")

        logger.info(f"✅ Train con SMOTE: {len(df_train_augmented):,} samples")
        
        """ # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STEP 4: Bilanciamento finale (limita Benign)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        MAX_SAMPLES_PER_CLASS = {
        #     "Benign": 2_000_000,
        }
        
        logger.info("\n🔧 Bilanciamento finale (limiti per classe)...")
        dfs_balanced = []
        
        for cls in df_train_augmented["Attack"].unique():
            cls_data = df_train_augmented[df_train_augmented["Attack"] == cls]
            max_allowed = MAX_SAMPLES_PER_CLASS.get(cls, len(cls_data))
            
            if len(cls_data) > max_allowed:
                cls_data = cls_data.sample(n=max_allowed, random_state=42)
                logger.info(f"   ⚠️  {cls}: ridotto a {max_allowed:,}")
            
            dfs_balanced.append(cls_data)
        
        df_train_final = pd.concat(dfs_balanced, ignore_index=True)
        df_train_final = df_train_final.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        logger.info(f"✅ Train finale: {len(df_train_final):,} samples") """
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STEP 4: UNDERSAMPLING classi attacco troppo numerose
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        logger.info("\n🔧 Bilanciamento finale con UNDERSAMPLING...")

        MAX_SAMPLES_PER_CLASS = {
            "Benign": 5_000_000,              # Limita dominanza
            
            # Classi grandi: riduci leggermente per bilanciare
            "DDOS attack-HOIC": 300_000,      # 400k → 300k
            "DoS attacks-Hulk": 120_000,      # 160k → 120k
            "DDoS attacks-LOIC-HTTP": 90_000, # 114k → 90k
            
            # Classi medie: mantieni originali
            "Bot": 53_223,                    # No modifica
            "Infilteration": 42_973,          # No modifica
            "SSH-Bruteforce": 35_314,         # No modifica (o 5k se vuoi ridurre)
            "DoS attacks-GoldenEye": 10_307,  # No modifica
            
            # Classi piccole: già aumentate da SMOTE
            # (non serve specificare, SMOTE le ha portate al target)
        }

        dfs_balanced = []

        for cls in df_train_augmented["Attack"].unique():
            cls_data = df_train_augmented[df_train_augmented["Attack"] == cls]
            max_allowed = MAX_SAMPLES_PER_CLASS.get(cls, len(cls_data))
            
            if len(cls_data) > max_allowed:
                cls_data = cls_data.sample(n=max_allowed, random_state=42)
                logger.info(f"   ⚠️  {cls}: ridotto da {len(df_train_augmented[df_train_augmented['Attack'] == cls]):,} a {max_allowed:,}")
            
            dfs_balanced.append(cls_data)

        df_train_final = pd.concat(dfs_balanced, ignore_index=True)
        df_train_final = df_train_final.sample(frac=1.0, random_state=42).reset_index(drop=True)

        logger.info(f"✅ Train finale: {len(df_train_final):,} samples")
        # Test rimane invariato (nessun SMOTE)
        df_test_final = df_test_raw
        
        logger.info(f"✅ Test finale: {len(df_test_final):,} samples")
        
        # Report finale
        logger.info("\n📊 Distribuzione finale train:")
        final_counts = df_train_final["Attack"].value_counts()
        max_final = final_counts.max()
        
        for cls, count in sorted(final_counts.items(), key=lambda x: x[1], reverse=True):
            ratio = max_final / count if count > 0 else float('inf')
            pct = (count / len(df_train_final)) * 100
            original = class_counts.get(cls, 0)
            added = count - original
            logger.info(
                f"   {cls}: {count:,} ({pct:.2f}%) - rapporto={ratio:.1f}:1 "
                f"[+{added:,} sintetici]"
            )
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STEP 5: Salvataggio
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        logger.info("\n💾 Salvataggio...")
        df_train_final.to_csv(CSV_TRAIN_BALANCED, index=False)
        df_test_final.to_csv(CSV_TEST, index=False)
        
        logger.info(f"   ✅ Train → {CSV_TRAIN_BALANCED}")
        logger.info(f"   ✅ Test → {CSV_TEST}")
        logger.info("="*70 + "\n")
            


    # ═══════════════════════════════════════════════════════════════════
    # 5️⃣CONFIGURAZIONE MODELLO E LOSS
    # ═══════════════════════════════════════════════════════════════════
    device = torch.device("cuda")  
    model = TabularClassifier(
        num_features=len(schema.numerical),
        cat_cardinalities=[20] * len(schema.categorical),
        num_classes=15,
        hidden_dims=[256, 128, 64],
        # hidden_dims=[128, 64],
        dropout=0.2, #0.3, #0.2,
    ).to(device)

    # Creo loss INIZIALE (senza class weighting)
    loss_fn = ClassificationLoss().to(device)
         
    # MODIFICA DEL 11/12
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # ← CAMBIATO da 0.00025 a 0.0003
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.00025)  # ← CAMBIATO da 0.00025 a 0.0003
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.00012)  # ← CAMBIATO da 0.00025 a 0.0003
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)  # ← CAMBIATO da 0.00025 a 0.0003
    
    #MODIFICA DEL 11/12
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.7,#0.7,              # ← CAMBIATO da 0.5 a 0.7 (riduzione più graduale)
        patience=4, #4,              # ← CAMBIATO da 2 a 4 (più paziente)
        min_lr=1e-6#1e-6,
    )
    
    # ═══════════════════════════════════════════════════════════════════
    # 6️⃣ CREAZIONE DELLO STATE INIZIALE
    # ═══════════════════════════════════════════════════════════════════
    # Lo State è il "contenitore globale" che passa tra tutti gli step
    state = State(
        {
            "device": device,
            "model": model,
            "loss": loss_fn,
            "optimizer": optimizer,
            "seed": 42,
        }
    )

    
    
    # ═══════════════════════════════════════════════════════════════════
    # 7️⃣ PIPELINE PREPROCESSING - Train e Test separati
    # ═══════════════════════════════════════════════════════════════════
    
    # Verifica se i parquet esistono
    preprocessing_needed = (
        not PARQUET_TRAIN.exists() or 
        not PARQUET_TEST.exists()
    )
    
    if preprocessing_needed:
        logger.info("\n" + "="*70)
        logger.info("⚙️ FASE 1: PREPROCESSING DATI")
        logger.info("="*70)
        
        # Pipeline per TRAIN (FIT + TRANSFORM)
        logger.info("\n🔧 Preprocessing TRAIN (fit + transform)...")
        preprocessing_train = ObservablePipeline(
            steps=[
                LoadData(
                    path_in=CSV_TRAIN_BALANCED,
                    schema=schema,
                    nrows=None,
                    in_scope="data",
                    out_scope="data",
                ),
                DropNulls(),
                fit_aware_pipeline,  # ← FIT qui
                SaveData(
                    file_path=PARQUET_DIR,
                    file_name="train_processed",
                    fmt="parquet",
                ),
            ],
            bus=bus,
            name="preprocessing_train",
        )
        
        preprocessing_train.run(state)
        logger.info(f"✅ Train processato salvato in: {PARQUET_TRAIN}\n")
        
        # Pipeline per TEST (SOLO TRANSFORM, no refit)
        logger.info("🔧 Preprocessing TEST (solo transform)...")
        preprocessing_test = ObservablePipeline(
            steps=[
                LoadData(
                    path_in=CSV_TEST,
                    schema=schema,
                    nrows=None,
                    in_scope="data",
                    out_scope="data",
                ),
                DropNulls(),
                fit_aware_pipeline,  # ← Usa trasformazioni già fittate
                SaveData(
                    file_path=PARQUET_DIR,
                    file_name="test_processed",
                    fmt="parquet",
                ),
            ],
            bus=bus,
            name="preprocessing_test",
        )
        
        preprocessing_test.run(state)
        logger.info(f"✅ Test processato salvato in: {PARQUET_TEST}\n")
        
        logger.info("="*70 + "\n")
    else:
        logger.info("\n" + "="*70)
        logger.info("✅ Parquet processati già esistenti, skip preprocessing")
        logger.info("="*70 + "\n")
    
       
    
    # ═══════════════════════════════════════════════════════════════════
    # 9️⃣ COSTRUZIONE DATASET E DATALOADER
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*70)
    logger.info("🔧 FASE 2: SETUP DATASET E DATALOADER")
    logger.info("="*70)
    
    setup_pipeline = ObservablePipeline(
        steps=[
            # Carica train processato
            LoadData(
                path_in=PARQUET_TRAIN,
                schema=schema,
                in_scope="data",
                out_scope="train",  # ← Metti direttamente in train
            ),
            # Carica test processato
            LoadData(
                path_in=PARQUET_TEST,
                schema=schema,
                in_scope="data",
                out_scope="test",  # ← Metti direttamente in test
            ),            
            # Assegna targets
            AssignSplitTarget(in_scope="train", out_scope="train"),
            AssignSplitTarget(in_scope="test", out_scope="test"),
            
            # Build datasets
            BuildDataset(in_scope="train", out_scope="train"),
            BuildDataLoader(
                in_scope="train",
                out_scope="train",
                batch_size=512,#1024,#512,
                num_workers=2,
                pin_memory=True,
                shuffle=True,
                collate_fn=default_collate,
            ),
            BuildDataset(in_scope="test", out_scope="test"),
            BuildDataLoader(
                in_scope="test",
                out_scope="test",
                batch_size=1024,
                shuffle=False,
                collate_fn=default_collate,
            )
        ],
        bus=bus,
        name="setup_pipeline",
    )
    
    setup_pipeline.run(state)
    
    # ═══════════════════════════════════════════════════════════════════
    # 📋 ESTRAZIONE NOMI CLASSI
    # ═══════════════════════════════════════════════════════════════════
    # Get the actual class names from LabelMap mapping
    logger.info("\n" + "="*70)
    logger.info("📋 ESTRAZIONE NOMI CLASSI")
    logger.info("="*70)
    
    # Leggi i nomi delle classi direttamente dal CSV raw
    csv_path = "c:/Users/simon/OneDrive/Documenti/TESI_UNI/SetUp/dataset_v2/cic_2018_v2.csv"
   
    try:
        # 🆕 Leggi SOLO la colonna "Attack" usando memory-efficient approach
        # Non caricare tutto il CSV in memoria, solo la colonna target
        attack_classes = pd.read_csv(
            csv_path,
            usecols=["Attack"],  # ← Leggi SOLO questa colonna
            dtype={"Attack": "category"}  # ← Usa dtype 'category' per risparmiare memoria
        )
        
        # Estrai le classi uniche e ordinate
        original_classes = sorted(attack_classes["Attack"].unique())
        class_names = [str(c) for c in original_classes]
        
        logger.info(f"✅ Class names from CSV (Attack column only): {class_names}")
        
    except Exception as e:
        logger.warning(f"⚠️ Impossibile leggere da CSV: {e}")
        logger.info("   Fallback: estrai dai train.targets già caricati")
        
        try:
            # Fallback: estrai dai targets caricati dallo State
            train_targets = state.get("train.targets", np.ndarray)
            unique_classes = np.unique(train_targets)
            num_classes = len(unique_classes)
            
            # Crea nomi standard
            class_names = [f"Attack_Type_{i}" for i in range(num_classes)]
            
            logger.info(f"✅ Class names (from train.targets): {class_names}")
            
        except Exception as e2:
            logger.error(f"❌ Impossibile estrarre class names: {e2}")
            num_classes = 15  # Dal tuo modello
            class_names = [f"Class_{i}" for i in range(num_classes)]
            logger.warning(f"⚠️ Fallback finale: {class_names}")

    logger.info(f"📝 Nomi classi finali: {class_names}\n")
    
    # ═══════════════════════════════════════════════════════════════════
    # 🔟 CORREZIONE LABEL SHIFT (bug fix LabelMap)
    # ═══════════════════════════════════════════════════════════════════

    if state.has("train.targets") and state.has("test.targets"):
        train_targets = state.get("train.targets", np.ndarray)
        test_targets = state.get("test.targets", np.ndarray)
        
        logger.info(f"🔧 Train targets - Prima shift: min={train_targets.min()}, max={train_targets.max()}")
        logger.info(f"🔧 Test targets - Prima shift: min={test_targets.min()}, max={test_targets.max()}")
        
        # Applica shift -1
        train_targets = train_targets - 1
        test_targets = test_targets - 1
        
        # Aggiorna lo state
        state.set("train.targets", train_targets, np.ndarray)
        state.set("test.targets", test_targets, np.ndarray)
        
        logger.info(f"✅ Train targets - Dopo shift: min={train_targets.min()}, max={train_targets.max()}")
        logger.info(f"✅ Test targets - Dopo shift: min={test_targets.min()}, max={test_targets.max()}")
    
    
    # ═══════════════════════════════════════════════════════════════════
    #  1️⃣1️⃣ PIPELINE EPOCA - Operazioni per ogni singola epoca
    # ═══════════════════════════════════════════════════════════════════
    
    import os
    log_dir = "C:/Users/simon/OneDrive/Documenti/TESI_UNI/SetUp/Modelli_Salvati/logs/v5.8"    
    os.makedirs(log_dir, exist_ok=True)
    
    # 🆕 CALCOLA train_class_counts QUI (prima di epoch_pipeline)
    train_targets = state.get("train.targets", np.ndarray)
    train_class_counts = dict(zip(*np.unique(train_targets, return_counts=True)))
        
    epoch_pipeline = ObservablePipeline(
        steps=[
            TrainOneEpoch(
                log_dir=log_dir,           # AGGIUNTO
                log_prefix="train",        # AGGIUNTO
                save_history=True,         # Opzionale
                # checkpoint_dir=f"{log_dir}/checkpoints", #AGGIUNTO
            ),  
            ValidateOneEpoch(
                log_dir=log_dir,           # AGGIUNTO
                log_prefix="val",          # AGGIUNTO 
                in_scope="test",           # ← mantieni "test" se usi test set
                out_scope="test",
                save_outputs=True,
            ),
            MakePredictions(pred_fn=lambda x: torch.argmax(x, dim=1)),  # Converte output in predizioni
            ClassificationMetrics(
                log_dir=log_dir,                   # CAMBIO path
                log_prefix="test",                 # AGGIUNTO
                class_names=class_names,           # AGGIUNTO
                in_scope="test",
                out_scope="test",
                save_confusion_matrix=True,  # ✅ Salva metriche
                save_classification_report=True,
                save_f1_per_class_plot = True,
                #Aggiunto
                train_class_counts=train_class_counts
            ),  # Calcola e salva metriche
        ],
        bus=bus,
        name="epoch_pipeline",
    )
    
    # ═══════════════════════════════════════════════════════════════════
    # 1️⃣2️⃣ CALCOLO CLASS WEIGHTS (gestione dataset sbilanciato)
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*70)
    logger.info("⚖️ FASE 3: CALCOLO CLASS WEIGHTS PER DATASET SBILANCIATO")
    logger.info("="*70)
    
    
    train_targets = state.get("train.targets", np.ndarray)
    unique_classes, class_counts = np.unique(train_targets, return_counts=True)
    
    # Stampa distribuzione classi
    logger.info("\n📊 Distribuzione classi nel training set:")
    total_samples = len(train_targets)
    for cls, count in zip(unique_classes, class_counts):
        percentage = (count / total_samples) * 100
        logger.info(f"   Classe {cls:2d}: {count:10,} samples ({percentage:5.2f}%)")
    
    # 🔧 STRATEGIA HYBRID: Class weights PIÙ FORTI per compensare SMOTE moderato
    smoothing_strategy = "sqrt"  # Exponent 0.55 (tra sqrt=0.5 e 4_root=0.25)
    
    # Calcola i pesi bilanciati automaticamente
    class_weights_balanced = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=train_targets
    )
        
    # Applica smoothing HYBRID (più aggressivo per compensare SMOTE ridotto)
    if smoothing_strategy == "hybrid_65":
        logger.info("\n✅ STRATEGIA HYBRID: Class Weighting con exponent 0.55")
        logger.info("   (Più forte di sqrt per compensare SMOTE moderato)")
        class_weights_smoothed = np.power(class_weights_balanced, 0.65)
    elif smoothing_strategy == "sqrt":
        logger.info("\n✅ STRATEGIA: Smoothed Class Weighting (radice quadrata)")
        class_weights_smoothed = np.sqrt(class_weights_balanced)
    elif smoothing_strategy == "4_root":
        logger.info("\n✅ STRATEGIA: Ultra-Smoothed Class Weighting (radice quarta)")
        class_weights_smoothed = np.power(class_weights_balanced, 0.25)
        #ATTENZIONE è NUOVO QUELLO CHE SEGUE:
    elif smoothing_strategy == "midway_root":
        logger.info("\n✅ STRATEGIA: MIDWAY-Smoothed Class Weighting (radice pari a 0.40)")
        class_weights_smoothed = np.power(class_weights_balanced, 0.40)
    else:
        raise ValueError(f"Strategia sconosciuta: {smoothing_strategy}")

    # Stampa confronto pesi
    logger.info("\n⚖️ Confronto pesi delle classi:")
    for cls in unique_classes:
        logger.info(
            f"   Classe {cls:2d}: balanced={class_weights_balanced[cls]:.4f}, "
            f"smoothed={class_weights_smoothed[cls]:.4f}"
        )
    # Converti in tensor PyTorch
    class_weights_tensor = torch.FloatTensor(class_weights_smoothed).to(device)
    logger.info(f"\n✅ Class weights tensor shape: {class_weights_tensor.shape}")
    
    
    # Aggiorna la loss con i pesi
    # loss_fn = ClassificationLoss().to(device)  # ← NO class_weight parameter
    loss_fn = ClassificationLoss(class_weight=class_weights_tensor).to(device)  # ← NO class_weight parameter
    state.set("loss", loss_fn, ClassificationLoss)
    # logger.info("🎯 Loss function: CrossEntropy standard (no weights)\n")
    logger.info(f"🎯 Loss function configurata: {smoothing_strategy}\n")

    
    # ═══════════════════════════════════════════════════════════════════
    # 1️⃣3️⃣ TRAINING CON EARLY STOPPING 
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*70)
    logger.info("🚀 FASE 4: TRAINING CON EARLY STOPPING")
    logger.info("="*70)
        
    #MODIFICA DEL 22/11
    training_step = TrainWithEarlyStopping(
        epoch_pipeline=epoch_pipeline,
        num_epochs=50,              # OK
        patience=8,                 # ← CAMBIATO da 5 a 8 (più paziente)
        min_delta=0.0002,#0.0002,           # ← CAMBIATO da 0.0005 a 0.0002 (più tollerante)
        checkpoint_dir=f"{log_dir}/checkpoints_ES",
        save_best_only=True,
        verbose=True,
        in_scope="train",      
        out_scope="train",     
        name="train_with_early_stopping",
        scheduler=scheduler
    )
    
    # 🆕 Esegui il training 
    training_step.run(state)
    
    # ═══════════════════════════════════════════════════════════════════
    # 1️⃣4️⃣ VISUALIZZAZIONE METRICHE CON MATPLOTLIB 
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*70)
    logger.info("📊 FASE 5: GENERAZIONE GRAFICI METRICHE")
    logger.info("="*70)
    
    # 🆕 Crea lo step per la visualizzazione
    plot_step = PlotMetrics(
        metrics_dir=log_dir,
        output_dir=f"{log_dir}/plots",
        metrics_to_plot=["accuracy", "precision", "recall", "f1_micro", "f1_macro", "f1_weighted"],
        figsize=(12, 8),
        dpi=300,                    
        show_plots=False,           # Non mostra interattivamente, solo salva
        in_scope="test",       
        out_scope="plots",     
        name="plot_metrics"
    )
    
    # 🆕 Genera i grafici
    plot_step.run(state)
    
    logger.info("\n" + "="*70)
    logger.info("✅ GRAFICI SALVATI")
    logger.info("="*70)
    
    # ═══════════════════════════════════════════════════════════════════
    # 1️⃣5️⃣ FINE!
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*70)
    logger.info("🎉 PIPELINE COMPLETA ESEGUITA CON SUCCESSO!")
    logger.info("="*70)

   
if __name__ == "__main__":
    main()

   