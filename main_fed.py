import logging
from pathlib import Path
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
from src.idspy.nn.losses.classification import ClassificationLoss, FocalLoss

# from src.idspy.steps.io.federated_saver import SaveFederatedData
from src.idspy.federated_learning.steps.io.fed_saver import SaveFederatedData

# from src.idspy.core.federated_split import(
#     IdentifyTopHosts,
#     SplitByHosts,
#     FederatedSplits,
#     AggregatedTestSet,
#     AnalyzeNonIID,
#     ApplyFitAwareToFederatedSplits
# )
from src.idspy.federated_learning.core.fed_split import(
    IdentifyTopHosts,
    SplitByHosts,
    FederatedSplits,
    AggregatedTestSet,
    AnalyzeNonIID,
    ApplyFitAwareToFederatedSplits
)

from src.idspy.federated_learning.orchestrator import FederatedTrainingRound
from src.idspy.federated_learning.steps.model.evaluator import EvaluateGlobalModel
from src.idspy.federated_learning.steps.transform.fed_scale import (
    ComputeGlobalNormalizationStats,
    ApplyGlobalNormalization,
)
from src.idspy.federated_learning.steps.transform.fed_map import (
    FrequencyMapGlobal,
    LabelMapGlobal
)

setup_logging() #inizializza il logging: cioè configura il modulo logging di Python
logger = logging.getLogger(__name__)    #crea un logger per questo modulo
set_seeds(42) #imposta il seed per la generazione casuale consentendo la riproducibilità

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_CUDA_ALLOC_MAX_SPLIT_SIZE_MB'] = '128'  # Limita frammentazione

def main():
    """
    Main federato completo con:
    - Fase 1: Preprocessing e generazione dataset federati
    - Fase 2: Training federato con aggregazione pesi
    """
    
    # ═══════════════════════════════════════════════════════════════════
    # 0️⃣ CONFIGURAZIONE GENERALE
    # ═══════════════════════════════════════════════════════════════════
    
    # 🎛️ Flag di controllo: cosa vuoi eseguire?
    RUN_PREPROCESSING = False  # True solo la PRIMA volta per generare i dati
    RUN_TRAINING = True       # True per fare il training federato
    
    # Percorsi
    RAW_DATA_PATH = "c:/Users/simon/OneDrive/Documenti/TESI_UNI/SetUp/dataset_v2/cic_2018_v2.csv"
    FEDERATED_DATA_PATH = "c:/Users/simon/OneDrive/Documenti/TESI_UNI/SetUp/dataset_processati_federati/log2"
    LOGS_PATH = "c:/Users/simon/OneDrive/Documenti/TESI_UNI/SetUp/Modelli_Salvati/logs_federated"
    
    # Parametri FL
    NUM_ROUNDS = 10           # Numero di round federati
    LOCAL_EPOCHS = 3          # Epoche locali per client
    CLIENT_FRACTION = 1.0     # Frazione client per round (1.0 = tutti)
    LEARNING_RATE = 0.001     # Learning rate per ottimizzatore
    
    # ═══════════════════════════════════════════════════════════════════
    # 1️⃣ DEFINIZIONE DELLO SCHEMA
    # ═══════════════════════════════════════════════════════════════════
    schema = Schema()
    schema.add(["Attack"], ColumnRole.TARGET)
    schema.add(
        [
            "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
            "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
            "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
            "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
            "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
            "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
            "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
            "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
            "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
            "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
            "NUM_PKTS_1024_TO_1514_BYTES",
            "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "DNS_TTL_ANSWER",
        ],
        ColumnRole.NUMERICAL,
    )
    schema.add(
        [
            "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO",
            "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
            "ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_ID", "DNS_QUERY_TYPE",
        ],
        ColumnRole.CATEGORICAL,
    )
    
    # ═══════════════════════════════════════════════════════════════════
    # 2️⃣ SETUP EVENTBUS
    # ═══════════════════════════════════════════════════════════════════
    bus = EventBus()
    bus.subscribe(callback=Logger(), event_type=PipelineEvent.BEFORE_STEP)
    
    # ═══════════════════════════════════════════════════════════════════
    # 3️⃣ STATO GLOBALE
    # ═══════════════════════════════════════════════════════════════════
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = State({
        "device": device,
        "seed": 42,
        "bus": bus
    })
    
    # ═══════════════════════════════════════════════════════════════════
    # 🔄 FASE 1: PREPROCESSING FEDERATO (Esegui 1 volta)
    # ═══════════════════════════════════════════════════════════════════
    if RUN_PREPROCESSING:
        logger.info("\n" + "="*70)
        logger.info("📡 FASE 1: GENERAZIONE DATASET FEDERATI (NON-IID)")
        logger.info("="*70)
        
        # Fit-aware steps (da applicare PER HOST)
        # fit_aware_steps = [
        #     # FrequencyMap(max_levels=20),
        #     # LabelMap1(),
        #     #StandardScale(),
        # ]
        fit_aware_pipeline = FitAwareObservablePipeline(
            steps=[
                #StandardScale(),           
                FrequencyMapGlobal(max_levels=20, in_scope="data",
                    out_scope="data"),  
                LabelMapGlobal(benign_tag="Benign", in_scope="data",
                    out_scope="data"), 
                # B. NORMALIZZAZIONE GLOBALE PRIMA dello split!
                ComputeGlobalNormalizationStats(
                    in_scope="data",
                    out_scope="data"
                ),
                ApplyGlobalNormalization(
                    in_scope="data",
                    out_scope="data"
                ),               
            ],
            bus=bus,
            name="federated_fit_aware_pipeline",
        )
        preprocessing_pipeline = ObservablePipeline(
            steps=[
                # A. Caricamento Dati Raw
                LoadData(
                    path_in=RAW_DATA_PATH,
                    schema=schema,
                    nrows=10_000_000  # 🔥 100M righe = dataset completo
                ),
                DropNulls(),
                
                fit_aware_pipeline,
                
                                
                # C. Identificazione Top Hosts
                IdentifyTopHosts(
                    num_hosts=10,  # 🎯 Prendi i 10 host con più traffico
                    src_ip_col='IPV4_SRC_ADDR',
                    dst_ip_col='IPV4_DST_ADDR',
                    attack_col='Attack',
                    in_scope="data",
                    out_scope="federated"
                ),
                
                # D. Split Fisico per Host
                SplitByHosts(
                    min_samples_per_host=5_000,  
                    src_ip_col='IPV4_SRC_ADDR',
                    dst_ip_col='IPV4_DST_ADDR',
                    attack_col='Attack',
                    in_scope="data",
                    out_scope="federated"
                ),
                
                # E. Train/Val/Test Split per ogni Host
                FederatedSplits(
                    train_size=0.8,
                    val_size=0.10,
                    test_size=0.10,
                    stratify_column='Attack',
                    min_samples_per_class=10,
                    in_scope="federated",
                    out_scope="federated"
                ),
                
                # F. Analisi Non-IID
                AnalyzeNonIID(
                    label_col='Attack',
                    benign_label='Benign', #aggiunto
                    attack_type_col='Label',
                    debug=True,#False,
                    in_scope="federated",
                    out_scope="federated"
                ),
                
                # G. Trasformazioni Fit-Aware (per ogni host)
                # ApplyFitAwareToFederatedSplits(
                #     fit_aware_steps=fit_aware_steps,
                #     use_shared_bus=True,
                #     in_scope="federated",
                #     out_scope="federated",
                # ),
                
                # H. Test Set Aggregato Globale
                AggregatedTestSet(
                    attack_col='Attack',
                    save_host_labels=True,
                    in_scope="federated",
                    out_scope="federated"
                ),
                
                # I. Salvataggio
                SaveFederatedData(
                    base_path=FEDERATED_DATA_PATH,
                    fmt="parquet",
                    save_meta=True,
                    save_statistics=True,  # 🆕
                    save_other_stats=True,  # 🆕
                ),
            ],
            bus=bus,
            name="federated_preprocessing_pipeline",
        )
        
        try:
            preprocessing_pipeline.run(state)
            
            logger.info("\n" + "="*70)
            logger.info("🎉 PREPROCESSING FEDERATO COMPLETATO!")
            logger.info("="*70)
            
            # Log metriche Non-IID
            if state.has("federated.non_iid_metrics"):
                metrics = state.get("federated.non_iid_metrics", dict)
                logger.info(f"📊 Metriche Non-IID:")
                for key, value in metrics.items():
                    logger.info(f"   • {key}: {value}")
            
            logger.info(f"\n💾 Dati salvati in: {FEDERATED_DATA_PATH}")
            logger.info("✅ Puoi ora eseguire RUN_TRAINING=True")
            
        except Exception as e:
            logger.error(f"❌ Errore nella pipeline di preprocessing: {e}", exc_info=True)
            return
    
    # ═══════════════════════════════════════════════════════════════════
    # 🚀 FASE 2: TRAINING FEDERATO (Esegui dopo preprocessing)
    # ═══════════════════════════════════════════════════════════════════
    if RUN_TRAINING:
        logger.info("\n" + "="*70)
        logger.info("🚀 FASE 2: TRAINING FEDERATO")
        logger.info("="*70)
        
        # ───────────────────────────────────────────────────────────────
        # 📥 CARICAMENTO DATI FEDERATI
        # ───────────────────────────────────────────────────────────────
        
        logger.info("📥 Caricamento dataset federati...")
        
        federated_data_dir = Path(FEDERATED_DATA_PATH)
        
        # Trova tutte le cartelle host_*
        host_dirs = [d for d in federated_data_dir.iterdir() if d.is_dir() and d.name.startswith("host_")]
        
        if len(host_dirs) == 0:
            logger.error(f"❌ Nessun host trovato in {federated_data_dir}")
            logger.error("   Devi prima eseguire RUN_PREPROCESSING=True!")
            return
        
        logger.info(f"✅ Trovati {len(host_dirs)} host")
        
        # Carica dati per ogni host
        federated_splits = {}
        
        for host_dir in host_dirs:
            host_id = host_dir.name  # es. "host_172_31_0_2"
            
            train_path = host_dir / "train.parquet"
            val_path = host_dir / "val.parquet"
            test_path = host_dir / "test.parquet"
            
            if not all([train_path.exists(), val_path.exists(), test_path.exists()]):
                logger.warning(f"⚠️ File mancanti per {host_id}, skip")
                continue
            
            ### cancellato federated_splits[host_id] = {...} e sostituito con quanto segue:
            
            ##################################################à
            # ⭐ CARICA E CONVERTI TUTTI I DATI IN NUMERICO
            train_df = pd.read_parquet(train_path)
            val_df = pd.read_parquet(val_path)
            test_df = pd.read_parquet(test_path)
            
            ########### ⭐ AGGIUNGI QUESTE RIGHE
            # Rimuovi colonne non necessarie per il training
            cols_to_drop = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'original_Attack', 'Label'] #aggiunto label prima non c'era
            train_df = train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns])
            val_df = val_df.drop(columns=[c for c in cols_to_drop if c in val_df.columns])
            test_df = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns])
            
            ################### 🔍 DEBUG: Stampa tipi di dato
            """ print(f"\n🔍 DEBUG {host_id}:")
            print(train_df.dtypes)
            object_cols = train_df.select_dtypes(include=['object']).columns.tolist()
            if object_cols:
                print(f"❌ Colonne problematiche: {object_cols}")
                for col in object_cols:
                    print(f"   {col}: {train_df[col].head()}") """
            
            ########### Forza conversione numerica per tutte le colonne tranne Attack
            for col in train_df.columns:
                if col != 'Attack':
                    train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)
                    val_df[col] = pd.to_numeric(val_df[col], errors='coerce').fillna(0)
                    test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0)
            
            # Assicurati che Attack sia int
            train_df['Attack'] = train_df['Attack'].astype(np.int64)
            val_df['Attack'] = val_df['Attack'].astype(np.int64)
            test_df['Attack'] = test_df['Attack'].astype(np.int64)
            
            # -----------------------------------------------------------
            # 🧹 RIMORZIONE LABEL NEGATIVI (-1)
            # -----------------------------------------------------------
            # Controlliamo se ci sono label negativi (es. -1) che fanno crashare la Loss
            for split_name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
                if (df['Attack'] < 0).any():
                    invalid_count = (df['Attack'] < 0).sum()
                    logger.warning(f"⚠️ {host_id} [{split_name}]: Trovati {invalid_count} sample con label negativo (-1)!")
                    
                    # Rimuoviamo le righe corrotte
                    logger.info("   🔧 Fix: Rimozione righe con label negativo")
                    rows_before = len(df)
                    # Manteniamo solo le righe con Attack >= 0
                    # Nota: dobbiamo sovrascrivere il DataFrame originale
                    df_clean = df[df['Attack'] >= 0].copy()
                    
                    # Sovrascriviamo le variabili locali per il dizionario successivo
                    if split_name == "Train": train_df = df_clean
                    elif split_name == "Val": val_df = df_clean
                    elif split_name == "Test": test_df = df_clean
                    
                    logger.info(f"   ✂️ Rimossi {rows_before - len(df_clean)} samples.")

            ###################################################
            # ⭐ AGGIUNGI: Controlla se train è vuoto DOPO la pulizia
            if len(train_df) == 0:
                logger.warning(f"⚠️ {host_id}: NESSUN CAMPIONE nel train dopo pulizia! Skip host.")
                continue  # Non aggiungere questo host a federated_splits
            
            ###################################################
            # Verifica finale che non ci siano label > 14 (num_classes - 1)
            # Se il modello ha 15 classi, il max label accettabile è 14.
            # Facciamo un controllo preventivo.
            max_label_found = train_df['Attack'].max()
            if max_label_found >= 15: 
                 logger.error(f"❌ {host_id}: Trovato label {max_label_found} che è >= num_classes (15)!")
                 # logghiamo errore.
            # -----------------------------------------------------------           
            
            federated_splits[host_id] = {
                'train': train_df,
                'val': val_df,
                'test': test_df,
            }
            ##################################################à
            logger.info(
                f"   ✅ {host_id}: train={len(federated_splits[host_id]['train'])}, "
                f"val={len(federated_splits[host_id]['val'])}, "
                f"test={len(federated_splits[host_id]['test'])}"
            )
        
        # Carica test set aggregato
        aggregated_test_path = federated_data_dir / "aggregated_test.parquet"
        if aggregated_test_path.exists():
            aggregated_test = pd.read_parquet(aggregated_test_path)
            logger.info(f"✅ Test aggregato: {len(aggregated_test)} samples")
        else:
            logger.warning("⚠️ Test aggregato non trovato")
            aggregated_test = None
        
        """ 
        logger.info("📥 Caricamento dataset federati...")
        
        federated_data_dir = Path(FEDERATED_DATA_PATH)
        
        # Trova tutte le cartelle host_*
        host_dirs = [d for d in federated_data_dir.iterdir() if d.is_dir() and d.name.startswith("host_")]
        
        if len(host_dirs) == 0:
            logger.error(f"❌ Nessun host trovato in {federated_data_dir}")
            logger.error("   Devi prima eseguire RUN_PREPROCESSING=True!")
            return
        
        logger.info(f"✅ Trovati {len(host_dirs)} host")
        
        # Carica dati per ogni host
        federated_splits = {}
        
        for host_dir in host_dirs:
            host_id = host_dir.name  # es. "host_172_31_0_2"
            
            train_path = host_dir / "train.parquet"
            val_path = host_dir / "val.parquet"
            test_path = host_dir / "test.parquet"
            
            if not all([train_path.exists(), val_path.exists(), test_path.exists()]):
                logger.warning(f"⚠️ File mancanti per {host_id}, skip")
                continue
            
            federated_splits[host_id] = {
                'train': pd.read_parquet(train_path),
                'val': pd.read_parquet(val_path),
                'test': pd.read_parquet(test_path),
            }
            
            logger.info(
                f"   ✅ {host_id}: train={len(federated_splits[host_id]['train'])}, "
                f"val={len(federated_splits[host_id]['val'])}, "
                f"test={len(federated_splits[host_id]['test'])}"
            )
        
        # Carica test set aggregato
        aggregated_test_path = federated_data_dir / "aggregated_test.parquet"
        if aggregated_test_path.exists():
            aggregated_test = pd.read_parquet(aggregated_test_path)
            logger.info(f"✅ Test aggregato: {len(aggregated_test)} samples")
        else:
            logger.warning("⚠️ Test aggregato non trovato")
            aggregated_test = None """
        
        # ───────────────────────────────────────────────────────────────
        # 🤖 CREAZIONE MODELLO GLOBALE
        # ───────────────────────────────────────────────────────────────
        
        """ logger.info("\n🤖 Creazione modello globale...")
        
        global_model = TabularClassifier(
            num_features=len(schema.numerical),
            cat_cardinalities=[20] * len(schema.categorical),
            num_classes=15,
            hidden_dims=[512, 256, 128],
            dropout=0.2,
        ).to(device)
        
        logger.info(f"✅ Modello creato: {sum(p.numel() for p in global_model.parameters()):,} parametri") """
        
        # ───────────────────────────────────────────────────────────────
        # 🤖 CREAZIONE MODELLO GLOBALE (DINAMICA)
        # ───────────────────────────────────────────────────────────────
        
        logger.info("\n🤖 Creazione modello globale...")
        
        # 1. Prendiamo un host a caso (il primo) per prendere le misure
        first_host_id = next(iter(federated_splits))
        sample_df = federated_splits[first_host_id]['train']
        
        # 2. Calcoliamo le colonne ESATTAMENTE come fa il Client
        # (Copia della logica del Client per garantire coerenza)
        real_numerical_cols = sample_df.select_dtypes(include=['float64', 'float32', 'float16']).columns.tolist()
        
        # Le categoriche sono tutto ciò che non è numerico e non è 'Attack' (o colonne escluse)
        # Nota: nel main hai già droppato IP e original_Attack, quindi restano solo features
        real_categorical_cols = [c for c in sample_df.columns 
                                 if c not in real_numerical_cols and c != 'Attack']
        
        num_real_features = len(real_numerical_cols)
        num_real_categorical = len(real_categorical_cols)
        
        logger.info(f"📏 Misure rilevate dai dati reali ({first_host_id}):")
        logger.info(f"   • Features Numeriche: {num_real_features} (Schema diceva: {len(schema.numerical)})")
        logger.info(f"   • Features Categoriche: {num_real_categorical} (Schema diceva: {len(schema.categorical)})")
        
        # Controllo di sicurezza
        if num_real_features != len(schema.numerical):
            logger.warning("⚠️ ATTENZIONE: Il numero di colonne numeriche nei dati differisce dallo Schema!")
            logger.warning(f"   Si userà il valore dei DATI ({num_real_features}) per evitare crash.")

        # 3. Creazione Modello con le misure REALI
        global_model = TabularClassifier(
            num_features=num_real_features,  # <--- Usa il valore calcolato dai dati
            cat_cardinalities=[20] * num_real_categorical, # <--- Anche qui, usa il conteggio reale
            num_classes=15, # O len(sample_df['Attack'].unique()) se vuoi essere dinamico
            hidden_dims=[512, 256, 128],
            dropout=0.2,
        ).to(device)
        
        logger.info(f"✅ Modello creato correttamente: Input Layer adattato a {num_real_features} numeriche + embeddings.")
        
        # ───────────────────────────────────────────────────────────────
        # 📊 LOSS E OPTIMIZER
        # ───────────────────────────────────────────────────────────────
        
        loss_fn = ClassificationLoss(#num_classes=2,
                                     label_smoothing=0.1,  # Rende il modello più robusto
                                     class_weight=None,      # O calcola class weight se serve
                                    ).to(device)
        optimizer_class = torch.optim.Adam
        optimizer_kwargs = {
            'lr': LEARNING_RATE,
            'weight_decay': 1e-5
        }
        
        # ───────────────────────────────────────────────────────────────
        # 🔄 TRAINING LOOP FEDERATO
        # ───────────────────────────────────────────────────────────────
        log_dir = Path(LOGS_PATH)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🔄 INIZIO TRAINING FEDERATO ({NUM_ROUNDS} rounds)")
        logger.info(f"{'='*70}")
        logger.info(f"   • Local Epochs: {LOCAL_EPOCHS}")
        logger.info(f"   • Client Fraction: {CLIENT_FRACTION}")
        logger.info(f"   • Learning Rate: {LEARNING_RATE}")
        
        round_history = []
        
        # 📂 RECUPERO CHECKPOINT (se esiste)
        RESUME_FROM_CHECKPOINT = None  # Es: "checkpoint_round_5.pt" per riprendere

        if RESUME_FROM_CHECKPOINT:
            ckpt_path = Path(LOGS_PATH) / RESUME_FROM_CHECKPOINT
            if ckpt_path.exists():
                checkpoint = torch.load(ckpt_path)
                global_model.load_state_dict(checkpoint['model_state_dict'])
                start_round = checkpoint['round'] + 1
                round_history = checkpoint['round_history']
                logger.info(f"✅ Ripreso da round {checkpoint['round']}")
            else:
                logger.warning(f"⚠️ Checkpoint non trovato: {ckpt_path}")
                start_round = 0
        else:
            start_round = 0
        
        # for round_num in range(NUM_ROUNDS): ----> modificato in come segue:
        for round_num in range(start_round, NUM_ROUNDS):
            # Esegui un round
            round_step = FederatedTrainingRound(
                round_num=round_num,
                local_epochs=LOCAL_EPOCHS,
                batch_size=1024,
                client_fraction=CLIENT_FRACTION,
            )
            
            round_state = State({
                'federated.federated_splits': federated_splits,
                'federated.global_model': global_model,
                'federated.loss_fn': loss_fn,
                'federated.optimizer_class': optimizer_class,
                'federated.optimizer_kwargs': optimizer_kwargs,
                'federated.device': device,
            })
            
            round_step.run(round_state)
            
            round_results = round_state.get('federated.round_results', dict)
            round_history.append(round_results)
            
            # 💾 SALVATAGGIO CHECKPOINT
            checkpoint_path = Path(LOGS_PATH) / f"checkpoint_round_{round_num}.pt"
            torch.save({
                'round': round_num,
                'model_state_dict': global_model.state_dict(),
                'round_history': round_history,
            }, checkpoint_path)
            logger.info(f"💾 Checkpoint salvato: {checkpoint_path.name}")
            
            # 🎯 VALUTAZIONE SU TEST AGGREGATO (ogni 5 round)
            if round_num % 2 == 0 and aggregated_test is not None:
                logger.info(f"\n🎯 Valutazione su test globale (Round {round_num})...")
                
                # 1. Istanzia lo Step
                eval_step = EvaluateGlobalModel(batch_size=1024)
                
                # 2. Esegui la valutazione passandogli i dati necessari
                eval_output = eval_step.run(
                    state=round_state, 
                    global_model=global_model, 
                    aggregated_test=aggregated_test, 
                    device=device
                )
                
                # 3. Estrai e logga i risultati
                metrics = eval_output["test_metrics"]
                logger.info(f"   📈 Accuracy: {metrics['accuracy']:.4f} | F1-Score: {metrics['f1']:.4f}")
                
                # Opzionale: salva i risultati nella history per grafici futuri
                # round_results['global_test_metrics'] = metrics
                round_history[-1]['global_test_metrics'] = metrics
                
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🎉 TRAINING FEDERATO COMPLETATO!")
        logger.info(f"{'='*70}")
        
        # 💾 SALVA MODELLO FINALE
        final_model_path = Path(LOGS_PATH) / "global_model_final.pt"
        torch.save(global_model.state_dict(), final_model_path)
        logger.info(f"💾 Modello finale salvato: {final_model_path}")

if __name__ == "__main__":
    main()

###############################################################################################
""" 
        # TO-DO: Qui implementeremo il training federato nella prossima sessione!
        # Per ora, implementiamo un training CENTRALIZZATO sul test aggregato
        # come baseline di confronto
        
        # ───────────────────────────────────────────────────────────────
        # 📥 CARICAMENTO DATI FEDERATI
        # ───────────────────────────────────────────────────────────────
        logger.info("📥 Caricamento dataset federati salvati...")
        
        # Per ora, usiamo il test set aggregato come baseline
        aggregated_test_path = Path(FEDERATED_DATA_PATH) / "aggregated_test.parquet"
        
        if not aggregated_test_path.exists():
            logger.error(f"❌ File non trovato: {aggregated_test_path}")
            logger.error("   Devi prima eseguire RUN_PREPROCESSING=True!")
            return
        
        # Carica test aggregato
        aggregated_test = pd.read_parquet(aggregated_test_path)
        logger.info(f"✅ Test aggregato caricato: {len(aggregated_test)} samples")
        
        # Metti nel state
        state.set("data.root", aggregated_test, pd.DataFrame)
        
        # ───────────────────────────────────────────────────────────────
        # 🔧 SETUP PIPELINE (DataLoaders, ecc.)
        # ───────────────────────────────────────────────────────────────
        setup_pipeline = ObservablePipeline(
            steps=[
                # Assegna partizioni (per ora usiamo solo test)
                # Nel training federato vero, caricheremo i dataset per host
                AssignSplitPartitions(),
                
                # Build DataLoader per test
                AssignSplitTarget(in_scope="data", out_scope="test"),
                BuildDataset(out_scope="test"),
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
        
        logger.info("🔧 Setup DataLoaders...")
        setup_pipeline.run(state)
        
        # ───────────────────────────────────────────────────────────────
        # 🤖 MODELLO E LOSS
        # ───────────────────────────────────────────────────────────────
        model = TabularClassifier(
            num_features=len(schema.numerical),
            cat_cardinalities=[20] * len(schema.categorical),
            num_classes=15,
            hidden_dims=[512, 256, 128],
            dropout=0.2,
        ).to(device)
        
        loss_fn = FocalLoss().to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.0003,
            weight_decay=1e-5
        )
        
        state.set("model", model, TabularClassifier)
        state.set("loss", loss_fn, FocalLoss)
        state.set("optimizer", optimizer, torch.optim.Adam)
        
        logger.info(f"🤖 Modello creato: {model.__class__.__name__}")
        logger.info(f"📊 Parametri: {sum(p.numel() for p in model.parameters()):,}")
        
        # ───────────────────────────────────────────────────────────────
        # 📊 CLASS WEIGHTS (gestione sbilanciamento)
        # ───────────────────────────────────────────────────────────────
        logger.info("\n⚖️ Calcolo class weights...")
        
        test_targets = state.get("test.targets", np.ndarray)
        unique_classes, class_counts = np.unique(test_targets, return_counts=True)
        
        class_weights_balanced = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=test_targets
        )
        
        # Smoothing
        class_weights_smoothed = np.power(class_weights_balanced, 0.40)
        class_weights_tensor = torch.FloatTensor(class_weights_smoothed).to(device)
        
        loss_fn_weighted = FocalLoss(
            class_weight=class_weights_tensor,
            gamma=1.5
        ).to(device)
        state.set("loss", loss_fn_weighted, FocalLoss)
        
        logger.info("✅ Class weights applicati")
        
        # ───────────────────────────────────────────────────────────────
        # 🏋️ EPOCH PIPELINE
        # ───────────────────────────────────────────────────────────────
        os.makedirs(LOGS_PATH, exist_ok=True)
        
        # Estrai nomi classi
        csv_path = RAW_DATA_PATH
        try:
            attack_classes = pd.read_csv(
                csv_path,
                usecols=["Attack"],
                dtype={"Attack": "category"}
            )
            original_classes = sorted(attack_classes["Attack"].unique())
            class_names = [str(c) for c in original_classes]
        except:
            class_names = [f"Class_{i}" for i in range(15)]
        
        epoch_pipeline = ObservablePipeline(
            steps=[
                # Per ora solo validation (non abbiamo train in questo setup)
                ValidateOneEpoch(
                    log_dir=LOGS_PATH,
                    log_prefix="test",
                    in_scope="test",
                    out_scope="test",
                    save_outputs=True,
                ),
                MakePredictions(
                    pred_fn=lambda x: torch.argmax(x, dim=1),
                    in_scope="test",
                    out_scope="test",
                ),
                ClassificationMetrics(
                    log_dir=LOGS_PATH,
                    log_prefix="test",
                    class_names=class_names,
                    in_scope="test",
                    out_scope="test",
                    save_confusion_matrix=True,
                    save_classification_report=True,
                    save_f1_per_class_plot=True,
                ),
            ],
            bus=bus,
            name="epoch_pipeline",
        )
        
        # ───────────────────────────────────────────────────────────────
        # 🎯 VALUTAZIONE (baseline senza training)
        # ───────────────────────────────────────────────────────────────
        logger.info("\n🎯 Valutazione baseline (modello random)...")
        epoch_pipeline.run(state)
        
        # ───────────────────────────────────────────────────────────────
        # 📊 PLOT METRICS
        # ───────────────────────────────────────────────────────────────
        logger.info("\n📊 Generazione grafici...")
        plot_step = PlotMetrics(
            metrics_dir=LOGS_PATH,
            output_dir=f"{LOGS_PATH}/plots",
            metrics_to_plot=["accuracy", "precision", "recall", "f1_micro", "f1_macro", "f1_weighted"],
            figsize=(12, 8),
            dpi=300,
            show_plots=False,
            in_scope="test",
            out_scope="plots",
            name="plot_metrics"
        )
        plot_step.run(state)
        
        logger.info("\n" + "="*70)
        logger.info("✅ VALUTAZIONE BASELINE COMPLETATA!")
        logger.info("="*70)
        logger.info(f"📂 Risultati salvati in: {LOGS_PATH}")
        logger.info("\n🚧 PROSSIMO STEP: Implementare training federato vero!") """
        
###############################################################################################
"""def main():
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
    # 4️⃣ PIPELINE PREPROCESSING - Caricamento e preparazione dati
    # ═══════════════════════════════════════════════════════════════════
    preprocessing_pipeline = ObservablePipeline(
        steps=[
            LoadData(
                path_in="c:/Users/simon/OneDrive/Documenti/TESI_UNI/SetUp/dataset_v2/cic_2018_v2.csv",
                schema=schema,
                nrows=100000000
            ),
            DropNulls(),
            StratifiedSplit(class_column=schema.target,
                train_size=0.8,
                val_size=0.1,
                test_size=0.1,
            ),
            fit_aware_pipeline,
            SaveData(
                file_path="c:/Users/simon/OneDrive/Documenti/TESI_UNI/SetUp/dataset_processati",
                file_name="cic_2018_v2",
                fmt="parquet",
            ),
        ],
        bus=bus,
        name="preprocessing_pipeline",
    )
    
    # ═══════════════════════════════════════════════════════════════════
    # 5️⃣ PIPELINE SETUP - Costruzione dataset e dataloader
    # ═══════════════════════════════════════════════════════════════════
    setup_pipeline = ObservablePipeline(
        steps=[
            # Carica train processato
            LoadData(
                path_in="c:/Users/simon/OneDrive/Documenti/TESI_UNI/SetUp/dataset_processati/cic_2018_v2.parquet"
            ),         
            #Assegna le partizioni di train - val - test
            AssignSplitPartitions(), 
            # 1. TRAIN
            AssignSplitTarget(in_scope="data", out_scope="train"),
            BuildDataset(out_scope="train"),
            BuildDataLoader(
                in_scope="train",
                out_scope="train",
                batch_size=2048,#1024,#512,
                num_workers=0,#2,  # Metti 0 se su Windows hai problemi, altrimenti 2
                pin_memory=True,
                shuffle=True,
                collate_fn=default_collate,
            ),
            # 2. VALIDATION
            AssignSplitTarget(in_scope="data", out_scope="val"),
            BuildDataset(out_scope="val"),
            BuildDataLoader(
                in_scope="val",
                out_scope="val",
                batch_size=1024, # Val e Test possono avere batch più grandi (non fanno backprop)
                shuffle=False,
                collate_fn=default_collate,
            ),
            # 3. TEST
            AssignSplitTarget(in_scope="data", out_scope="test"),
            BuildDataset(out_scope="test"),
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
    
    # ═══════════════════════════════════════════════════════════════════
    # 6️⃣ CONFIGURAZIONE MODELLO E LOSS
    # ═══════════════════════════════════════════════════════════════════
    device = torch.device("cuda")  
    model = TabularClassifier(
        num_features=len(schema.numerical),
        cat_cardinalities=[20] * len(schema.categorical),
        num_classes=15,
        hidden_dims=[512, 256, 128],  # <--- NUOVA ARCHITETTURA [BIG]
        dropout=0.2,                      # <--- DROPOUT AUMENTATO
        # hidden_dims=[512, 256, 128, 64],  # <--- NUOVA ARCHITETTURA [BIG]
        # dropout=0.3,                      # <--- DROPOUT AUMENTATO
    ).to(device)

    
    # Creo loss INIZIALE (senza class weighting)
    # loss_fn = ClassificationLoss().to(device)
    loss_fn = FocalLoss().to(device)
    # LA TENTEREMO PER IL TRY-SUCCESSIVO V_THELAST4: loss_fn = ClassificationLoss().to(device)
         
    # MODIFICA DEL 22/11
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)  # ← CAMBIATO da 0.00025 a 0.0003
    optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=0.0003,           # <--- MANTENIAMO 0.0003 (più sicuro per modello profondo)
            weight_decay=1e-5    # <--- AGGIUNTO WEIGHT DECAY (Leggero)
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # 7️⃣ CREAZIONE DELLO STATE INIZIALE
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
        
    #MODIFICA DEL 22/11
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,          # <--- FACTOR prima era 0.6
        patience=3,          # Rimetterei 4 o 3, visto che il modello è grosso e lento a muoversi
        min_lr=1e-6,
    )
    
    # ═══════════════════════════════════════════════════════════════════
    # 8️⃣ ESECUZIONE PREPROCESSING
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*70)
    logger.info("⚙️ FASE 1: PREPROCESSING DATI")
    logger.info("="*70)
 
    preprocessing_pipeline.run(state)
    
    # ═══════════════════════════════════════════════════════════════════
    # 9️⃣ COSTRUZIONE DATASET E DATALOADER
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*70)
    logger.info("🔧 FASE 2: SETUP DATASET E DATALOADER")
    logger.info("="*70)
    
    setup_pipeline.run(state)
    
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
    # LabelMap fa codes+1, quindi le label sono [1-15] invece di [0-14]
    # BuildDataset corregge il dataset, ma NON corregge train.targets/test.targets
    # usati dalle metriche, quindi dobbiamo farlo manualmente
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
    log_dir = "C:/Users/simon/OneDrive/Documenti/TESI_UNI/SetUp/Modelli_Salvati/logs1/v_DieselComeBack_v2"    
    os.makedirs(log_dir, exist_ok=True)
    
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
                log_prefix="val",
                in_scope="val",      # Uso "val" per monitorare l'andamento
                out_scope="val",
                save_outputs=False,
            ),
            # 3. EVALUATE su TEST (NUOVO STEP: Genera gli outputs che MakePredictions richiede)
            ValidateOneEpoch(
                log_dir=log_dir,
                log_prefix="test",
                in_scope="test",    # Usa il Test DataLoader
                out_scope="test",   # Scrive i risultati (outputs) nello scope 'test'
                save_outputs=True,  # Necessario per i passi successivi
            ),
            
            # 4. PREDICTIONS (Ora trova outputs in state['test.outputs'])
            MakePredictions(
                pred_fn=lambda x: torch.argmax(x, dim=1),
                in_scope="test",
                out_scope="test", # Esplicitiamo lo scope per chiarezza
            ), 
            ClassificationMetrics(
                log_dir=log_dir,                   # CAMBIO path
                log_prefix="test",                 # AGGIUNTO
                class_names=class_names,           # AGGIUNTO
                in_scope="test",
                out_scope="test",
                save_confusion_matrix=True,  # ✅ Salva metriche
                save_classification_report=True,
                save_f1_per_class_plot = True,
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
    
    # 🔧 SCEGLI LA STRATEGIA DI SMOOTHING
    smoothing_strategy = "log"  # Opzioni: "sqrt" (default) o "fourth_root" (più conservativo)
    
    # Calcola i pesi bilanciati automaticamente
    class_weights_balanced = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=train_targets
    )
    class_weights_log = 1 + np.log(total_samples / (class_counts + 1)) 
        
    # # Applica smoothing
    if smoothing_strategy == "log":
        logger.info("\n✅ STRATEGIA: Log Class Weighting (logaritmo)")
        class_weights_smoothed = class_weights_log
    elif smoothing_strategy == "power_0_3":
        logger.info("\n✅ STRATEGIA: Hybrid Power Weighting (Power 0.3)")
        class_weights_smoothed = np.power(class_weights_balanced, 0.30)
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
    loss_fn = FocalLoss(class_weight=class_weights_tensor, gamma=1.5).to(device)
    # loss_fn = ClassificationLoss(class_weight=class_weights_tensor).to(device)
    # state.set("loss", loss_fn, ClassificationLoss)
    state.set("loss", loss_fn, FocalLoss)
    logger.info(f"🎯 Loss function configurata: {smoothing_strategy}\n")
    # logger.info(f"🎯 Classification-Loss con pesi bilanciati NON smoothed\n")
    
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
        patience=12, #8,                 # ← CAMBIATO da 5 a 8 (più paziente)
        min_delta=0.0002, #0.0002,           # ← CAMBIATO da 0.0005 a 0.0002 (più tollerante)
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
    main()"""