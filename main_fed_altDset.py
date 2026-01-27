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
from src.idspy.real_fed_learn.steps.io.fed_saver import SaveFederatedData

# from src.idspy.core.federated_split import(
#     IdentifyTopHosts,
#     SplitByHosts,
#     FederatedSplits,
#     AggregatedTestSet,
#     AnalyzeNonIID,
#     ApplyFitAwareToFederatedSplits
# )
from src.idspy.real_fed_learn.core.fed_split import(
    IdentifyTopHosts,
    SplitByHosts,
    FederatedSplits,
    AggregatedTestSet,
    AnalyzeNonIID,
    ApplyFitAwareToFederatedSplits
)

from src.idspy.real_fed_learn.orchestrator import FederatedTrainingRound_C
from src.idspy.real_fed_learn.steps.model.evaluator import  EvaluateClusterModels_C
from src.idspy.real_fed_learn.steps.transform.fed_scale import (
    ComputeGlobalNormalizationStats,
    ApplyGlobalNormalization,
)
from src.idspy.real_fed_learn.steps.transform.fed_map import (
    FrequencyMapGlobal,
    LabelMapGlobal,
    CreateOneVsRestLabels
)
from src.idspy.real_fed_learn.steps.metrics.plotMetrics import ClusterRoundMetrics_C, ClusterPlotMetrics_C

setup_logging() #inizializza il logging: cioè configura il modulo logging di Python
logger = logging.getLogger(__name__)    #crea un logger per questo modulo
set_seeds(42) #imposta il seed per la generazione casuale consentendo la riproducibilità

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
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
    
    # directory base comune a tutti
    BASE_DIR = Path("c:/Users/simon/OneDrive/Documenti/TESI_UNI/SetUp")
    BASE_DIR2 = Path("c:/Users/simon/OneDrive/Documenti/TESI_UNI/SetUp/Modelli_Salvati/logs_federated2/attacks_log")
    
    # Percorsi
    RAW_DATA_PATH = BASE_DIR / "dataset_v2/nb15_v2.csv"
    FEDERATED_DATA_PATH = BASE_DIR / "dataset_processati_federati/Dset_nb15_v2/log1"
    
    # LOGS_PATH = BASE_DIR2 / "Exploits/smartCL/1_smartCL_100_LR0005"
    # LOGS_PATH = BASE_DIR2 / "Exploits/Greedy/1_Greedy_100_LR0005"
    
    # LOGS_PATH = BASE_DIR2 / "Fuzzers/smartCL/1_smartCL_100_LR0005"
    # LOGS_PATH = BASE_DIR2 / "Fuzzers/Greedy/1_Greedy_100_LR0005"
    
    # LOGS_PATH = BASE_DIR2 / "Generic/smartCL/1_smartCL_100_LR0005"
    #LOGS_PATH = BASE_DIR2 / "Generic/Greedy/1_Greedy_100_LR0005"
    
    # LOGS_PATH = BASE_DIR2 / "Reconnaissance/smartCL/1_smartCL_100_LR0005"
    LOGS_PATH = BASE_DIR2 / "Reconnaissance/Greedy/1_Greedy_100_LR0005"
    
    
    # Parametri FL
    NUM_ROUNDS = 15 #4#11 #DA            # Numero di round federati
    # LOCAL_EPOCHS = 5 #DA 3-->4-->5          # Epoche locali per client
    LOCAL_EPOCHS = 5 #DA 3-->4-->5          # Epoche locali per client
    CLIENT_FRACTION = 1.0     # Frazione client per round (1.0 = tutti)
    
    # LEARNING_RATE = 0.001 #DA 0.002-->0.0005-->0.001     # Learning rate per ottimizzatore
    LEARNING_RATE = 0.0005 
    
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
        
        fit_aware_pipeline = FitAwareObservablePipeline(
            steps=[
                #StandardScale(),           
                FrequencyMapGlobal(max_levels=20, in_scope="data",
                    out_scope="data"),  
                LabelMapGlobal(benign_tag="Benign",
                                # mapping_output_path=Path(FEDERATED_DATA_PATH) / "label_mapping.json", # 🆕 
                               in_scope="data",
                               out_scope="data"
                        ),
                CreateOneVsRestLabels(
                                        # attack_types=["DDOS attack-HOIC", "DoS attacks-Hulk", "Bot", "Infilteration", "DDoS attacks-LOIC-HTTP", "DDOS attack-LOIC-UDP", "DoS attacks-GoldenEye"],  
                                        attack_types=["Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Shellcode", "Worms", "Reconnaissance", "Generic"],  
                                        in_scope="data",
                                        out_scope="data"
                                    ),
                 
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
                    nrows=100_000_000  # 🔥 100M righe = dataset completo
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
        
        federated_data_dir = Path(FEDERATED_DATA_PATH)
        
        # ───────────────────────────────────────────────────────────────
        # 1️⃣ 📥 CARICAMENTO E PREPROCESSING DATI PER OGNI HOST
        # ───────────────────────────────────────────────────────────────
        
        logger.info("📥 Caricamento dataset federati...")
                
        # Trova tutte le cartelle host_*
        host_dirs = [d for d in federated_data_dir.iterdir() if d.is_dir() and d.name.startswith("host_")]
        
        if len(host_dirs) == 0:
            logger.error(f"❌ Nessun host trovato in {federated_data_dir}")
            logger.error("   Devi prima eseguire RUN_PREPROCESSING=True!")
            return
            
        logger.info(f"✅ Trovati {len(host_dirs)} host su disco.")
        
        federated_splits = {}
        
        for host_dir in host_dirs:
            host_id = host_dir.name  # es. "host_172_31_0_2"
            
            # Percorsi file
            train_path = host_dir / "train.parquet"
            val_path = host_dir / "val.parquet"
            test_path = host_dir / "test.parquet"
            
            if not all([train_path.exists(), val_path.exists(), test_path.exists()]):
                logger.warning(f"⚠️ File mancanti per {host_id}, skip.")
                continue
            
            # ==== Caricamento DataFrames e conversione di tutti i dati in numerico ====
            train_df = pd.read_parquet(train_path)
            val_df = pd.read_parquet(val_path)
            test_df = pd.read_parquet(test_path)
            
            # ⭐ DEBUG: Verifica presenza colonne One-vs-Rest
            ovr_cols = [c for c in train_df.columns if c.startswith('is_')]
            if len(ovr_cols) == 0:
                logger.error(f"❌ {host_id}: NESSUNA colonna 'is_*' trovata! Preprocessing fallito!")
                logger.error(f"   Colonne disponibili: {train_df.columns.tolist()}")
                continue
            else:
                logger.info(f"✅ {host_id}: Trovate {len(ovr_cols)} colonne One-vs-Rest: {ovr_cols}")
            
            # === Pulizia Colonne: rimozione di colonne non necessarie per il training ====
            cols_to_drop = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'original_Attack', 'Label']
            for df in [train_df, val_df, test_df]:
                df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
            
            # ==== Conversione Tipi (Numerici e Int per Attack) ====
            for df in [train_df, val_df, test_df]:
                # Converti tutto in numerico (tranne Attack)
                for col in df.columns:
                    if col != 'Attack':
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                # Forza Attack a intero
                if 'Attack' in df.columns:
                    df['Attack'] = df['Attack'].astype(np.int64)
            
            
            # ==== Sanity Check: Rimozione di Possibili Label Negativi (es. -1) che fanno crashare la Loss ===
            for split_name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
                if (df['Attack'] < 0).any():
                    invalid_count = (df['Attack'] < 0).sum()
                    logger.warning(f"⚠️ {host_id} [{split_name}]: Trovati e Rimossi {invalid_count} campioni con label negativo.")
                    
                    rows_before = len(df)

                    # Filtra mantenendo solo label validi; manteniamo solo le righe con Attack >= 0
                    df_clean = df[df['Attack'] >= 0].copy()
                    
                    # Aggiorna il riferimento al dataframe corretto
                    if split_name == "Train": train_df = df_clean
                    elif split_name == "Val": val_df = df_clean
                    elif split_name == "Test": test_df = df_clean
                               
                    logger.info(f"   ✂️ Rimossi {rows_before - len(df_clean)} samples.")

            # ==== Controllo Finale Post-Pulizia: verifica che train sia vuoto ====
            if len(train_df) == 0:
                logger.warning(f"⚠️ {host_id}: Train set vuoto dopo la pulizia! Skip host.")
                continue # Non aggiungere questo host a federated_splits
                        
            # Se il modello ha 15 classi, il max label accettabile è 14.
            max_label_found = train_df['Attack'].max()
            if max_label_found >= 15: 
                logger.error(f"❌ {host_id}: Trovato label {max_label_found} che è >= num_classes (15)!")
                        
            # Salvataggio nel dizionario temporaneo
            federated_splits[host_id] = {'train': train_df, 'val': val_df, 'test': test_df}
            
            logger.info(
                f"   ✅ {host_id}: train={len(federated_splits[host_id]['train'])}, "
                f"val={len(federated_splits[host_id]['val'])}, "
                f"test={len(federated_splits[host_id]['test'])}"
            )
        
        # ──────────────────────────────────────────────────────────────────────────────
        # 2️⃣ GESTIONE METADATA E FILTRAGGIO HOST (SELEZIONE ATTIVA)
        # ──────────────────────────────────────────────────────────────────────────────
        ATTACK_ID = 4  # ← CAMBIA SOLO QUESTO NUMERO!

        # 📋 Mappatura ID → Nome Attacco
        ATTACK_MAPPING = {
            1: "is_exploits",
            2: "is_fuzzers",
            3: 'is_generic',
            4: 'is_reconnaissance',
        }

        # ✅ Validazione + Selezione
        if ATTACK_ID not in ATTACK_MAPPING:
            logger.error(f"❌ ATTACK_ID={ATTACK_ID} non valido! Usa 1-5.")
            logger.error(f"   Mapping disponibile: {ATTACK_MAPPING}")
            return

        TARGET_ATTACK = ATTACK_MAPPING[ATTACK_ID]
        logger.info(f"\n🎯 ATTACK_ID={ATTACK_ID} → {TARGET_ATTACK}")

        # Filtra solo client che HANNO questo attacco nei dati
        selected_clients = {}
        for host_id, splits in federated_splits.items():
            train_df = splits['train']
            
            # Verifica se l'attacco è presente
            if TARGET_ATTACK in train_df.columns:
                n_attack_samples = (train_df[TARGET_ATTACK] == 1).sum()
                
                if n_attack_samples >= 100:  # Soglia minima
                    selected_clients[host_id] = splits
                    logger.debug(f"✅ {host_id}: {n_attack_samples} campioni di {TARGET_ATTACK}")
                else:
                    logger.warning(f"⚠️ {host_id}: troppo pochi sample ({n_attack_samples}), skip")

        federated_splits = selected_clients

        if len(federated_splits) < 3:
            logger.error(f"❌ Servono almeno 3 client con {TARGET_ATTACK} per il clustering!")
            return

        logger.info(f"\n🎯 Focus attacco: {TARGET_ATTACK}")
        logger.info(f"📊 Client partecipanti: {len(federated_splits)}")
        
        
        # === Caricamento Test Set Aggregato: ottenuto dalla somma dei test set di ogni client (Globale) ====
        aggregated_test_path = federated_data_dir / "aggregated_test.parquet"
        aggregated_test = pd.read_parquet(aggregated_test_path) if aggregated_test_path.exists() else None
        if aggregated_test is not None:
             logger.info(f"✅ Test Set Aggregato caricato: {len(aggregated_test)} samples")
        else:
             logger.warning("⚠️ Test Set Aggregato non trovato.")
        
        
        # ───────────────────────────────────────────────────────────────
        # 3️⃣ 🤖 CREAZIONE MODELLO GLOBALE E SETUP TRAINING
        # ───────────────────────────────────────────────────────────────
        
        logger.info("\n🤖 Creazione modello globale...")
        logger.debug("\n🔍 ============== DEBUG COLONNE ==============")

        # 1. Mostra TUTTE le colonne e i loro dtype
        first_host_id = next(iter(federated_splits))
        sample_df = federated_splits[first_host_id]['train']

        logger.debug(f"📊 Totale colonne nel DataFrame: {len(sample_df.columns)}")

        # 2. Raggruppa per dtype
        for dtype_name in ['float64', 'float32', 'int64', 'int32', 'object']:
            cols_of_type = sample_df.select_dtypes(include=[dtype_name]).columns.tolist()
            if cols_of_type:
                logger.debug(f"\n📌 Colonne {dtype_name} ({len(cols_of_type)}):")
                for col in cols_of_type:
                    logger.debug(f"   • {col}")

        # 3. Verifica se le colonne is_* sono numerical
        is_cols_in_df = [c for c in sample_df.columns if c.startswith('is_')]
        logger.debug(f"\n🔎 Colonne is_* presenti ({len(is_cols_in_df)}):")
        for col in is_cols_in_df:
            dtype = sample_df[col].dtype
            logger.debug(f"   • {col} → dtype: {dtype}")

        logger.debug("🔍 ============================================\n")
        
        # ⭐ DEFINIZIONE ESPLICITA (dal tuo schema originale)
        KNOWN_CATEGORICAL_COLS = [
            'L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO',
            'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS',
            'ICMP_TYPE', 'ICMP_IPV4_TYPE', 'DNS_QUERY_ID', 'DNS_QUERY_TYPE'
        ]

        # Escludi target + metadata
        all_target_cols = [c for c in sample_df.columns if c.startswith('is_')]
        excluded_cols = {'Attack', 'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'original_Attack', 'Label'}
        excluded_cols.update(all_target_cols)

        # ✅ Categoriche: quelle in KNOWN_CATEGORICAL_COLS che esistono nel df
        real_categorical_cols = [c for c in KNOWN_CATEGORICAL_COLS 
                                if c in sample_df.columns and c not in excluded_cols]

        # ✅ Numeriche: tutto il resto (escluse categoriche, target, metadata)
        real_numerical_cols = [c for c in sample_df.columns 
                            if c not in real_categorical_cols 
                            and c not in excluded_cols]
        
        # ═══════════════════════════════════════════════════════════════════════
        # 🔍 CALCOLO CARDINALITIES GLOBALI (su tutti gli host)
        # ═══════════════════════════════════════════════════════════════════════

        logger.info("🔢 Calcolo cardinalities globali...")

        # Inizializza dizionario per tracciare i max per colonna
        global_max_values = {col: -np.inf for col in real_categorical_cols}

        # Scansiona TUTTI gli host
        for host_id, splits in federated_splits.items():
            host_train = splits['train']
            
            for col in real_categorical_cols:
                if col in host_train.columns:
                    col_data = host_train[col]
                    
                    # Arrotonda se float
                    if col_data.dtype in ['float64', 'float32']:
                        max_val = np.ceil(col_data.max())
                    else:
                        max_val = col_data.max()
                    
                    # Aggiorna max globale
                    global_max_values[col] = max(global_max_values[col], max_val)

        # Converti in cardinalities
        real_cat_cardinalities = []
        for col in real_categorical_cols:
            max_val = global_max_values[col]
            cardinality = int(max_val) + 1
            real_cat_cardinalities.append(cardinality)
            # logger.debug(f"   🔢 {col}: global_max={max_val} → cardinality={cardinality}")

        num_real_features = len(real_numerical_cols)
        num_real_categorical = len(real_categorical_cols)

        logger.debug(f"✅ Cardinalities globali: {real_cat_cardinalities}")
        
        

        logger.debug(f"✅ DOPO ESCLUSIONE:")
        logger.debug(f"📏 Misure rilevate dai dati reali ({first_host_id}):")
        logger.debug(f"   • Features Numeriche VALIDE: {num_real_features}---(Schema diceva: {len(schema.numerical)})")
        logger.debug(f"   • Features Categoriche VALIDE: {num_real_categorical}---(Schema diceva: {len(schema.categorical)})")
        logger.debug(f"   • Totale Features: {num_real_features + num_real_categorical}")
        
        # Controllo di sicurezza
        if num_real_features != len(schema.numerical):
            logger.warning("⚠️ ATTENZIONE: Il numero di colonne numeriche nei dati differisce dallo Schema!")
            logger.warning(f"   Si userà il valore dei DATI ({num_real_features}) per evitare crash.")

        # Creazione Modello con le misure REALI
        global_model = TabularClassifier(
            num_features=num_real_features,  
            cat_cardinalities=real_cat_cardinalities, 
            num_classes=2, 
            hidden_dims=[512, 256, 128],
            dropout=0.2,
        ).to(device)
        
        # ===== Verifica veloce output shape
        dummy_num = torch.randn(2, num_real_features).to(device)
        dummy_cat = torch.zeros(2, num_real_categorical, dtype=torch.long).to(device)
        output = global_model({'numerical': dummy_num, 'categorical': dummy_cat})
        print(f"✅ Output shape: {output.logits.shape}")
        # ===== DEVE stampare: torch.Size([2]) ← NON torch.Size([2, 2])!
        
        logger.info(f"✅ Modello creato correttamente: Input Layer adattato a {num_real_features} numeriche + embeddings.")
        
        logger.info(f"⚖️ pos_weight sarà calcolato dinamicamente da ogni client")

        # --- Setup Loss: Binary Cross Entropy con class weight
        from torch import nn
        loss_fn_class = nn.BCEWithLogitsLoss
        
        from src.idspy.nn.losses.classification import BinaryFocalLoss
        # loss_fn_class = BinaryFocalLoss  # Invece di BCEWithLogitsLos
        
        
        # --- Setup scheduler+Optimizer
        optimizer_class = torch.optim.Adam
        optimizer_kwargs = {'lr': LEARNING_RATE, 'weight_decay': 1e-5}
        # optimizer_kwargs = {'lr': LEARNING_RATE, 'weight_decay': 1e-4}
        
        scheduler_config = {
            'type': torch.optim.lr_scheduler.ReduceLROnPlateau,
            'kwargs': {
                'mode': 'min',
                'factor': 0.5,
                'patience': 1,#2,  # ← Più aggressivo nel federato
                'threshold': 0.001, #da 0.001 a 0.0001
                'min_lr': 1e-6,
            }
        }
        # scheduler_config = {
        #     'type': torch.optim.lr_scheduler.ReduceLROnPlateau,
        #     'kwargs': {
        #         'mode': 'min',
        #         'factor': 0.3,      # ← Più aggressivo (era 0.5)
        #         'patience': 1,
        #         'threshold': 0.0001, # ← Più sensibile (era 0.001)
        #         'min_lr': 1e-6,
        #         # 'min_lr': 5e-7,     # ← Più basso (era 1e-6)
        #     }
        # }
        logger.info(f"📉 Scheduler: {scheduler_config['type']} con patience={scheduler_config['kwargs']['patience']}")
        
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
            ckpt_path = log_dir / RESUME_FROM_CHECKPOINT
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
        
        round_state = State({
            'federated.federated_splits': federated_splits,
            'federated.global_model': global_model,
            # 'federated.loss_fn': loss_fn,
            'federated.loss_fn_class': loss_fn_class,
            'federated.optimizer_class': optimizer_class,
            'federated.optimizer_kwargs': optimizer_kwargs,
            'federated.scheduler_config': scheduler_config, # ⬅️ NUOVO
            'federated.device': device,
            'federated.target_attack': TARGET_ATTACK
        })
        
        for round_num in range(start_round, NUM_ROUNDS):
            
            # ✅ Aggiorna solo il modello (che cambia ogni round)
            round_state.set('federated.global_model', global_model, TabularClassifier)
            
            # Esegui un round
            round_step = FederatedTrainingRound_C(
                round_num=round_num,
                local_epochs=LOCAL_EPOCHS,
                batch_size=512, #1024,
                client_fraction=CLIENT_FRACTION,
            )
            round_step.run(round_state)
            
            # DEBUG MOMENTANEO CHE VERIFICA CHE I DATI PERSISTANO
            if round_state.has('federated.client_to_cluster'):
                mapping = round_state.get('federated.client_to_cluster', dict)
                logger.debug(f"🔍 Fine Round {round_num}: Mapping ha {len(mapping)} client")
            else:
                logger.warning(f"⚠️ Fine Round {round_num}: Mapping NON trovato!")
            
            ##################################################          
            
            # Salvataggio Risultati round
            round_results = round_state.get('federated.round_results', dict)
            round_history.append(round_results)
            
            # 💾 # Salvataggio CHECKPOINT
            checkpoint_path = log_dir / f"checkpoint_round_{round_num}.pt"
            torch.save({
                'round': round_num,
                'model_state_dict': global_model.state_dict(),
                'round_history': round_history,
            }, checkpoint_path)
            logger.info(f"💾 Checkpoint salvato: {checkpoint_path.name}")
            
            
            
            # ==== 🎯 Valutazione Periodica (ogni 2 round) ==== 
            if round_num % 2 == 0 and aggregated_test is not None:
                logger.info(f"\n🎯 Valutazione per Cluster (Round {round_num})...")
                
                # Verifica che abbiamo i dati necessari
                # if not round_state.has('federated.cluster_specific_weights'):
                #     logger.warning("⚠️ Nessun peso cluster disponibile, skip valutazione")
                # elif not round_state.has('federated.stable_target_to_cluster'):
                #     logger.warning("⚠️ Nessun mapping target→cluster disponibile, skip valutazione")
                # else:
                if not round_state.has('federated.cluster_specific_weights'):
                    logger.warning("⚠️ Nessun peso cluster disponibile, skip valutazione")
                else:
                                        
                    eval_cluster_step = EvaluateClusterModels_C(batch_size=1024)
                    
                    # Esegui valutazione
                    eval_output = eval_cluster_step.run(
                        state=round_state,
                        global_model=global_model,
                        aggregated_test=aggregated_test,
                        device=device,
                        cluster_specific_weights=round_state.get('federated.cluster_specific_weights', dict),

                        real_cat_cardinalities=real_cat_cardinalities,
                    )
                    
                    # Salva metriche nella history
                    round_results['cluster_test_metrics'] = eval_output['cluster_metrics']
                    round_results['aggregated_test_metrics'] = eval_output['aggregated_metrics']
                    
                    # ✅ NUOVO - Salvataggio metriche cluster (confusion matrix, report, F1)
                    cluster_metrics_step = ClusterRoundMetrics_C(
                        log_dir=str(Path(LOGS_PATH) / "cluster_metrics"),
                        class_names=["Other", "Target_Attack"]  # Binary classification
                    )
                    
                    # Salva metriche aggregate (non per cluster)
                    cluster_metrics_step.run(
                        state=round_state,
                        round_num=round_num,
                        cluster_results_detailed=eval_output['cluster_results_detailed']  # ← NUOVO campo con predictions

                    )


        # ──────────────────────────────────────────────────────────────────────────────
        # 🏁 FINE TRAINING E PLOTTING
        # ──────────────────────────────────────────────────────────────────────────────
        logger.info(f"\n{'='*70}")
        logger.info(f"🎉 TRAINING FEDERATO COMPLETATO!")
        logger.info(f"{'='*70}")

        # ✅ Generazione grafici finali
        cluster_plot_step = ClusterPlotMetrics_C(
            output_dir=str(Path(LOGS_PATH) / "plots"),
            rounds_history=round_history
        )

        cluster_plot_step.run(state=round_state)

        # 💾 Salvataggio Modello Finale
        final_model_path = Path(LOGS_PATH) / "global_model_final.pt"
        torch.save(global_model.state_dict(), final_model_path)
        logger.info(f"💾 Modello finale salvato: {final_model_path}")

if __name__ == "__main__":
    main()

"""  # ═══════════════════════════════════════════════════════════════════
    # 🚀 FASE 2: TRAINING FEDERATO (Esegui dopo preprocessing)
    # ═══════════════════════════════════════════════════════════════════
    if RUN_TRAINING:
        logger.info("\n" + "="*70)
        logger.info("🚀 FASE 2: TRAINING FEDERATO")
        logger.info("="*70)
        
        federated_data_dir = Path(FEDERATED_DATA_PATH)
        
        # ───────────────────────────────────────────────────────────────
        # 1️⃣ 📥 CARICAMENTO E PREPROCESSING DATI PER OGNI HOST
        # ───────────────────────────────────────────────────────────────
        
        logger.info("📥 Caricamento dataset federati...")
                
        # Trova tutte le cartelle host_*
        host_dirs = [d for d in federated_data_dir.iterdir() if d.is_dir() and d.name.startswith("host_")]
        
        if len(host_dirs) == 0:
            logger.error(f"❌ Nessun host trovato in {federated_data_dir}")
            logger.error("   Devi prima eseguire RUN_PREPROCESSING=True!")
            return
            
        logger.info(f"✅ Trovati {len(host_dirs)} host su disco.")
        
        federated_splits = {}
        
        for host_dir in host_dirs:
            host_id = host_dir.name  # es. "host_172_31_0_2"
            
            # Percorsi file
            train_path = host_dir / "train.parquet"
            val_path = host_dir / "val.parquet"
            test_path = host_dir / "test.parquet"
            
            if not all([train_path.exists(), val_path.exists(), test_path.exists()]):
                logger.warning(f"⚠️ File mancanti per {host_id}, skip.")
                continue
            
            # ==== Caricamento DataFrames e conversione di tutti i dati in numerico ====
            train_df = pd.read_parquet(train_path)
            val_df = pd.read_parquet(val_path)
            test_df = pd.read_parquet(test_path)
            
            # ⭐ DEBUG: Verifica presenza colonne One-vs-Rest
            ovr_cols = [c for c in train_df.columns if c.startswith('is_')]
            if len(ovr_cols) == 0:
                logger.error(f"❌ {host_id}: NESSUNA colonna 'is_*' trovata! Preprocessing fallito!")
                logger.error(f"   Colonne disponibili: {train_df.columns.tolist()}")
                continue
            else:
                logger.info(f"✅ {host_id}: Trovate {len(ovr_cols)} colonne One-vs-Rest: {ovr_cols}")
            
            # === Pulizia Colonne: rimozione di colonne non necessarie per il training ====
            cols_to_drop = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'original_Attack', 'Label']
            for df in [train_df, val_df, test_df]:
                df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
            
            # ==== Conversione Tipi (Numerici e Int per Attack) ====
            for df in [train_df, val_df, test_df]:
                # Converti tutto in numerico (tranne Attack)
                for col in df.columns:
                    if col != 'Attack':
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                # Forza Attack a intero
                if 'Attack' in df.columns:
                    df['Attack'] = df['Attack'].astype(np.int64)
            
            
            # ==== Sanity Check: Rimozione di Possibili Label Negativi (es. -1) che fanno crashare la Loss ===
            for split_name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
                if (df['Attack'] < 0).any():
                    invalid_count = (df['Attack'] < 0).sum()
                    logger.warning(f"⚠️ {host_id} [{split_name}]: Trovati e Rimossi {invalid_count} campioni con label negativo.")
                    
                    rows_before = len(df)

                    # Filtra mantenendo solo label validi; manteniamo solo le righe con Attack >= 0
                    df_clean = df[df['Attack'] >= 0].copy()
                    
                    # Aggiorna il riferimento al dataframe corretto
                    if split_name == "Train": train_df = df_clean
                    elif split_name == "Val": val_df = df_clean
                    elif split_name == "Test": test_df = df_clean
                               
                    logger.info(f"   ✂️ Rimossi {rows_before - len(df_clean)} samples.")

            # ==== Controllo Finale Post-Pulizia: verifica che train sia vuoto ====
            if len(train_df) == 0:
                logger.warning(f"⚠️ {host_id}: Train set vuoto dopo la pulizia! Skip host.")
                continue # Non aggiungere questo host a federated_splits
                        
            # Se il modello ha 15 classi, il max label accettabile è 14.
            max_label_found = train_df['Attack'].max()
            if max_label_found >= 15: 
                logger.error(f"❌ {host_id}: Trovato label {max_label_found} che è >= num_classes (15)!")
                        
            # Salvataggio nel dizionario temporaneo
            federated_splits[host_id] = {'train': train_df, 'val': val_df, 'test': test_df}
            
            logger.info(
                f"   ✅ {host_id}: train={len(federated_splits[host_id]['train'])}, "
                f"val={len(federated_splits[host_id]['val'])}, "
                f"test={len(federated_splits[host_id]['test'])}"
            )
        
        # ──────────────────────────────────────────────────────────────────────────────
        # 2️⃣ GESTIONE METADATA E FILTRAGGIO HOST (SELEZIONE ATTIVA)
        # ──────────────────────────────────────────────────────────────────────────────
        meta_path = federated_data_dir / "metadata.json"
        host_specialization = {}
        label_encoder_classes = ["Benign", "Attack"] # Default
        
        if meta_path.exists():
            import json
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            host_specialization = metadata.get('host_specialization', {})
            
            # --- Caricamento Label Encoder ---
            if 'training_label_map' in metadata:
                label_mapping = metadata['training_label_map']
                label_encoder_classes = [None] * len(label_mapping)
                for name, idx in label_mapping.items():
                    label_encoder_classes[idx] = name
                logger.info(f"✅ Label encoder caricato da metadata.")
            
            # --- Calcolo Distribuzione Esperti (Log) ---
            expert_counts = {}
            for ip, spec in host_specialization.items():
                target = spec.get('target_column')
                if target:
                    expert_counts[target] = expert_counts.get(target, 0) + 1
            
            logger.info(f"✅ Specializzazione caricata per {len(host_specialization)} host.")
            logger.info(f"📊 Distribuzione esperti: {expert_counts}")

        else:
            logger.error("❌ metadata.json non trovato!")
            return

        # ==== Filtro 1: Solo host con specializzazione definita ====
        active_hosts = {}
        for host_id, splits in federated_splits.items():
            # Ricostruisce l'IP dal nome cartella per cercare nei metadata
            ip_key = host_id.replace('host_', '').replace('_', '.')
            if host_specialization.get(ip_key, {}).get('target_column') is not None:
                active_hosts[host_id] = splits
        
        logger.info(f"🎯 Host con specializzazione valida: {len(active_hosts)}/{len(federated_splits)}")

        federated_splits = active_hosts  # Usa solo host con attacchi
                
        
        # ==== Filtro 2: Host con sufficienti attacchi (Quality Check) ====
        MIN_ATTACK_RATIO = 0.05  # Almeno 5% Attacks (flessibile)
        MIN_ATTACK_SAMPLES = 300  # O almeno 300 sample assoluti

        filtered_splits = {}
        client_attack_info = {}  # Per weighted aggregation
        for host_id, splits in federated_splits.items():
            train_df = splits['train']
            attack_count = (train_df['Attack'] == 1).sum()
            total_count = len(train_df)
            attack_ratio = attack_count / total_count if total_count > 0 else 0
            
            # Criterio: Almeno 5% Attacks O almeno 300 sample
            if attack_ratio >= MIN_ATTACK_RATIO or attack_count >= MIN_ATTACK_SAMPLES:
                filtered_splits[host_id] = splits
                client_attack_info[host_id] = {
                    'attack_ratio': attack_ratio,
                    'attack_count': attack_count,
                    'total_count': total_count
                }
                logger.info(f"✅ {host_id}: {attack_ratio*100:.2f}% Attacks ({attack_count}/{total_count})")
            else:
                logger.warning(f"⏭️ SKIP {host_id}: {attack_ratio*100:.2f}% Attacks (troppo pochi)")


        federated_splits = filtered_splits # Sostituisci con la lista filtrata
        if len(federated_splits) < 2:
            logger.error("❌ Meno di 2 client validi! Abbassa MIN_ATTACK_RATIO o riduci num_hosts.")
            return

        logger.info(f"\n📊 Client finali: {len(federated_splits)}/{len(host_dirs)}")
        
        
        # === Caricamento Test Set Aggregato (Globale) ====
        aggregated_test_path = federated_data_dir / "aggregated_test.parquet"
        aggregated_test = pd.read_parquet(aggregated_test_path) if aggregated_test_path.exists() else None
        if aggregated_test is not None:
             logger.info(f"✅ Test Set Aggregato caricato: {len(aggregated_test)} samples")
        else:
             logger.warning("⚠️ Test Set Aggregato non trovato.")
        
        
        # ───────────────────────────────────────────────────────────────
        # 3️⃣ 🤖 CREAZIONE MODELLO GLOBALE E SETUP TRAINING
        # ───────────────────────────────────────────────────────────────
        
        logger.info("\n🤖 Creazione modello globale...")
        logger.debug("\n🔍 ============== DEBUG COLONNE ==============")

        # 1. Mostra TUTTE le colonne e i loro dtype
        first_host_id = next(iter(federated_splits))
        sample_df = federated_splits[first_host_id]['train']

        logger.debug(f"📊 Totale colonne nel DataFrame: {len(sample_df.columns)}")

        # 2. Raggruppa per dtype
        # for dtype_name in ['float64', 'float32', 'int64', 'int32', 'object']:
        #     cols_of_type = sample_df.select_dtypes(include=[dtype_name]).columns.tolist()
        #     if cols_of_type:
        #         logger.debug(f"\n📌 Colonne {dtype_name} ({len(cols_of_type)}):")
        #         for col in cols_of_type:
        #             logger.debug(f"   • {col}")

        # # 3. Verifica se le colonne is_* sono numerical
        # is_cols_in_df = [c for c in sample_df.columns if c.startswith('is_')]
        # logger.debug(f"\n🔎 Colonne is_* presenti ({len(is_cols_in_df)}):")
        # for col in is_cols_in_df:
        #     dtype = sample_df[col].dtype
        #     logger.debug(f"   • {col} → dtype: {dtype}") 

        logger.debug("🔍 ============================================\n")

        # ==== POI APPLICA LA STESSA LOGICA DEL CLIENT ====
        # Escludi TUTTE le colonne is_* + metadata
        # all_target_cols = [c for c in sample_df.columns if c.startswith('is_')]
        # excluded_cols = {'Attack', 'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'original_Attack', 'Label'}
        # excluded_cols.update(all_target_cols)

        # real_numerical_cols = sample_df.select_dtypes(include=['float64', 'float32', 'float16']).columns.tolist()
        # real_numerical_cols = [c for c in real_numerical_cols if c not in excluded_cols]  # ✅ Escludi is_*

        # real_categorical_cols = [c for c in sample_df.columns 
        #                         if c not in real_numerical_cols 
        #                         and c not in excluded_cols]

        # num_real_features = len(real_numerical_cols)
        # num_real_categorical = len(real_categorical_cols)
        
        
        # ⭐ DEFINIZIONE ESPLICITA (dal tuo schema originale)
        KNOWN_CATEGORICAL_COLS = [
            'L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO',
            'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS',
            'ICMP_TYPE', 'ICMP_IPV4_TYPE', 'DNS_QUERY_ID', 'DNS_QUERY_TYPE'
        ]

        # Escludi target + metadata
        all_target_cols = [c for c in sample_df.columns if c.startswith('is_')]
        excluded_cols = {'Attack', 'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'original_Attack', 'Label'}
        excluded_cols.update(all_target_cols)

        # ✅ Categoriche: quelle in KNOWN_CATEGORICAL_COLS che esistono nel df
        real_categorical_cols = [c for c in KNOWN_CATEGORICAL_COLS 
                                if c in sample_df.columns and c not in excluded_cols]

        # ✅ Numeriche: tutto il resto (escluse categoriche, target, metadata)
        real_numerical_cols = [c for c in sample_df.columns 
                            if c not in real_categorical_cols 
                            and c not in excluded_cols]
        
        # Calcola max value per ogni colonna categorical
        # real_cat_cardinalities = []
        # for col in real_categorical_cols:
        #     max_val = sample_df[col].max()
        #     cardinality = int(max_val) + 1  # +1 perché indici partono da 0
        #     real_cat_cardinalities.append(cardinality)
        #     logger.debug(f"   🔢 {col}: max={max_val} → cardinality={cardinality}")
        # ═══════════════════════════════════════════════════════════════════════
        # 🔍 CALCOLO CARDINALITIES GLOBALI (su tutti gli host)
        # ═══════════════════════════════════════════════════════════════════════

        logger.info("🔢 Calcolo cardinalities globali...")

        # Inizializza dizionario per tracciare i max per colonna
        global_max_values = {col: -np.inf for col in real_categorical_cols}

        # Scansiona TUTTI gli host
        for host_id, splits in federated_splits.items():
            host_train = splits['train']
            
            for col in real_categorical_cols:
                if col in host_train.columns:
                    col_data = host_train[col]
                    
                    # Arrotonda se float
                    if col_data.dtype in ['float64', 'float32']:
                        max_val = np.ceil(col_data.max())
                    else:
                        max_val = col_data.max()
                    
                    # Aggiorna max globale
                    global_max_values[col] = max(global_max_values[col], max_val)

        # Converti in cardinalities
        real_cat_cardinalities = []
        for col in real_categorical_cols:
            max_val = global_max_values[col]
            cardinality = int(max_val) + 1
            real_cat_cardinalities.append(cardinality)
            # logger.debug(f"   🔢 {col}: global_max={max_val} → cardinality={cardinality}")

        num_real_features = len(real_numerical_cols)
        num_real_categorical = len(real_categorical_cols)

        logger.debug(f"✅ Cardinalities globali: {real_cat_cardinalities}")
        
        

        logger.debug(f"✅ DOPO ESCLUSIONE:")
        logger.debug(f"📏 Misure rilevate dai dati reali ({first_host_id}):")
        logger.debug(f"   • Features Numeriche VALIDE: {num_real_features}---(Schema diceva: {len(schema.numerical)})")
        logger.debug(f"   • Features Categoriche VALIDE: {num_real_categorical}---(Schema diceva: {len(schema.categorical)})")
        logger.debug(f"   • Totale Features: {num_real_features + num_real_categorical}")
        

         # 1. Prendiamo un host a caso (il primo) per prendere le misure
        # first_host_id = next(iter(federated_splits))
        # sample_df = federated_splits[first_host_id]['train']
        
        # # 2. Calcoliamo le colonne ESATTAMENTE come fa il Client
        # # (Copia della logica del Client per garantire coerenza)
        # real_numerical_cols = sample_df.select_dtypes(include=['float64', 'float32', 'float16']).columns.tolist()
        
        #         # Le categoriche sono tutto ciò che non è numerico e non è 'Attack' (o colonne escluse)
        #         # Nota: nel main hai già droppato IP e original_Attack, quindi restano solo features
        #         # real_categorical_cols = [c for c in sample_df.columns 
        #         #                          if c not in real_numerical_cols and c != 'Attack']
        # all_target_cols = [c for c in sample_df.columns if c.startswith('is_')]
        # excluded_cols = {'Attack', 'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'original_Attack', 'Label'}
        # excluded_cols.update(all_target_cols)

        # real_categorical_cols = [c for c in sample_df.columns 
        #                         if c not in real_numerical_cols 
        #                         and c not in excluded_cols]                         
                                 
                                 
                                 
                                 
                                 
        
        # num_real_features = len(real_numerical_cols)
        # num_real_categorical = len(real_categorical_cols)
        
        # logger.info(f"📏 Misure rilevate dai dati reali ({first_host_id}):")
        # logger.info(f"   • Features Numeriche: {num_real_features} (Schema diceva: {len(schema.numerical)})")
        # logger.info(f"   • Features Categoriche: {num_real_categorical} (Schema diceva: {len(schema.categorical)})") 
        
        # Controllo di sicurezza
        if num_real_features != len(schema.numerical):
            logger.warning("⚠️ ATTENZIONE: Il numero di colonne numeriche nei dati differisce dallo Schema!")
            logger.warning(f"   Si userà il valore dei DATI ({num_real_features}) per evitare crash.")

        # Creazione Modello con le misure REALI
        # global_model = TabularClassifier(
        #     num_features=num_real_features,  
        #     cat_cardinalities=[20] * num_real_categorical, 
        #     # num_classes=15, # O len(sample_df['Attack'].unique()) se vuoi essere dinamico
        #     num_classes=2, 
        #     hidden_dims=[512, 256, 128],
        #     dropout=0.2,
        # ).to(device)
        # Creazione Modello con le misure REALI
        global_model = TabularClassifier(
            num_features=num_real_features,  
            cat_cardinalities=real_cat_cardinalities, 
            num_classes=2, 
            hidden_dims=[512, 256, 128],
            dropout=0.2,
        ).to(device)
        
        # ===== Verifica veloce output shape
        dummy_num = torch.randn(2, num_real_features).to(device)
        dummy_cat = torch.zeros(2, num_real_categorical, dtype=torch.long).to(device)
        output = global_model({'numerical': dummy_num, 'categorical': dummy_cat})
        print(f"✅ Output shape: {output.logits.shape}")
        # ===== DEVE stampare: torch.Size([2]) ← NON torch.Size([2, 2])!
        
        logger.info(f"✅ Modello creato correttamente: Input Layer adattato a {num_real_features} numeriche + embeddings.")
        
        # === Calcolo Pesi per Loss (Bilanciamento Classi) ====
        # Basato su stime globali dei log e dei file json: Benign ~424k, Attacks ~171k
        # class_counts = np.array([424527, 171049])
        # weights = 1.0 / class_counts
        # normalized_weights = weights / weights[0] # Normalizza su classe 0 (Benign)
        
        # device_weights = torch.FloatTensor([normalized_weights[1]]).to(device)
        # logger.info(f"⚖️ Peso classe Attack (pos_weight): {normalized_weights[1]:.2f}")

        # # --- Setup Loss: Binary Cross Entropy con class weight
        # from torch import nn
        # loss_fn = nn.BCEWithLogitsLoss(pos_weight=device_weights).to(device)
        
        # # --- Setup Optimizer
        # optimizer_class = torch.optim.Adam
        # optimizer_kwargs = {'lr': LEARNING_RATE, 'weight_decay': 1e-5}
        
        
        logger.info(f"⚖️ pos_weight sarà calcolato dinamicamente da ogni client")

        # --- Setup Loss: Binary Cross Entropy con class weight
        from torch import nn
        loss_fn_class = nn.BCEWithLogitsLoss
        
        # --- Setup scheduler+Optimizer
        optimizer_class = torch.optim.Adam
        optimizer_kwargs = {'lr': LEARNING_RATE, 'weight_decay': 1e-5}
        
        scheduler_config = {
            'type': torch.optim.lr_scheduler.ReduceLROnPlateau,
            'kwargs': {
                'mode': 'min',
                'factor': 0.5,
                'patience': 1,#2,  # ← Più aggressivo nel federato
                'threshold': 0.0001,
                'min_lr': 1e-5,
            }
        }
        logger.info(f"📉 Scheduler: {scheduler_config['type']} con patience={scheduler_config['kwargs']['patience']}")
        
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
            ckpt_path = log_dir / RESUME_FROM_CHECKPOINT
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
        
        
        
        # SPOSTATO FUORI DAL CICLO FOR, PERCHE' ALTRIMENTI SI CREAVA OGNI VOLTA
        # LO STATO EX-NOVO DA CAPO PIUTTOSTO CHE AGGIORNARSI----Preparazione Stato per il round
        round_state = State({
            'federated.federated_splits': federated_splits,
            'federated.global_model': global_model,
            # 'federated.loss_fn': loss_fn,
            'federated.loss_fn_class': loss_fn_class,
            'federated.optimizer_class': optimizer_class,
            'federated.optimizer_kwargs': optimizer_kwargs,
            'federated.scheduler_config': scheduler_config, # ⬅️ NUOVO
            'federated.device': device,
            'federated.client_attack_info': client_attack_info,  # ⬅️ NUOVO
            'federated.host_specialization': host_specialization,  # ⬅️ NUOVO

        })
        
        for round_num in range(start_round, NUM_ROUNDS):
            
            # # Preparazione Stato per il round
            # round_state = State({
            #     'federated.federated_splits': federated_splits,
            #     'federated.global_model': global_model,
            #     'federated.loss_fn': loss_fn,
            #     'federated.optimizer_class': optimizer_class,
            #     'federated.optimizer_kwargs': optimizer_kwargs,
            #     'federated.device': device,
            #     'federated.client_attack_info': client_attack_info,  # ⬅️ NUOVO
            #     'federated.host_specialization': host_specialization,  # ⬅️ NUOVO

            # })
            
            # ✅ Aggiorna solo il modello (che cambia ogni round)
            round_state.set('federated.global_model', global_model, TabularClassifier)
            # round_state.set('federated.scheduler_config', scheduler_config, dict)
            
            # Esegui un round
            round_step = FederatedTrainingRound_B(
                round_num=round_num,
                local_epochs=LOCAL_EPOCHS,
                batch_size=512, #1024,
                client_fraction=CLIENT_FRACTION,
            )
            round_step.run(round_state)
            
            
            # DEBUG MOMENTANEO CHE VERIFICA CHE I DATI PERSISTANO
            if round_state.has('federated.client_to_cluster'):
                mapping = round_state.get('federated.client_to_cluster', dict)
                logger.debug(f"🔍 Fine Round {round_num}: Mapping ha {len(mapping)} client")
            else:
                logger.warning(f"⚠️ Fine Round {round_num}: Mapping NON trovato!")
            
            ##################################################          
            
            # Salvataggio Risultati round
            round_results = round_state.get('federated.round_results', dict)
            round_history.append(round_results)
            
            # 💾 # Salvataggio CHECKPOINT
            checkpoint_path = log_dir / f"checkpoint_round_{round_num}.pt"
            torch.save({
                'round': round_num,
                'model_state_dict': global_model.state_dict(),
                'round_history': round_history,
            }, checkpoint_path)
            logger.info(f"💾 Checkpoint salvato: {checkpoint_path.name}")
            
            
             # ==== 🎯 Valutazione Periodica (ogni 2 round) ==== 
            # if round_num % 2 == 0 and aggregated_test is not None:
            #     logger.info(f"\n🎯 Valutazione su test globale (Round {round_num})...")
                
            #     # 1. Istanzia lo Step
            #     eval_step = EvaluateGlobalModel(batch_size=1024)
                
            #     # 2. Esegui la valutazione passandogli i dati necessari
            #     eval_output = eval_step.run(
            #         state=round_state, 
            #         global_model=global_model, 
            #         aggregated_test=aggregated_test, 
            #         device=device
            #     )
                
            #     # 3. Estrai e logga i risultati
            #     metrics = eval_output["test_metrics"]
            #     # salva i risultati nella history per grafici futuri
            #     round_results['global_test_metrics'] = metrics
                
            #     logger.info(f"   📈 Accuracy: {metrics['accuracy']:.4f} | F1-Score: {metrics['f1']:.4f}")              
                
            #     # Salvataggio metriche dettagliate (CSV/JSON)
            #     metrics_step = FederatedRoundMetrics(
            #         log_dir=str(Path(LOGS_PATH) / "metrics"),
            #         class_names=label_encoder_classes
            #     )
                
            #     metrics_step.run(
            #         state=round_state,
            #         round_num=round_num,
            #         predictions=eval_output['predictions'],
            #         targets=eval_output['targets'],
            #         round_results=round_results,
            #     ) 
                # ==== 🎯 Valutazione Periodica (ogni 2 round) ==== 
            if round_num % 2 == 0 and aggregated_test is not None:
                logger.info(f"\n🎯 Valutazione per Cluster (Round {round_num})...")
                
                # Verifica che abbiamo i dati necessari
                if not round_state.has('federated.cluster_specific_weights'):
                    logger.warning("⚠️ Nessun peso cluster disponibile, skip valutazione")
                elif not round_state.has('federated.stable_target_to_cluster'):
                    logger.warning("⚠️ Nessun mapping target→cluster disponibile, skip valutazione")
                else:
                                        
                    eval_cluster_step = EvaluateClusterModels(batch_size=1024)
                    
                    # Esegui valutazione
                    eval_output = eval_cluster_step.run(
                        state=round_state,
                        global_model=global_model,
                        aggregated_test=aggregated_test,
                        device=device,
                        cluster_specific_weights=round_state.get('federated.cluster_specific_weights', dict),
                        stable_target_to_cluster=round_state.get('federated.stable_target_to_cluster', dict),
                    )
                    
                    # Salva metriche nella history
                    round_results['cluster_test_metrics'] = eval_output['cluster_metrics']
                    round_results['aggregated_test_metrics'] = eval_output['aggregated_metrics']
                    
                    # ✅ NUOVO - Salvataggio metriche cluster (confusion matrix, report, F1)
                    cluster_metrics_step = ClusterRoundMetrics(
                        log_dir=str(Path(LOGS_PATH) / "cluster_metrics"),
                        class_names=["Other", "Target_Attack"]  # Binary classification
                    )
                    
                    # Salva metriche aggregate (non per cluster)
                    cluster_metrics_step.run(
                        state=round_state,
                        round_num=round_num,
                        cluster_results_detailed=eval_output['cluster_results_detailed']  # ← NUOVO campo con predictions

                    )


        # ──────────────────────────────────────────────────────────────────────────────
        # 🏁 FINE TRAINING E PLOTTING
        # ──────────────────────────────────────────────────────────────────────────────
        logger.info(f"\n{'='*70}")
        logger.info(f"🎉 TRAINING FEDERATO COMPLETATO!")
        logger.info(f"{'='*70}")
        from src.idspy.federated_learning.steps.metrics.visualization import ClusterPlotMetrics
        # ✅ Generazione grafici finali
        cluster_plot_step = ClusterPlotMetrics(
            output_dir=str(Path(LOGS_PATH) / "plots"),
            rounds_history=round_history
        )

        cluster_plot_step.run(state=round_state)

        # 💾 Salvataggio Modello Finale
        final_model_path = Path(LOGS_PATH) / "global_model_final.pt"
        torch.save(global_model.state_dict(), final_model_path)
        logger.info(f"💾 Modello finale salvato: {final_model_path}")
 """  