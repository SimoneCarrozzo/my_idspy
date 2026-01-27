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
from src.idspy.steps.transforms.scale import StandardScale2, StandardScaleMemoryEfficient
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

Flusso completo:
===============
1. Definizione dello schema dei dati
2. Setup del sistema di eventi (EventBus)
3. Preprocessing dei dati (fit-aware pipeline)
4. Configurazione del modello e della loss
5. Training con Early Stopping
6. Visualizzazione delle metriche
"""


""" E' IMPORTANTE PRECISARE CHE PRIMA DI ARRIVARE A QUESTA VERSIONE, CHE CHIAMEREMO V2_SND_TRY_ES E
    CHE MI HA FATTO OTTENERE:
        a) accuracy: 99,5
        b) f1_macro: 80.4
        c) f1_micro: 99.5
        d) f1_weighted: 99.4
        e) precision: 95.9
        f) recall: 78
        g) EPOCHE RUNNATE: BEN 32/50
    
    IL SETUP IMPLEMENTATO NELLA VERSIONE PRECEDENTE V2_FIRST_TRY_ES, CHE MI HA FATTO OTTENERE:
        a) accuracy: 99,5
        b) f1_macro: 80.0
        c) f1_micro: 99.5
        d) f1_weighted: 99.4
        e) precision: 93.7
        f) recall: 77.4
        g) EPOCHE RUNNATE: BEN 7/50  
    
    PREVEDEVA:
    
    BuildDataLoader(
                in_scope="train",
                out_scope="train",
                batch_size=256, # MODIFICATO RISP V1_TRY_STATO_ARTE
                num_workers=2,  # MODIFICATO RISP V1_TRY_STATO_ARTE
                pin_memory=True, #AGGIUNTO
                shuffle=True,
                collate_fn=default_collate,
            )

    model = TabularClassifier(
        num_features=len(schema.numerical),
        cat_cardinalities=[20] * len(schema.categorical),
        num_classes=15,
        hidden_dims=[128, 64, 32],  # MODIFICATO RISP V1_TRY_STATO_ARTE
        dropout=0.15,               # MODIFICATO RISP V1_TRY_STATO_ARTE
    ).to(device)
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00025)    # MODIFICATO RISP V1_TRY_STATO_ARTE

    # #aggiunto
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2,      # MODIFICATO RISP V1_TRY_STATO_ARTE: ridotto da 3 a 2
        min_lr=1e-6,
        ####verbose=False
    )
    
    # 🔧 SCEGLI LA STRATEGIA DI SMOOTHING
    smoothing_strategy = "midway_root"  # Opzioni: "sqrt" (default) o "fourth_root" (più conservativo)  --> AGGIUNTO
            
    # Applica smoothing
    elif smoothing_strategy == "midway_root":
        logger.info("\n✅ STRATEGIA: MIDWAY-Smoothed Class Weighting (radice pari a 0.40)")
        class_weights_smoothed = np.power(class_weights_balanced, 0.40)
    else:
        raise ValueError(f"Strategia sconosciuta: {smoothing_strategy}")
        
     # 🆕 Crea lo step di training con early stopping
    training_step = TrainWithEarlyStopping(
        epoch_pipeline=epoch_pipeline,
        num_epochs=50,              # Numero massimo di epoche
        patience=5,                 # Ferma se nessun miglioramento per 3 epoche
        min_delta=0.0005,            # Miglioramento minimo significativo #update di punto 3) da min_delta=0.001 a min_delta=0.0005
        checkpoint_dir=f"{log_dir}/checkpoints_ES",
        save_best_only=True,        # Salva solo il miglior modello
        verbose=True,
        in_scope="train",      
        out_scope="train",     
        name="train_with_early_stopping",
        scheduler=scheduler
    ) 
    
"""

#AGGIUNTO 21/11:
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_CUDA_ALLOC_MAX_SPLIT_SIZE_MB'] = '128'  # Limita frammentazione

def main():
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
    # Questa pipeline esegue trasformazioni che devono essere "fittate"
    # sui dati di training (es. StandardScaler impara media e std)
    # StandardScale(), #standardizza le feature numeriche
    # FrequencyMap(max_levels=20), #codifica le classi categoriche in base alla frequenza, parametro max_lv imposta il numero massimo di livelli da considerare
    # LabelMap(), #trasforma le etichette di stringa in numeri interi
    fit_aware_pipeline = FitAwareObservablePipeline(
        steps=[
            StandardScaleMemoryEfficient(),           
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
            StratifiedSplit(class_column=schema.target),
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
            LoadData(
                path_in="c:/Users/simon/OneDrive/Documenti/TESI_UNI/SetUp/dataset_processati/cic_2018_v2.parquet"
            ),
            AssignSplitPartitions(),  # Assegna train/test splits
            AssignSplitTarget(in_scope="data", out_scope="train"),
            AssignSplitTarget(in_scope="data", out_scope="test"),
            BuildDataset(out_scope="train"),
            BuildDataLoader(
                in_scope="train",
                out_scope="train",
                batch_size=768, # 1024,#512
                num_workers=0,  # ← Windows non gestisce bene multiprocessing
                pin_memory=False, 
                shuffle=True,
                collate_fn=default_collate,
            ),
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
        hidden_dims=[256, 128, 64],
        dropout=0.2,
    ).to(device)

    
    # Creo loss INIZIALE (senza class weighting)
    loss_fn = ClassificationLoss().to(device)
         
    # 
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)  # 
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0004, weight_decay=1e-5)  # 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)  # 
    
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
        
    #MODIFICA DEL 6/12 causa 100M righe dataset
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.6, #0.5, #0.6,              # ← passo da 0.7 a 0.5 (riduzione più decisa)
        patience=10,#8 #6,              # ← passo da 4 a 5 (giusto compromesso)
        min_lr=1e-6 #5e-7,             # ← passo da 1e-6 a 5e-7 (più margine)
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
    log_dir = "C:/Users/simon/OneDrive/Documenti/TESI_UNI/SetUp/Modelli_Salvati/logs/v3_the_seventh"    
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
                log_prefix="val",          # AGGIUNTO 
                in_scope="test",           # ← mantengo "test" 
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
    
    """ # 🔧 NUOVA STRATEGIA: Smoothing Progressivo Dinamico
    # ═══════════════════════════════════════════════════════════════════
    smoothing_strategy = "adaptive_aggressive"  # ← CAMBIA QUI

    # Calcola i pesi bilanciati automaticamente
    class_weights_balanced = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=train_targets
    )

    # 🆕 SMOOTHING ADATTIVO BASATO SULLA CARDINALITÀ DELLA CLASSE
    if smoothing_strategy == "adaptive_aggressive":
        logger.info("\n✅ STRATEGIA: Adaptive Aggressive Smoothing")
        logger.info("   → Classi rare: esponente 0.60 (più peso)")
        logger.info("   → Classi medie: esponente 0.50")
        logger.info("   → Classi frequenti: esponente 0.40 (meno peso)\n")
        
        # Calcola percentuali classi
        class_percentages = class_counts / total_samples
        
        # Applica smoothing adattivo
        class_weights_smoothed = np.zeros_like(class_weights_balanced)
        
        for idx, (cls, pct) in enumerate(zip(unique_classes, class_percentages)):
            if pct < 0.01:  # Classi con < 1% dei dati (le 4 problematiche)
                exponent = 0.65  # ← Peso MASSIMO per classi rarissime
                category = "RARE"
            elif pct < 0.05:  # Classi con 1-5% dei dati
                exponent = 0.55
                category = "MEDIUM-RARE"
            elif pct < 0.15:  # Classi con 5-15% dei dati
                exponent = 0.45
                category = "MEDIUM"
            else:  # Classi maggioritarie (>15%)
                exponent = 0.35  # ← Peso MINIMO per classi frequenti
                category = "FREQUENT"
            
            class_weights_smoothed[idx] = np.power(class_weights_balanced[idx], exponent)
            
            logger.info(
                f"   Classe {cls:2d} ({class_names[cls]:20s}): "
                f"pct={pct*100:5.2f}% | {category:12s} | "
                f"exponent={exponent:.2f} | weight={class_weights_smoothed[idx]:.4f}"
            )

    elif smoothing_strategy == "sqrt":
        logger.info("\n✅ STRATEGIA: Smoothed Class Weighting (radice quadrata)")
        class_weights_smoothed = np.sqrt(class_weights_balanced)
    
    elif smoothing_strategy == "4_root":
        logger.info("\n✅ STRATEGIA: Ultra-Smoothed Class Weighting (radice quarta)")
        class_weights_smoothed = np.power(class_weights_balanced, 0.25)
        
    elif smoothing_strategy == "midway_root":
        logger.info("\n✅ STRATEGIA: MIDWAY-Smoothed Class Weighting (radice 0.40)")
        class_weights_smoothed = np.power(class_weights_balanced, 0.40)
        
    elif smoothing_strategy == "aggressive_uniform":
        # 🆕 Opzione alternativa: peso uniforme alle classi rare
        logger.info("\n✅ STRATEGIA: Aggressive Uniform (0.50 per tutte)")
        class_weights_smoothed = np.power(class_weights_balanced, 0.50)
        
    else:
        raise ValueError(f"Strategia sconosciuta: {smoothing_strategy}")

    # Stampa confronto pesi
    logger.info("\n⚖️ Confronto pesi delle classi:")
    for cls in unique_classes:
        logger.info(
            f"   Classe {cls:2d}: balanced={class_weights_balanced[cls]:.4f}, "
            f"smoothed={class_weights_smoothed[cls]:.4f}, "
            f"ratio={class_weights_smoothed[cls]/class_weights_balanced[cls]:.2%}"
        ) """
    """ # 🔧 NUOVA STRATEGIA: Adaptive Aggressive V2 (Ottimizzato per 100M)
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n✅ STRATEGIA: Adaptive Aggressive V2 (Ottimizzato per 100M)")
    logger.info("   → Classi < 0.1%: esponente 0.70 (BOOST MASSIMO)")
    logger.info("   → Classi 0.1-1%: esponente 0.60 (BOOST ALTO)")
    logger.info("   → Classi 1-5%: esponente 0.50 (BOOST MEDIO)")
    logger.info("   → Classi 5-15%: esponente 0.40 (BOOST LEGGERO)")
    logger.info("   → Classi >15%: esponente 0.30 (PENALIZZAZIONE)\n")

    # Calcola i pesi bilanciati automaticamente
    class_weights_balanced = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=train_targets
    )

    # Calcola percentuali classi
    class_percentages = class_counts / total_samples

    # Applica smoothing adattivo V2
    class_weights_smoothed = np.zeros_like(class_weights_balanced)

    for idx, (cls, pct) in enumerate(zip(unique_classes, class_percentages)):
        # 🎯 CALIBRAZIONE OTTIMIZZATA PER 100M SAMPLES
        if pct < 0.001:  # < 0.1% (SQL-Inj, SSH-BTF - le 2 più difficili)
            exponent = 0.70  # ← BOOST MASSIMO
            category = "ULTRA-RARE"
            emoji = "🔴"
        elif pct < 0.01:  # 0.1-1% (Loic-UDP, FTP-BTF)
            exponent = 0.60  # ← BOOST ALTO
            category = "RARE"
            emoji = "🟠"
        elif pct < 0.05:  # 1-5%
            exponent = 0.50  # ← BOOST MEDIO
            category = "MEDIUM-RARE"
            emoji = "🟡"
        elif pct < 0.15:  # 5-15%
            exponent = 0.40  # ← BOOST LEGGERO
            category = "MEDIUM"
            emoji = "🟢"
        else:  # >15% (classi maggioritarie)
            exponent = 0.30  # ← PENALIZZAZIONE
            category = "FREQUENT"
            emoji = "🔵"
        
        class_weights_smoothed[idx] = np.power(class_weights_balanced[idx], exponent)
        
        logger.info(
            f"{emoji} Classe {cls:2d} ({class_names[cls]:20s}): "
            f"freq={pct*100:6.3f}% | {category:12s} | "
            f"exp={exponent:.2f} | bal_w={class_weights_balanced[idx]:6.2f} | "
            f"final_w={class_weights_smoothed[idx]:6.2f}"
        )

    # Normalizza per avere media=1 (opzionale ma consigliato)
    class_weights_smoothed = class_weights_smoothed / class_weights_smoothed.mean()
 """
    logger.info("\n✅ STRATEGIA V4: Balanced Boost (Ottimizzato per 100M)")
    logger.info("   🔴 Classi < 0.1%:  exp=0.65 (era 0.80 in V3, 0.70 in V2)")
    logger.info("   🟠 Classi 0.1-1%:  exp=0.58 (era 0.70 in V3, 0.60 in V2)")
    logger.info("   🟡 Classi 1-5%:    exp=0.48 (era 0.55 in V3, 0.50 in V2)")
    logger.info("   🟢 Classi 5-15%:   exp=0.40 (invariato)")
    logger.info("   🔵 Classi >15%:    exp=0.30 (era 0.20 in V3, 0.30 in V2)\n")

    # Calcola i pesi bilanciati
    class_weights_balanced = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=train_targets
    )

    # Calcola percentuali classi
    class_percentages = class_counts / total_samples
    class_weights_smoothed = np.zeros_like(class_weights_balanced)

    for idx, (cls, pct) in enumerate(zip(unique_classes, class_percentages)):
        # 🎯 CALIBRAZIONE V4: Punto medio tra V2 e V3
        if pct < 0.001:  # SQL-Inj, SSH-BTF
            exponent = 0.65  # ← Intermedio tra 0.70 (V2) e 0.80 (V3)
            category = "ULTRA-RARE"
            emoji = "🔴"
        elif pct < 0.01:  # Loic-UDP, FTP-BTF
            exponent = 0.58  # ← Intermedio tra 0.60 (V2) e 0.70 (V3)
            category = "RARE"
            emoji = "🟠"
        elif pct < 0.05:  # Classi medie-rare
            exponent = 0.48  # ← Intermedio tra 0.50 (V2) e 0.55 (V3)
            category = "MEDIUM-RARE"
            emoji = "🟡"
        elif pct < 0.15:  # Classi medie
            exponent = 0.40  # ← Invariato (funziona bene)
            category = "MEDIUM"
            emoji = "🟢"
        else:  # Benign, DDoS-HTTP (classi maggioritarie)
            exponent = 0.30  # ← Torna a V2 (0.20 di V3 era troppo punitivo)
            category = "FREQUENT"
            emoji = "🔵"
        
        class_weights_smoothed[idx] = np.power(class_weights_balanced[idx], exponent)
        
        logger.info(
            f"{emoji} Classe {cls:2d} ({class_names[cls]:20s}): "
            f"freq={pct*100:6.3f}% | {category:12s} | "
            f"exp={exponent:.2f} | bal_w={class_weights_balanced[idx]:6.2f} | "
            f"final_w={class_weights_smoothed[idx]:6.2f}"
        )

    # Normalizza
    class_weights_smoothed = class_weights_smoothed / class_weights_smoothed.mean()

    # Statistiche
    logger.info(f"\n📊 Statistiche pesi V4:")
    logger.info(f"   Min: {class_weights_smoothed.min():.4f}")
    logger.info(f"   Max: {class_weights_smoothed.max():.4f}")
    logger.info(f"   Ratio max/min: {class_weights_smoothed.max()/class_weights_smoothed.min():.2f}x")

    # Converti in tensor
    class_weights_tensor = torch.FloatTensor(class_weights_smoothed).to(device)
    loss_fn = ClassificationLoss(class_weight=class_weights_tensor).to(device)
    state.set("loss", loss_fn, ClassificationLoss)
    logger.info(f"🎯 Loss function configurata: V4 Balanced Boost\n")
   
    
    # ═══════════════════════════════════════════════════════════════════
    # 1️⃣3️⃣ TRAINING CON EARLY STOPPING 
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*70)
    logger.info("🚀 FASE 4: TRAINING CON EARLY STOPPING")
    logger.info("="*70)
        
    #MODIFICA DEL 6/12
    training_step = TrainWithEarlyStopping(
        epoch_pipeline=epoch_pipeline,
        num_epochs=70,              
        patience=15, #18,                 # ← passo da 8 a 10 poiché con 100M più pazienza
        min_delta= 0.00005,#0.00002, #0.00005,           # ← passo da 0.0002 a 0.0001 con 100M più sensibile
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
