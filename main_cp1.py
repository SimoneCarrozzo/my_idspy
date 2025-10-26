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

Flusso completo:
===============
1. Definizione dello schema dei dati
2. Setup del sistema di eventi (EventBus)
3. Preprocessing dei dati (fit-aware pipeline)
4. Configurazione del modello e della loss
5. Training con Early Stopping
6. Visualizzazione delle metriche
"""

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
                path_in="c:/Users/simon/OneDrive/Documenti/TESI_UNI/DataSets/dataset_v2/cic_2018_v2.csv",
                schema=schema,
                nrows=10000000
            ),
            DropNulls(),
            StratifiedSplit(class_column=schema.target),
            fit_aware_pipeline,
            SaveData(
                file_path="c:/Users/simon/OneDrive/Documenti/TESI_UNI/DataSets/processati",
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
                path_in="c:/Users/simon/OneDrive/Documenti/TESI_UNI/DataSets/processati/cic_2018_v2.parquet"
            ),
            AssignSplitPartitions(),  # Assegna train/test splits
            AssignSplitTarget(in_scope="data", out_scope="train"),
            AssignSplitTarget(in_scope="data", out_scope="test"),
            BuildDataset(out_scope="train"),
            BuildDataLoader(
                in_scope="train",
                out_scope="train",
                batch_size=1024,#512,
                num_workers=6,
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
    # 6️⃣ PIPELINE EPOCA - Operazioni per ogni singola epoca
    # ═══════════════════════════════════════════════════════════════════
    epoch_pipeline = ObservablePipeline(
        steps=[
            TrainOneEpoch(),  
            ValidateOneEpoch(
                in_scope="test", 
                out_scope="test", 
                save_outputs=True
            ),  # Validazione su test set
            MakePredictions(pred_fn=lambda x: torch.argmax(x, dim=1)),  # Converte output in predizioni
            ClassificationMetrics("c:/Users/simon/OneDrive/Documenti/TESI_UNI/DataSets/Salvataggi/classification_exp_report"),  # Calcola e salva metriche
        ],
        bus=bus,
        name="epoch_pipeline",
    )
    
    # ═══════════════════════════════════════════════════════════════════
    # 7️⃣ CONFIGURAZIONE MODELLO E LOSS
    # ═══════════════════════════════════════════════════════════════════
    device = torch.device("cuda")  
    model = TabularClassifier(
        num_features=len(schema.numerical),
        cat_cardinalities=[20] * len(schema.categorical),
        num_classes=15,
        hidden_dims=[128, 64],
        dropout=0.1,
    ).to(device)
    
    # Creo loss INIZIALE (senza class weighting)
    loss = ClassificationLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)#era 0.001
    
    # ═══════════════════════════════════════════════════════════════════
    # 8️⃣ CREAZIONE DELLO STATE INIZIALE
    # ═══════════════════════════════════════════════════════════════════
    # Lo State è il "contenitore globale" che passa tra tutti gli step
    state = State(
        {
            "device": device,
            "model": model,
            "loss": loss,
            "optimizer": optimizer,
            "seed": 42,
        }
    )
    
    #aggiunto
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3,
        min_lr=1e-6,
        #verbose=False
    )
    #state.set("scheduler", scheduler, object)
    
    # ═══════════════════════════════════════════════════════════════════
    # 9️⃣ ESECUZIONE PREPROCESSING
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*70)
    logger.info("⚙️ FASE 1: PREPROCESSING DATI")
    logger.info("="*70)
 
    preprocessing_pipeline.run(state)
    
    # ═══════════════════════════════════════════════════════════════════
    # 🔟 COSTRUZIONE DATASET E DATALOADER
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*70)
    logger.info("🔧 FASE 2: SETUP DATASET E DATALOADER")
    logger.info("="*70)
    
    setup_pipeline.run(state)
    
    # ═══════════════════════════════════════════════════════════════════
    # 1️⃣1️⃣ CORREZIONE LABEL SHIFT (bug fix LabelMap)
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
    
    # Calcola i pesi bilanciati automaticamente
    class_weights_balanced = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=train_targets
    )
    
    # Smoothing: attenua pesi troppo aggressivi con radice quadrata
    class_weights_smoothed = np.power(class_weights_balanced, 0.25)#np.sqrt(class_weights_balanced)
    
    # Stampa confronto pesi
    logger.info("\n⚖️ Confronto pesi delle classi:")
    for cls in unique_classes:
        logger.info(
            f"   Classe {cls}: balanced={class_weights_balanced[cls]:.2f}, "
            f"smoothed={class_weights_smoothed[cls]:.2f}"
        )
    
    # Converti in tensor PyTorch
    class_weights_tensor = torch.FloatTensor(class_weights_smoothed).to(device)
    logger.info(f"\n✅ Class weights tensor shape: {class_weights_tensor.shape}")
    
    # Aggiorna la loss con i pesi
    loss_weighted = ClassificationLoss(class_weight=class_weights_tensor).to(device)
    state.set("loss", loss_weighted, ClassificationLoss)
    logger.info("🎯 Loss function aggiornata con class weighting!")
    
    # ═══════════════════════════════════════════════════════════════════
    # 1️⃣3️⃣ TRAINING CON EARLY STOPPING 
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*70)
    logger.info("🚀 FASE 4: TRAINING CON EARLY STOPPING")
    logger.info("="*70)
    
    # 🆕 Crea lo step di training con early stopping
    training_step = TrainWithEarlyStopping(
        epoch_pipeline=epoch_pipeline,
        num_epochs=30,              # Numero massimo di epoche
        patience=5,                 # Ferma se nessun miglioramento per 3 epoche
        min_delta=0.001,            # Miglioramento minimo significativo
        checkpoint_dir="c:/Users/simon/OneDrive/Documenti/TESI_UNI/DataSets/Salvataggi/checkpoints",
        save_best_only=True,        # Salva solo il miglior modello
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
        metrics_dir="c:/Users/simon/OneDrive/Documenti/TESI_UNI/DataSets/Salvataggi/classification_exp_report",
        output_dir="c:/Users/simon/OneDrive/Documenti/TESI_UNI/DataSets/Salvataggi/plots",
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
    
    # ═══════════════════════════════════════════════════════════════════
    # 1️⃣5️⃣ FINE!
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*70)
    logger.info("🎉 PIPELINE COMPLETA ESEGUITA CON SUCCESSO!")
    logger.info("="*70)
    logger.info("\n📁 Output prodotti:")
    logger.info("   • Dataset processati: .../processati/")
    logger.info("   • Checkpoint modelli: .../checkpoints/")
    logger.info("   • Report metriche: .../classification_exp_report/")
    logger.info("   • Grafici delle metriche: .../plots/")
    logger.info("\n✅ Tutto pronto per l'analisi!")

   
if __name__ == "__main__":
    main()
