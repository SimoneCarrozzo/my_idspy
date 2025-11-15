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
from src.idspy.events.handlers.logging import Logger, DataFrameProfiler

from src.idspy.steps.io.saver import SaveData
from src.idspy.steps.io.loader import LoadData
from src.idspy.steps.builders.dataloader import BuildDataLoader
from src.idspy.steps.builders.dataset import BuildDataset
from src.idspy.steps.transforms.adjust import DropNulls, FilterRareLabels
from src.idspy.steps.transforms.map import FrequencyMap, LabelMap
from src.idspy.steps.transforms.scale import StandardScale
from src.idspy.steps.transforms.split import (
    AssignSplitPartitions,
    StratifiedSplit,
    AssignSplitTarget,
)
from src.idspy.steps.model.training import TrainOneEpoch
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
    # sistema di pubblicazione/sottoscrizione per la gestione degli eventi, che traccia cosa succede ad ogni step della pipeline
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
                path_in="c:/Users/simon/OneDrive/Documenti/TESI_UNI/SetUp/dataset_v2/cic_2018_v2.csv",
                schema=schema,
                nrows=10000000
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
    #ex training_pipeline = ObservablePipeline(
    setup_pipeline = ObservablePipeline(
        steps=[
            LoadData(path_in="c:/Users/simon/OneDrive/Documenti/TESI_UNI/SetUp/dataset_processati/cic_2018_v2.parquet"),
            AssignSplitPartitions(),
            #aggiunto per assegnare i target corretti anche al train
            AssignSplitTarget(in_scope="data", out_scope="train"),
            # prima c'era solo questo
            AssignSplitTarget(in_scope="data", out_scope="test"),
            BuildDataset(out_scope="train"),
            BuildDataLoader(
                in_scope="train",
                out_scope="train",
                batch_size=512,
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
            ),
        ],
        bus=bus,
        name="setup_pipeline",
    )

    
    # ═══════════════════════════════════════════════════════════════════
    # 6️⃣ CONFIGURAZIONE MODELLO E LOSS
    # ═══════════════════════════════════════════════════════════════════    
    device = torch.device("cuda")   #get_device()
    model = TabularClassifier(
        num_features=len(schema.numerical),
        cat_cardinalities=[20] * len(schema.categorical),
        num_classes=15,
        hidden_dims=[128, 64],
        dropout=0.1,
    ).to(device)
    
    #  Creo loss INIZIALE (senza weighting)
    loss = ClassificationLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ═══════════════════════════════════════════════════════════════════
    # 7️⃣ CREAZIONE DELLO STATE INIZIALE
    # ═══════════════════════════════════════════════════════════════════
    # Creo lo state iniziale
    state = State(
        {
            "device": device,
            "model": model,
            "loss": loss,
            "optimizer": optimizer,
            "seed": 42,
        }
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
    
    # Leggi i nomi delle classi direttamente dal CSV raw (veloce: solo 1000 righe)
    csv_path = "c:/Users/simon/OneDrive/Documenti/TESI_UNI/SetUp/dataset_v2/cic_2018_v2.csv"
    try:
        sample_data = pd.read_csv(csv_path, nrows=10000000) #1000
        original_classes = sorted(sample_data["Attack"].unique())
        class_names = [str(c) for c in original_classes]
        logger.info(f"✅ Class names from raw data: {class_names}")
    except Exception as e:
        logger.warning(f"⚠️ Impossibile leggere da CSV: {e}")
        logger.info("   Utilizzo fallback: auto-generated class names")
        train_targets = state.get("train.targets", np.ndarray)
        num_classes = len(np.unique(train_targets))
        class_names = [f"Attack_Type_{i}" for i in range(num_classes)]
        logger.info(f"✅ Class names (fallback): {class_names}")
    
    logger.info(f"📝 Nomi classi finali: {class_names}\n")
    
    
    # ═══════════════════════════════════════════════════════════════════
    # 🔟 CORREZIONE LABEL SHIFT (bug fix LabelMap)
    # ═══════════════════════════════════════════════════════════════════
    # Il problema: `LabelMap` fa `codes + 1`, quindi le label sono [1-15] invece di [0-14]!
    # Poi `BuildDataset` fa lo shift per riportarle a [0-14], MA solo per il dataset, non per `test.targets` che viene usato dalle metriche!
    if state.has("train.targets") and state.has("test.targets"):
        train_targets = state.get("train.targets", np.ndarray)
        test_targets = state.get("test.targets", np.ndarray)
        logger.info(f"Train targets --- Prima shift: min={train_targets.min()}, max={train_targets.max()}")
        logger.info(f"Test targets --- Prima shift: min={test_targets.min()}, max={test_targets.max()}")
        
        train_targets = train_targets - 1
        test_targets = test_targets - 1
        
        state.set("train.targets", train_targets, np.ndarray)
        state.set("test.targets", test_targets, np.ndarray)
        
        logger.info(f"Train targets --- Dopo shift: min={train_targets.min()}, max={train_targets.max()} ✅")
        logger.info(f"Test targets --- Dopo shift: min={test_targets.min()}, max={test_targets.max()} ✅")   
    
    # ═══════════════════════════════════════════════════════════════════
    # 1️⃣1️⃣ PIPELINE EPOCA - Operazioni per ogni singola epoca
    # ═══════════════════════════════════════════════════════════════════
    # epoch_pipeline = ObservablePipeline(
    #     steps=[
    #         TrainOneEpoch(),
    #         ValidateOneEpoch(in_scope="test", out_scope="test", save_outputs=True),
    #         MakePredictions(pred_fn=lambda x: torch.argmax(x, dim=1)),
    #         ClassificationMetrics("c:/Users/simon/OneDrive/Documenti/TESI_UNI/DataSets/classification_exp_report"),
    #         #creo custom pipeline con condizione di terminazione: al posto del for dove runno le epoche
    #         # questa pipeline presenta l'early stopping interno
    #     ],
    #     bus=bus,
    #     name="epoch_pipeline",
    # )         -----------------> CAMBIATO IL 13/11
    
    # OTTENGO AUTOMATICAMENTE IL NUMERO DI CLASSI  --> AGGIUNTO 13/11
    # train_targets = state.get("train.targets", np.ndarray)
    # num_classes = len(np.unique(train_targets))
    # class_names = [f"Class_{i}" for i in range(num_classes)]

    # logger.info(f"\n📊 Numero di classi rilevate: {num_classes}")
    # logger.info(f"📝 Nomi classi: {class_names}\n")
    
    import os
    log_dir = "c:/Users/simon/OneDrive/Documenti/TESI_UNI/SetUp/Modelli_Salvati/logs/v0_smoothed"    
    os.makedirs(log_dir, exist_ok=True)

    epoch_pipeline = ObservablePipeline(
        steps=[
            TrainOneEpoch(
                log_dir=log_dir,           # AGGIUNTO
                log_prefix="train",        # AGGIUNTO
                save_history=True,         # Opzionale
                checkpoint_dir=f"{log_dir}/checkpoints", #AGGIUNTO
            ),
            ValidateOneEpoch(
                log_dir=log_dir,           # AGGIUNTO
                log_prefix="val",          # AGGIUNTO 
                in_scope="test",           # ← mantieni "test" se usi test set
                out_scope="test",
                save_outputs=True,
            ),
            MakePredictions(pred_fn=lambda x: torch.argmax(x, dim=1)),
            ClassificationMetrics(
                log_dir=log_dir,                   # CAMBIO path
                log_prefix="test",                 # AGGIUNTO
                class_names=class_names,           # AGGIUNTO
                in_scope="test",
                out_scope="test",
                save_confusion_matrix=True,  # ✅ Salva metriche
                save_classification_report=True,
                save_f1_per_class_plot = True,
            ),
        ],
        bus=bus,
        name="epoch_pipeline",
    )
    
    # ═══════════════════════════════════════════════════════════════════
    # 1️⃣2️⃣ Calcolo automatico dei pesi di classe
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*70)
    logger.info("⚖️ Calcolo automatico dei pesi di classe")
    logger.info("="*70)
    
    train_targets = state.get("train.targets", np.ndarray)
    
    unique_classes, class_counts = np.unique(train_targets, return_counts=True)
    
    # 📊 STAMPA la distribuzione
    logger.info("\n📊 Distribuzione classi:")
    total_samples = len(train_targets)
    for cls, count in zip(unique_classes, class_counts):
        percentage = (count / total_samples) * 100
        logger.info(f"   Classe {cls:2d}: {count:10,} samples ({percentage:5.2f}%)")
    
    """ # ⚖️ CALCOLA i pesi automaticamente; cambiato class_weight in class_weights_balanced
    class_weights_balanced = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=train_targets
    )
    
    # 🎯 A causa di un bilanciamento che ha restituito dei risultati troppo agressivi
    # li attenuiamo con con una radice quadrata
    class_weights_smoothed = np.sqrt(class_weights_balanced)
    
    # 📊 STAMPA i pesi
    # logger.info("\n⚖️ Class weights:")
    # for cls, weight in zip(unique_classes, class_weights):
    #     logger.info(f"   Classe {cls:2d}: weight = {weight:.4f}")
    logger.info("\n⚖️ Confronto pesi:")
    for cls in unique_classes:
        logger.info(f"   Classe {cls}: balanced={class_weights_balanced[cls]:.2f}, "
          f"smoothed={class_weights_smoothed[cls]:.2f}")
    
    
    
    # 7️⃣ Converto in tensor; cambiato class_weights in class_weights_smoothed
    class_weights_tensor = torch.FloatTensor(class_weights_smoothed).to(device)
    logger.info(f"\n✅ Class weights tensor shape: {class_weights_tensor.shape}\n")
    
    # 8️⃣ AGGIORNA la loss con i pesi
    loss_weighted = ClassificationLoss(
        class_weight=class_weights_tensor
    ).to(device)
    
    state.set("loss", loss_weighted, ClassificationLoss)  # ← Sostituisce la loss nello state
    logger.info("🎯 Loss function aggiornata con class weighting!\n")
     """
    # 🔧 DEFINISCO L'APPROCCIO DI CLASS WEIGHTING
    weighting_strategy = "smoothed"  #  "no_weight" (run 1)
                                       # "balanced" (run 2)
                                       # "smoothed" (run 3)
    
    if weighting_strategy == "no_weight":
        logger.info("\n✅ STRATEGIA: No Class Weighting")
        loss_fn = ClassificationLoss().to(device)
        state.set("loss", loss_fn, ClassificationLoss)
        
    elif weighting_strategy == "balanced":
        logger.info("\n✅ STRATEGIA: Balanced Class Weighting")
        class_weights_balanced = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=train_targets
        )
        
        logger.info("\n⚖️ Class weights (balanced):")
        for cls in unique_classes:
            logger.info(f"   Classe {cls:2d}: weight = {class_weights_balanced[cls]:.4f}")
        
        class_weights_tensor = torch.FloatTensor(class_weights_balanced).to(device)
        loss_fn = ClassificationLoss(class_weight=class_weights_tensor).to(device)
        state.set("loss", loss_fn, ClassificationLoss)
        logger.info(f"✅ Class weights tensor shape: {class_weights_tensor.shape}\n")
        
    elif weighting_strategy == "smoothed":
        logger.info("\n✅ STRATEGIA: Smoothed Class Weighting (sqrt)")
        class_weights_balanced = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=train_targets
        )
        
        # 🎯 Applica smoothing con radice quadrata
        class_weights_smoothed = np.sqrt(class_weights_balanced)
        
        logger.info("\n⚖️ Confronto pesi:")
        for cls in unique_classes:
            logger.info(f"   Classe {cls:2d}: balanced={class_weights_balanced[cls]:.4f}, "
                       f"smoothed={class_weights_smoothed[cls]:.4f}")
        
        class_weights_tensor = torch.FloatTensor(class_weights_smoothed).to(device)
        loss_fn = ClassificationLoss(class_weight=class_weights_tensor).to(device)
        state.set("loss", loss_fn, ClassificationLoss)
        logger.info(f"✅ Class weights tensor shape: {class_weights_tensor.shape}\n")
    
    else:
        raise ValueError(f"Strategia sconosciuta: {weighting_strategy}")
    
    logger.info(f"🎯 Loss function configurata: {weighting_strategy}\n")
    # ═══════════════════════════════════════════════════════════════════
    # 1️⃣3️⃣ TRAINING  
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*70)
    logger.info("🚀 TRAINING ")
    logger.info("="*70)
    
    num_epochs = 10  

    for epoch in range(num_epochs):
        logger.info(f"\n🚀 Starting EPOCH {epoch + 1}/{num_epochs}")
        
        # AGGIUNTO 13/11
        state.set("epoch", epoch, int)
        
        # DEBUG: Stampa tutte le chiavi nello state PRIMA della pipeline
        logger.info(f"🔍 DEBUG - Chiavi nello state PRIMA dell'epoca:")
        all_keys = [k for k in dir(state) if not k.startswith('_')]
        for key in all_keys:
            try:
                if state.has(key):
                    logger.info(f"   ✓ {key}")
            except:
                pass
        
        # DEBUG: Verifica dimensioni PRIMA
        if state.has("test.predictions"):
            preds = state.get("test.predictions", np.ndarray)
            logger.info(f"⚠️ WARNING: test.predictions ESISTE prima dell'epoca! Size: {len(preds)}")
        if state.has("test.outputs"):
            outputs = state.get("test.outputs", np.ndarray)
            logger.info(f"⚠️ WARNING: test.outputs ESISTE prima dell'epoca! Size: {len(outputs)}")
        
        # Eseguo l'epoca PRIMA di pulire
        epoch_pipeline.run(state)
        
        # DEBUG: Verifica dimensioni DOPO
        if state.has("test.predictions"):
            preds = state.get("test.predictions", ndarray)
            logger.info(f"📊 DOPO epoca: test.predictions size = {len(preds)}")
        if state.has("test.targets"):
            targets = state.get("test.targets", ndarray)
            logger.info(f"📊 DOPO epoca: test.targets size = {len(targets)}")
        
        # Salvo il modello
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pt")
        logger.info(f"✅ Epoch {epoch + 1} completata e modello salvato")
        
        # PULIZIA DOPO: rimuovo TUTTE le chiavi che possono accumularsi
        # Questo previene l'accumulo tra le epoche
        keys_to_clean = [
            # Output dei modelli (temporanei)
            "train.model", "test.model", "val.model",
            # Output delle predizioni/validazione (temporanei)
            "train.outputs", "test.outputs", "val.outputs",
            # Predizioni (temporanee)
            "train.predictions", "test.predictions", "val.predictions",
            # Loss e metriche (temporanee)
            "train.loss", "test.loss", "val.loss",
            "train.metrics", "test.metrics", "val.metrics",
            # History (si accumula)
            "train.history", "test.history", "val.history",
            # Epoch counter (si accumula)
            # ################################"train.epoch", "test.epoch", "val.epoch",
        ]
        
        for key in keys_to_clean:
            if state.has(key):
                logger.debug(f"🧹 Cleaning state key: {key}")
                state.delete(key)
        
    
    logger.info("\n🔒 Chiusura TensorBoard writers...")

    # Recupera i writer dalle pipeline steps e chiudili
    for step in epoch_pipeline.steps:
        if hasattr(step, 'writer') and step.writer is not None:
            step.writer.close()
            logger.info(f"   ✅ Chiuso writer: {step.name}")
                
    logger.info("\n✅ Training completato con successo!")
    
    # DOPO il loop:
    logger.info("\n🧹 Pulizia finale dello state dopo training...")
    final_cleanup_keys = ["epoch"]  # Pulisci solo dopo
    for key in final_cleanup_keys:
        if state.has(key):
            state.delete(key)
            logger.debug(f"✅ Rimosso: {key}")

    # ═══════════════════════════════════════════════════════════════════
    # 1️⃣4️⃣ VISUALIZZAZIONE METRICHE CON MATPLOTLIB 
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*70)
    logger.info("📊 GENERAZIONE GRAFICI METRICHE")
    logger.info("="*70)

    # Crea step plot
    plot_step = PlotMetrics(
        metrics_dir=log_dir,  # Stessa dir dove sono i file TensorBoard
        output_dir=f"{log_dir}/plots",  # Dove salvare i grafici
        metrics_to_plot=["accuracy", "precision", "recall", "f1_macro", "f1_micro", "f1_weighted"],
        figsize=(12, 8),
        dpi=300,
        show_plots=False,  # True se vuoi vederli interattivamente
        in_scope="test",       
        out_scope="plots",     
        name="plot_metrics"
    )

    # Esegui generazione grafici
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

#plot f1 a barre con matplotlib

#CONSIDERAZIONI SULLE METRICHE DOPO 10 EPOCHE:
# Accuracy: 99.5%
# F1 Macro: 78.6%
# F1 Micro: 99.5%
# F1 Weighted: 99.4%
# Precision: 96%
# Recall: 76%
# il modello funziona molto bene, ma è presente uno sbilanciamento nelle classi:
# 
# F1 Micro (99.5%) >> F1 Macro (78.6%) ==> ciò significa che alcune classi sono
# molto più frequenti di altre, e il modello predice benissimo quelle maggioritarie,
# ma ha più difficoltà con le classi minoritarie

# Precision (96%) > Recall (76%) ==> ciò significa che il modello è conservativo, cioè
# quando predice una classe, è quasi sempre corretto (poche false positive),
# ma manca alcuni esempi (false negative più alti), preferisce "astenersi" piuttosto che sbagliare

#possibili miglioramenti:
#   Split Manuale nel preprocessing
# 
    # StratifiedSplit(
    #     class_column=schema.target,
    #     train_size=0.7,  
    #     val_size=0.15,   
    #     test_size=0.15   
    # )
    
    
# ---------------------------------------------------------------------#
# class weighting nella loss per bilanciare le classi minoritarie
#nonostante il tentativo di bilanciare le classi più rare, mediante i pesi
# i risultati mostrati dalle metriche indicano un bilanciamento troppo aggressivo, 
#che ha portato:
# Accuracy: 56,3%
# F1 Macro: 73,3%
# F1 Micro: 56,3%
# F1 Weighted: 70%
# Precision: 79,5%
# Recall: 78,2%


# --------------------------------------------------#
# Smooted class weights con radice quadrata

# │ Accuracy   95.8%     
# │ F1 Macro   79.0%     
# │ F1 Micro   95.8%  
# │ F1 Weighted  99.5% 
# │ Precision  88.5% 
# │ Recall    77.2%  

# ##  Interpretazione dei Risultati
# ##  Smoothed Class Weighting è la giusta via di mezzo!

# Perché:
# 1. F1 Macro 79.0% → +0.4% vs baseline (MEGLIO!)
# 2. Accuracy 95.8% → Solo -3.7% vs baseline (accettabile!)
# 3. F1 Weighted 99.5% → UGUALE al baseline (perfetto!)
# 4. Recall 77.2% → +1.2% vs baseline (meglio!)
# 5. Precision 88.5% → -7.5% vs baseline (trade-off accettabile)

# Trade-off Precision vs Recall:

# Baseline (no weighting):
# - Precision: 96% → "Quando predico una classe, sono quasi sempre corretto"
# - Recall: 76%   → "Ma perdo il 24% degli esempi (specialmente classi rare)"

# Smoothed (con weighting):
# - Precision: 88.5% → "Sono un po' meno sicuro (faccio più tentativi)"
# - Recall: 77.2%   → "Ma trovo più esempi delle classi rare!"

# F1 Macro: ovvero la media delle F1 di TUTTE le classi (anche le rare)

# Baseline:  78.6% → Alcune classi rare hanno F1 basso
# Smoothed:  79.0% → Classi rare hanno F1 più alto! 

# 📈 Analisi dei Grafici

# 🟢 Trend Positivo:
# - Tutte le metriche crescono nel tempo
# - Convergenza intorno all'epoca 7-8
# - Salto finale nell'ultima epoca (da 75% a 79% F1 macro)

# 🟡 Recall in Calo Finale:
# Epoch 1-2: Recall ~78.4%  --> recall alta perché il modello "spara nel mucchio";
# Epoch 3-9: Recall ~77.8%  --> recall si stabilizza poichè comincia la fase di apprendimento;
# Epoch 10:  Recall ~77.2%  --> recall che cala leggermente perchè il modello diventa più "cauto" e preciso;
# Dunque, è normale dato che il modello sta bilanciando precision/recall:
# - Non vuole fare troppi falsi positivi (↑ precision)
# - Ma vuole trovare abbastanza esempi (↑ recall)

# 📌 Punti Chiave da Evidenziare 
# 1) il Class Weighting Funziona se bilanciato: "L'uso di class weighting con smoothing (radice quadrata) ha permesso 
# di migliorare l'F1 Macro dal 78.6% al 79.0%, con una riduzione accettabile dell'accuracy dal 99.5% al 95.8%."

# 2) Trade-off Precision/Recall: "Il class weighting ha spostato il bilancio da alta precision (96%) e bassa recall (76%)
# a precision più moderata (88.5%) e recall migliorata (77.2%), risultando in una migliore detection delle classi minoritarie."

# 3) Importanza della Tecnica di Smoothing: "Il class weighting 'balanced' standard ha prodotto risultati inaccettabili 
# come accuracy 56%, mentre la tecnica di smoothing con  radice quadrata ha mantenuto le performance elevate migliorando il 
# bilanciamento tra classi.


# 📊 Tabella riepilogativa
# ┌─────────────────────────────────────────────────────────────┐
# │          CONFRONTO APPROCCI DI CLASS BALANCING              │
# ├────────────────────┬─────────┬─────────┬─────────┬──────────┤
# │ Metrica            │ No Wgt  │ Full Wgt│ Smoothed│ Δ (%)    │
# ├────────────────────┼─────────┼─────────┼─────────┼──────────┤
# │ Accuracy           │ 99.5    │ 56.2    │ 95.8    │ -3.7     │
# │ F1 Macro           │ 78.6    │ 73.2    │ 79.0    │ +0.4 ✅  │
# │ F1 Micro           │ 99.5    │ 56.2    │ 95.8    │ -3.7     │
# │ F1 Weighted        │ 99.4    │ 70.3    │ 99.5    │ +0.1 ✅  │
# │ Precision          │ 96.0    │ 79.5    │ 88.5    │ -7.5     │
# │ Recall             │ 76.0    │ 78.2    │ 77.2    │ +1.2 ✅  │
# ├────────────────────┴─────────┴─────────┴─────────┴──────────┤
# │ CONCLUSIONE: Smoothed class weighting ottimizza il          │
# │ bilanciamento tra classi maggioritarie e minoritarie,        │
# │ migliorando F1 Macro con perdita minima di accuracy.        │
# └──────────────────────────────────────────────────────────────┘
#
# QUANTO FATTO FINORA:
# 0) training di un modello supervisionato locale;

# CIO' CHE C'E' DA FARE:
# voglio un dataset per ogni indirizzo ip;
#1) integrare approccio federated learning:

# 1.1) ho un dataset per ogni host (--> quindi aggiungere nel preprocessing lo step che crea dei sottodataset
# contenenti il traffico in/out relativo al singolo host),
# identificato con gli ip più frequenti dal dataset di partenza (base= 5-10); 
# 1.2) ogni host addestra un modello locale sul proprio dataset;
# 1.3) ogni host invia i pesi del modello ad un server centrale;
# 1.4) il server aggrega i pesi (es. media pesata) e invia il modello aggiornato a tutti gli host;
# 1.5) confronto modelli locali e aggregato globale su test set aggregato su tutti gli host;

# 2) dopo aver fatto considerazioni sul punto 1, implemento un "clustering" in cui raggruppo i pesi sulla similarità dell'ultimo layer (--> 
# questo perchè c'è un teorema che dice che 2 reti neurali sono simili se i pesi dell'ultimo layer sono simili; a questo punto, io non 
# faccio altro che aggregare i pesi dell'ultimo layer di un host che sono simili ai pesi dell'ultimo layer di altri host);
# a questo punto, il server mantiene lo stato di più modelli aggregati, per poi restituire il modello corretto all'host che 
# appartiene alla specifica aggregazione;
# COME CAMBIANO I RISULTATI?  


# Metrica            Smoothed
# ├────────────────────
# │ Accuracy           95.8    
# │ F1 Macro           79.0    
# │ F1 Micro           95.8
# │ F1 Weighted        99.5
# │ Precision          88.5
# │ Recall             77.2 