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
from src.idspy.steps.metrics.classification import ClassificationMetrics

from src.idspy.nn.batch import default_collate, Batch
from src.idspy.nn.helpers import get_device
from src.idspy.nn.checkpoints import save_checkpoint, save_weights
from src.idspy.nn.models.classifier import TabularClassifier
from src.idspy.nn.losses.classification import ClassificationLoss


setup_logging() #inizializza il logging: cioè configura il modulo logging di Python
logger = logging.getLogger(__name__)    #crea un logger per questo modulo
set_seeds(42) #imposta il seed per la generazione casuale consentendo la riproducibilità


def main():
    schema = Schema()
    schema.add(["Attack"], ColumnRole.TARGET)
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
    
    # sistema di pubblicazione/sottoscrizione per la gestione degli eventi, che traccia cosa succede ad ogni step della pipeline
    bus = EventBus()
    bus.subscribe(callback=Logger(), event_type=PipelineEvent.BEFORE_STEP)
    # bus.subscribe(callback=Tracer())
    # bus.subscribe(
    #     callback=DataFrameProfiler(),
    #     event_type=PipelineEvent.AFTER_STEP,
    #     predicate=only_id("training_pipeline.load_data"),
    # )

    fit_aware_pipeline = FitAwareObservablePipeline(
        steps=[
            StandardScale(), #standardizza le feature numeriche
            FrequencyMap(max_levels=20), #codifica le classi categoriche in base alla frequenza, parametro max_lv imposta il numero massimo di livelli da considerare
            LabelMap(), #trasforma le etichette di stringa in numeri interi
        ],
        bus=bus,
        name="fit_aware_pipeline",
    )

    preprocessing_pipeline = ObservablePipeline(
        steps=[
            LoadData(
                path_in="c:/Users/simon/OneDrive/Documenti/TESI_UNI/DataSets/dataset_v2/cic_2018_v2.csv",
                schema=schema,
                nrows=10000000
            ),
            DropNulls(),
            #DownsampleToMinority(class_column=schema.columns(ColumnRole.TARGET)[0]),
            # FilterRareLabels(target_column=schema.target, min_count=3000),
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

    #ex training_pipeline = ObservablePipeline(
    setup_pipeline = ObservablePipeline(
        steps=[
            LoadData(path_in="c:/Users/simon/OneDrive/Documenti/TESI_UNI/DataSets/processati/cic_2018_v2.parquet"),
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
            # TrainOneEpoch(),
            # ValidateOneEpoch(in_scope="test", out_scope="test", save_outputs=True),
            # MakePredictions(pred_fn=lambda x: torch.argmax(x, dim=1)),
            # ClassificationMetrics("c:/Users/simon/OneDrive/Documenti/TESI_UNI/DataSets/classification_exp_report"),
        ],
        bus=bus,
        name="setup_pipeline",
    )

    # --- Passi per ogni epoca ---
    epoch_pipeline = ObservablePipeline(
        steps=[
            TrainOneEpoch(),
            ValidateOneEpoch(in_scope="test", out_scope="test", save_outputs=True),
            MakePredictions(pred_fn=lambda x: torch.argmax(x, dim=1)),
            ClassificationMetrics("c:/Users/simon/OneDrive/Documenti/TESI_UNI/DataSets/classification_exp_report"),
            #creo custom pipeline con condizione di terminazione: al posto del for dove runno le epoche
            # questa pipeline presenta l'early stopping interno
        ],
        bus=bus,
        name="epoch_pipeline",
    )
    
    device = torch.device("cuda")   #get_device()
    model = TabularClassifier(
        num_features=len(schema.numerical),
        cat_cardinalities=[20] * len(schema.categorical),
        num_classes=15,
        hidden_dims=[128, 64],
        dropout=0.1,
    ).to(device)
    
    # 1️⃣ Creo loss INIZIALE (senza weighting)
    loss = ClassificationLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 2️⃣ Creo lo state iniziale
    state = State(
        {
            "device": device,
            "model": model,
            "loss": loss,
            "optimizer": optimizer,
            "seed": 42,
        }
    )
  
    # 3️⃣ Eseguo il preprocessing
    preprocessing_pipeline.run(state)
    
    # 4️⃣ Costruisco Setup iniziale (costruisce dataloader e dataset)
    setup_pipeline.run(state)
    
    # 5️⃣ Correggo le label di test (shift -1)
    # Il problema: `LabelMap` fa `codes + 1`, quindi le label sono [1-15] invece di [0-14]!
    # Poi `BuildDataset` fa lo shift per riportarle a [0-14], MA solo per il dataset, non per `test.targets` che viene usato dalle metriche!
    if state.has("train.targets") and state.has("test.targets"):
        train_targets = state.get("train.targets", np.ndarray)
        test_targets = state.get("test.targets", np.ndarray)
        print(f"Train targets --- Prima shift: min={train_targets.min()}, max={train_targets.max()}")
        print(f"Test targets --- Prima shift: min={test_targets.min()}, max={test_targets.max()}")
        
        train_targets = train_targets - 1
        test_targets = test_targets - 1
        
        state.set("train.targets", train_targets, np.ndarray)
        state.set("test.targets", test_targets, np.ndarray)
        
        print(f"Train targets --- Dopo shift: min={train_targets.min()}, max={train_targets.max()} ✅")
        print(f"Test targets --- Dopo shift: min={test_targets.min()}, max={test_targets.max()} ✅")   
    
    # 6️⃣ Calcolo automatico dei pesi di classe
    train_targets = state.get("train.targets", np.ndarray)
    
    unique_classes, class_counts = np.unique(train_targets, return_counts=True)
    
    # 📊 STAMPA la distribuzione
    print("\n📊 Distribuzione classi:")
    total_samples = len(train_targets)
    for cls, count in zip(unique_classes, class_counts):
        percentage = (count / total_samples) * 100
        print(f"   Classe {cls:2d}: {count:10,} samples ({percentage:5.2f}%)")
    
    # ⚖️ CALCOLA i pesi automaticamente; cambiato class_weight in class_weights_balanced
    class_weights_balanced = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=train_targets
    )
    
    # 🎯 A causa di un bilanciamento che ha restituito dei risultati troppo agressivi
    # li attenuiamo con con una radice quadrata
    class_weights_smoothed = np.sqrt(class_weights_balanced)
    
    # 📊 STAMPA i pesi
    # print("\n⚖️ Class weights:")
    # for cls, weight in zip(unique_classes, class_weights):
    #     print(f"   Classe {cls:2d}: weight = {weight:.4f}")
    print("\n⚖️ Confronto pesi:")
    for cls in unique_classes:
        print(f"   Classe {cls}: balanced={class_weights_balanced[cls]:.2f}, "
          f"smoothed={class_weights_smoothed[cls]:.2f}")
    
    
    
    # 7️⃣ Converto in tensor; cambiato class_weights in class_weights_smoothed
    class_weights_tensor = torch.FloatTensor(class_weights_smoothed).to(device)
    print(f"\n✅ Class weights tensor shape: {class_weights_tensor.shape}\n")
    
    # 8️⃣ AGGIORNA la loss con i pesi
    loss_weighted = ClassificationLoss(
        class_weight=class_weights_tensor
    ).to(device)
    
    state.set("loss", loss_weighted, ClassificationLoss)  # ← Sostituisce la loss nello state
    print("🎯 Loss function aggiornata con class weighting!\n")
    
    # 9️⃣ Ora Training per più epoche
    num_epochs = 10  

    for epoch in range(num_epochs):
        print(f"\n🚀 Starting EPOCH {epoch + 1}/{num_epochs}")
        
        # DEBUG: Stampa tutte le chiavi nello state PRIMA della pipeline
        print(f"🔍 DEBUG - Chiavi nello state PRIMA dell'epoca:")
        all_keys = [k for k in dir(state) if not k.startswith('_')]
        for key in all_keys:
            try:
                if state.has(key):
                    print(f"   ✓ {key}")
            except:
                pass
        
        # DEBUG: Verifica dimensioni PRIMA
        if state.has("test.predictions"):
            preds = state.get("test.predictions", np.ndarray)
            print(f"⚠️ WARNING: test.predictions ESISTE prima dell'epoca! Size: {len(preds)}")
        if state.has("test.outputs"):
            outputs = state.get("test.outputs", np.ndarray)
            print(f"⚠️ WARNING: test.outputs ESISTE prima dell'epoca! Size: {len(outputs)}")
        
        # Eseguo l'epoca PRIMA di pulire
        epoch_pipeline.run(state)
        
        # DEBUG: Verifica dimensioni DOPO
        if state.has("test.predictions"):
            preds = state.get("test.predictions", ndarray)
            print(f"📊 DOPO epoca: test.predictions size = {len(preds)}")
        if state.has("test.targets"):
            targets = state.get("test.targets", ndarray)
            print(f"📊 DOPO epoca: test.targets size = {len(targets)}")
        
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
            "train.epoch", "test.epoch", "val.epoch",
        ]
        
        for key in keys_to_clean:
            if state.has(key):
                logger.debug(f"🧹 Cleaning state key: {key}")
                state.delete(key)
    
    print("\n✅ Training completato con successo!")

    # Controllo DataFrame finale
    # df_filtered = state.get("data", pd.DataFrame)
    # print("\n[Main] Conteggio finale classi dopo filtro:")
    # print(df_filtered[schema.target].value_counts())

    # # stampo cosa ha salvato lo step
    # print("\n[Main] Classi mantenute:", state.get("kept_classes", list))
    # print("[Main] Classi eliminate:", state.get("dropped_label_counts", dict))
    # print("[Main] Numero righe eliminate:", state.get("filter_rare_labels.dropped_rows", int))


    # training_pipeline.run(state)
    # print(state.get("test.metrics", dict))
    # for step in preprocessing_pipeline.steps:
    #     print(step.__class__.__name__, step.name)

    #     # -------------------------
    # # Controlli di sicurezza finale
    # # -------------------------
    # counts = df_filtered[schema.target].value_counts()

    # # 1️⃣ Controllo che tutte le etichette abbiano almeno min_count esempi
    # min_count = 3000
    # if (counts < min_count).any():
    #     print("[Errore] Alcune etichette hanno meno di 3000 esempi:")
    #     print(counts[counts < min_count])
    # else:
    #     print("[OK] Tutte le etichette hanno almeno 3000 esempi.")

    # # 2️⃣ Verifica che le etichette eliminate non siano presenti
    # dropped_labels = state.get("dropped_label_counts", dict).keys()
    # remaining_labels = df_filtered[schema.target].unique()
    # for label in dropped_labels:
    #     assert label not in remaining_labels, f"[Errore] Etichetta {label} ancora presente!"
    # print("[OK] Tutte le etichette eliminate non sono più presenti.")

   
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