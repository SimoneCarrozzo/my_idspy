<<<<<<< HEAD
from typing import Optional, Dict       #Optional indica che una variabile può essere di un certo tipo oppure None finchè non viene inizializzata;  
                                        #Dict è un dizionario
=======
from typing import Optional, Dict, Any

>>>>>>> upstream/main
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype   #per definire tipi di dati categorici ordinati in pandas

from ...core.step import FitAwareStep, Step
from ...core.state import State
<<<<<<< HEAD
from ...core.step import FitAwareStep
from ...data.partition import PartitionName
from ...data.tab_accessor import reattach_meta  #funzione per riattaccare informazioni aggiuntive al DataFrame dopo una trasformazione
                                                #preserva lo schema/metadati/partizioni del DataFrame qunado si lavora su una copia del DataFrame
#Contesto
#Queste due classi trasformano variabili categoriali e target in numeri, usando informazioni imparate solo sul train (come si deve fare).
###FrequencyMap → codifica le colonne categoriali (features).
###LabelMap → codifica la colonna target (label), gestendo sia il caso binario che multiclasse.
#Entrambe ereditano da FitAwareStep: prima fit_impl(state) (calcola le regole sul train), poi run(state) (applica le regole a tutto il dataset).
=======
>>>>>>> upstream/main


#Mappare le categorie in ogni colonna categoriale in codici interi basati sulla frequenza (la categoria più frequente ottiene il codice più basso)
class FrequencyMap(FitAwareStep):
    """Map categorical columns to frequency-rank codes."""

    def __init__(
        self,
        max_levels: Optional[int] = None,
        default: int = 0,
        in_scope: str = "data",
        out_scope: str = "data",
        name: Optional[str] = None,
    ) -> None:
        self.max_levels = max_levels
        self.default = default
        self.cat_types: Dict[str, CategoricalDtype] = {}    #dizionario che mappa il nome di ogni colonna categoriale al suo CategoricalDtype 
                                                            #(definisce le categorie ordinate imparate durante il fitting)
        super().__init__(
            name=name or "frequency_map",
            in_scope=in_scope,
            out_scope=out_scope,
        )

<<<<<<< HEAD
    def fit_impl(self, state: State) -> None:   #tale metodo impara le categorie ordinate per ogni colonna categoriale nel training set
        """Infer ordered categories by frequency from train split."""
        dataframe = state[self.dataframe_in]
        validate_instance(dataframe, pd.DataFrame, self.name)

        train_df = dataframe.tab.train  #estrae il DataFrame della partizione di training
        self.cat_types.clear()          #pulisce il dizionario delle categorie imparate
=======
    @Step.requires(root=pd.DataFrame)
    def fit_impl(self, state: State, root: pd.DataFrame) -> None:
        """Infer ordered categories by frequency from train split."""
        train_df = root.tab.train
        self.cat_types.clear()
>>>>>>> upstream/main

        cat_cols = train_df.tab.categorical.columns #estrae i nomi delle colonne categoriali
        for col in cat_cols:                    #per ogni colonna categoriale
            vc = train_df[col].value_counts(dropna=False)   #calcola il conteggio dei valori di ogni categoria, includendo NaN
            if vc.empty:
                continue

            if self.max_levels is None: #se non è specificato un limite massimo di livelli, prende tutte le categorie
                cats = vc.index.tolist()    #lista delle categorie ordinate per frequenza
            else:  #altrimenti prende solo le prime max_levels categorie più frequenti
                cats = vc.head(self.max_levels).index.tolist()  

            self.cat_types[col] = CategoricalDtype(categories=cats, ordered=True)   #oridne significativo: la categoria più frequente ha il codice più basso
            #crea un CategoricalDtype con le categorie ordinate e lo salva nel dizionario

    @Step.requires(root=pd.DataFrame)
    @Step.provides(root=pd.DataFrame, cat_mapping=dict)
    def run(self, state: State, root: pd.DataFrame) -> None:
        """Apply learned frequency mapping to categorical columns."""
<<<<<<< HEAD
        dataframe = state[self.dataframe_in]
        validate_instance(dataframe, pd.DataFrame, self.name)

        out = dataframe.copy()  #lavora su una copia del DataFrame per non modificare l'originale

        # Early exit if no categorical mappings learned
        #se non sono state imparate categorie (es. nessuna colonna categoriale nel train), salva il DataFrame così com'è e ritorna
        if not self.cat_types:  
            state[self.dataframe_out] = reattach_meta(dataframe, out)
            state["mapping.categorical"] = self.cat_types
            return

        cat_cols = dataframe.tab.categorical.columns    #estrae i nomi delle colonne categoriali e per ogni colonna
        for col in cat_cols:
            if col not in self.cat_types or col not in out.columns: #se non è stata imparata una mappatura per questa colonna o la colonna non esiste nel DataFrame, salta
                continue
                #altrimenti converte la colonna al tipo categorico imparato, forzando le categorie nell'ordine appreso dal fitting
            s = out[col].astype(self.cat_types[col])
            codes = s.cat.codes #estrae i codici interi associati a ogni categoria (quella + frequente codice 0, la seconda più frequente 1, ecc.)
            out[col] = np.where(codes != -1, codes + 1, self.default).astype("int32") #sostituisce la colonna con i codici, mappando -1 (categorie non viste) al valore default

        state[self.dataframe_out] = reattach_meta(dataframe, out)
        state["mapping.categorical"] = self.cat_types   #salva il dizionario delle mappature imparate nello State con la chiave "mapping.categorical"
=======

        # Early exit if no categorical mappings learned
        if not self.cat_types:
            return {"root": root, "cat_mapping": {}}

        cat_cols = root.tab.categorical.columns
        for col in cat_cols:
            if col not in self.cat_types or col not in root.columns:
                continue

            s = root[col].astype(self.cat_types[col])
            codes = s.cat.codes
            root[col] = np.where(codes != -1, codes + 1, self.default).astype("int32")

        return {"root": root, "cat_mapping": self.cat_types}

>>>>>>> upstream/main

#tale classe mappa la colonna target in codici interi, gestendo sia il caso binario (con un benign_tag) che multiclasse (ordinal categories 1,2,3...)
class LabelMap(FitAwareStep):
    """Encode `target`: binary with `benign_tag`, else ordinal categories."""

    def __init__(
        self,
<<<<<<< HEAD
        dataframe_in: str = "data.root",
        dataframe_out: Optional[str] = None,
        benign_tag: Optional[str] = None,   #se specificato => classificazione binaria, sennò multiclasse
=======
        benign_tag: Optional[str] = None,
>>>>>>> upstream/main
        default: int = -1,
        in_scope: str = "data",
        out_scope: str = "data",
        name: Optional[str] = None,
    ) -> None:
        self.benign_tag = benign_tag
        self.default = default
        self.cat_types: Optional[CategoricalDtype] = None

        super().__init__(
            name=name or "target_map",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    @Step.requires(root=pd.DataFrame)
    def fit_impl(self, state: State, root: pd.DataFrame) -> None:
        """Learn ordered categories for the target col (if not binary)."""

        # Early exit for binary case
        #se il tag è specificato, non serve imparare categorie ordinate, ritorna subito
        if self.benign_tag is not None:
            self.cat_types = None
            return
<<<<<<< HEAD
        #altrimenti impara le categorie ordinate dal training set
        train_df = dataframe.tab.train #estrae il DataFrame della partizione di training
        tgt_cols = train_df.tab.target.columns  #estrae i nomi delle colonne target
        if len(tgt_cols) != 1:  #se non c'è esattamente una colonna target, solleva un errore
            raise ValueError(
                f"Expected exactly 1 target column, found {len(tgt_cols)}: {tgt_cols}"
            )
        #altrimenti prende il nome della colonna target
        tgt_col = tgt_cols[0]
        vc = train_df[tgt_col].value_counts(dropna=False)   #conta le occorrenze di ogni categoria, includendo NaN
        self.cat_types = CategoricalDtype(categories=vc.index.tolist(), ordered=True)  #crea un CategoricalDtype con le categorie ordinate per frequenza
=======

        train_df = root.tab.train
        tgt_col = train_df.tab.schema.target

        vc = train_df[tgt_col].value_counts(dropna=False)
        self.cat_types = CategoricalDtype(categories=vc.index.tolist(), ordered=True)
>>>>>>> upstream/main

    @Step.requires(root=pd.DataFrame)
    @Step.provides(root=pd.DataFrame, target_mapping=CategoricalDtype | None)
    def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:
        tgt_col = root.tab.schema.target

<<<<<<< HEAD
        tgt_cols = dataframe.tab.target.columns #estrae i nomi delle colonne target
        if len(tgt_cols) != 1:  #se non c'è esattamente una colonna target, solleva un errore
            raise ValueError(
                f"Expected exactly 1 target column, found {len(tgt_cols)}: {tgt_cols}"
            )
        tgt_col = tgt_cols[0]   #prende il nome della colonna target
        prev = dataframe[tgt_col].copy()    #copia la colonna target originale per conservarla come "original_{tgt_col}"
=======
        prev = root[tgt_col].copy()
>>>>>>> upstream/main

        if self.benign_tag is not None: #se è specificato il benign_tag, esegue la codifica binaria
            tgt = (prev == self.benign_tag).astype("int32")
<<<<<<< HEAD
            tgt = tgt.where(tgt == 0, 1)    #mappa True a 1 (benigno) e False a 0 (maligno)
        else:   #altrimenti esegue la codifica multiclasse  
            if self.cat_types is None:  #se non sono state imparate categorie ordinate, solleva un errore
                raise RuntimeError("LabelMap was not fitted with category types.")
                
                #converte la colonna target al tipo categorico imparato, forzando le categorie nell'ordine appreso dal fitting
            s = prev.astype(self.cat_types) 
            codes = s.cat.codes #estrae i codici interi associati a ogni categoria
=======
            tgt = tgt.where(tgt == 0, 1)
        else:
            s = prev.astype(self.cat_types)
            codes = s.cat.codes
>>>>>>> upstream/main
            tgt = pd.Series(
                np.where(codes != -1, codes, self.default).astype("int32"),
                index=s.index,
                name=tgt_col,
            )   #sostituisce la colonna con i codici, mappando -1 (categorie non viste) al valore default

<<<<<<< HEAD
        dataframe[f"original_{tgt_col}"] = prev #mantiene una copia della colonna target originale come "original_{tgt_col}"
        dataframe.tab.target = tgt  #sostituisce la colonna target con i codici interi (0/1 per binario, 1,2,3... per multiclasse)
        state[self.dataframe_out] = dataframe   #salva il DataFrame trasformato nello State usando la chiave specificata in dataframe_out
        state["mapping.target"] = self.cat_types    #salva il CategoricalDtype (o None se binario) nello State con la chiave "mapping.target"
        
# Esempio concreto
# Train target counts:

# 'Warrior' (15), 'Support' (9), 'Scout' (4), 'Leader' (2) → categorie in questo ordine.
# Mapping:

# 'Warrior' → codice 0 + 1 = 1

# 'Support' → 2

# 'Scout' → 3

# 'Leader' → 4

# Target non visto → default (es. -1): utile per segnalare errori o dati anomali.
# Se invece benign_tag='Benign', allora:
# row con 'Benign' → 1, tutte le altre → 0.

# Differenze importanti tra FrequencyMap e LabelMap
### Default value: FrequencyMap usa default 0 (utile per features: 0 = other/missing), mentre 
##LabelMap usa default -1 (utile per target: -1 = segnale di errore/unknown).
### Conservazione originale: LabelMap crea original_<target> per tenere la label testuale prima della codifica.
### Uso di ordered categories: in entrambi i casi l’ordine deriva da value_counts, cioè dalle frequenze del train
# — perciò la categoria più frequente riceve il codice più “piccolo” (poi si shiftano +1).
=======
        root[f"original_{tgt_col}"] = prev
        root.tab.target = tgt
        return {"root": root, "target_mapping": self.cat_types}
>>>>>>> upstream/main
