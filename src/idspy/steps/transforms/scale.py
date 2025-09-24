<<<<<<< HEAD
from typing import Optional     #indica che una variabile può essere di un certo tipo oppure None finchè non viene inizializzata
=======
from typing import Optional, Dict, Any
>>>>>>> upstream/main

import numpy as np
import pandas as pd

from ...core.step import FitAwareStep, Step
from ...core.state import State
<<<<<<< HEAD
from ...core.step import FitAwareStep   #importa la superclasse FitAwareStep che estende Step con funzionalità di fitting + run
from ...data.partition import PartitionName #enum con i nomi delle partizioni (TRAIN, VAL, TEST)
=======
>>>>>>> upstream/main

#tale classe standardizza le colonne numeriche del DataFrame usando media 0 e deviazione standard 1 calcolate sul training set evitando over/underflow
class StandardScale(FitAwareStep):  
    """Standardize numerical columns using mean/std with overflow-safe scaling."""

    def __init__(
        self,
        in_scope: str = "data",
        out_scope: str = "data",
        name: Optional[str] = None,
    ) -> None:
<<<<<<< HEAD
        self.dataframe_in = dataframe_in
        self.dataframe_out = dataframe_out or dataframe_in

        self._scale: Optional[pd.Series] = None     #serie pandas per memorizzare i fattori di scala calcolati durante il fitting
        self._means_s: Optional[pd.Series] = None   #serie pandas per memorizzare le medie calcolate durante il fitting
        self._stds_s: Optional[pd.Series] = None    #serie pandas per memorizzare le deviazioni standard calcolate durante il fitting
=======
        self._scale: Optional[pd.Series] = None
        self._means_s: Optional[pd.Series] = None
        self._stds_s: Optional[pd.Series] = None
>>>>>>> upstream/main

        super().__init__(
            name=name or "standard_scale",
            in_scope=in_scope,
            out_scope=out_scope,
        )

<<<<<<< HEAD
    def fit_impl(self, state: State) -> None:       #tale metodo calcola i parametri di scaling (media, deviazione standard) sul training set
=======
    @Step.requires(root=pd.DataFrame)
    def fit_impl(self, state: State, root: pd.DataFrame) -> None:
>>>>>>> upstream/main
        """Fit scaling stats on train split (overflow-safe)."""

<<<<<<< HEAD
        numerical_data = dataframe.tab.train.tab.numerical  #estrae le colonne numeriche del DataFrame della partizione di training
        if numerical_data.shape[1] == 0:                    #se non ci sono colonne numeriche, inizializza le serie vuote e ritorna
            self._scale = pd.Series(dtype="float64")        #serie vuota per i fattori di scala
            self._means_s = pd.Series(dtype="float64")      #serie vuota per le medie
            self._stds_s = pd.Series(dtype="float64")       #serie vuota per le deviazioni standard
=======
        numerical_data = root.tab.train.tab.numerical
        if numerical_data.shape[1] == 0:
            self._scale = pd.Series(dtype="float64")
            self._means_s = pd.Series(dtype="float64")
            self._stds_s = pd.Series(dtype="float64")
>>>>>>> upstream/main
            return

        numerical_data = numerical_data.astype(np.float64, copy=False).replace( #converte i dati in float64 e sostituisce infiniti con NaN
            [np.inf, -np.inf], np.nan                       
        )

        # Overflow-safe: compute scale and scaled values efficiently
        abs_max = numerical_data.abs().max(axis=0)  #calcola il valore assoluto massimo per ogni colonna numerica
        self._scale = (                             #calcola i fattori di scala per ogni colonna numerica
            abs_max.fillna(0.0).clip(lower=1e-10, upper=None).where(abs_max > 0.0, 1.0) #se una colonna è tutta nan la riempie con 0 
        )   #e se è minore di 1e-10, evita le scale troppo piccole, quindi evita di dividere per 0; se il max è 0 la sostituisce con 1.0                 

<<<<<<< HEAD
        num_scaled = numerical_data / self._scale   #scala i dati numerici dividendo per i fattori di scala calcolati
        self._means_s = num_scaled.mean()       #calcola le medie delle colonne numeriche scalate
        self._stds_s = num_scaled.std(ddof=0).clip(lower=1e-10, upper=None)  #calcola le dev standard delle colonne numeriche scalate, evitando valori troppo piccoli
    
    #prende tutte le colonne numeriche (non solo quelle train set) e applica le stats calcolate prima durante il fitting
    def run(self, state: State) -> None:    
=======
        num_scaled = numerical_data / self._scale
        self._means_s = num_scaled.mean()
        self._stds_s = num_scaled.std(ddof=0).clip(lower=1e-10, upper=None)

    @Step.requires(root=pd.DataFrame)
    @Step.provides(root=pd.DataFrame)
    def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:
>>>>>>> upstream/main
        """Apply standardization to numerical columns."""

<<<<<<< HEAD
        numerical_data = dataframe.tab.numerical    #estrae le colonne numeriche del DataFrame
        if numerical_data.shape[1] == 0:            #se non ci sono colonne numeriche, salva il DataFrame così com'è e ritorna
            state[self.dataframe_out] = dataframe
            return
=======
        numerical_data = root.tab.numerical
        if numerical_data.shape[1] == 0:
            return {"root": root}
>>>>>>> upstream/main

        numerical_data = numerical_data.astype(np.float64, copy=False).replace( #converte i dati in float64 e sostituisce infiniti con NaN
            [np.inf, -np.inf], np.nan
        )

<<<<<<< HEAD
        # Reindex scaling parameters efficiently
        # reindicizza i parametri di scaling per allinearli alle colonne numeriche attuali, riempiendo con valori di default se necessario
        cols = numerical_data.columns   #prende i nomi delle colonne numeriche attuali
        scale = self._scale.reindex(cols, fill_value=1.0)   #reindicizza i fattori di scala, riempiendo con 1.0 se manca qualche colonna
        means_s = self._means_s.reindex(cols, fill_value=0.0)   #reindicizza le medie, riempiendo con 0.0 se manca qualche colonna
        stds_s = self._stds_s.reindex(cols, fill_value=1.0)  #reindicizza le dev standard, riempiendo con 1.0 se manca qualche colonna

        dataframe.tab.numerical = (numerical_data / scale - means_s) / stds_s   #applica la standardizzazione alle colonne numeriche, cioè scala, sottrae la media e divide per la dev standard
        state[self.dataframe_out] = dataframe   #salva il DataFrame standardizzato nello State usando la chiave specificata in dataframe_out
=======
        cols = numerical_data.columns
        scale = self._scale.reindex(cols, fill_value=1.0)
        means_s = self._means_s.reindex(cols, fill_value=0.0)
        stds_s = self._stds_s.reindex(cols, fill_value=1.0)

        root.tab.numerical = (numerical_data / scale - means_s) / stds_s
        return {"root": root}
>>>>>>> upstream/main


class MinMaxScale(FitAwareStep):
    """Scale numerical columns to [0, 1] via min/max."""

    def __init__(
        self,
        in_scope: str = "data",
        out_scope: str = "data",
        name: Optional[str] = None,
    ) -> None:
<<<<<<< HEAD
        self.dataframe_in = dataframe_in
        self.dataframe_out = dataframe_out or dataframe_in

        self._min: Optional[pd.Series] = None   #salva i valori minimi calcolati durante il fitting del training set
        self._max: Optional[pd.Series] = None   #salva i valori massimi calcolati durante il fitting del training set
=======
        self._min: Optional[pd.Series] = None
        self._max: Optional[pd.Series] = None
>>>>>>> upstream/main

        super().__init__(
            name=name or "min_max_scale",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    @Step.requires(root=pd.DataFrame)
    def fit_impl(self, state: State, root: pd.DataFrame) -> None:
        """Fit min/max on train split."""
        numerical_data = root.tab.train.tab.numerical
        if numerical_data.shape[1] == 0:
            self._min = pd.Series(dtype="float64")
            self._max = pd.Series(dtype="float64")
            return

        numerical_data = numerical_data.astype(np.float64, copy=False).replace(
            [np.inf, -np.inf], np.nan
        )
        self._min = numerical_data.min()
        self._max = numerical_data.max()

    @Step.requires(root=pd.DataFrame)
    @Step.provides(root=pd.DataFrame)
    def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Apply min-max scaling to numerical columns."""

        numerical_data = root.tab.numerical
        if numerical_data.shape[1] == 0:
            return {"root": root}

        numerical_data = numerical_data.astype(np.float64, copy=False).replace(
            [np.inf, -np.inf], np.nan
        )

<<<<<<< HEAD
        # Reindex scaling parameters and apply min-max scaling efficiently
        # reindicizza i parametri di scaling per allinearli alle colonne numeriche attuali, riempiendo con valori di default se necessario
        cols = numerical_data.columns   #prende i nomi delle colonne numeriche attuali
        col_min = self._min.reindex(cols, fill_value=0.0)   #reindicizza i valori minimi, riempiendo con 0.0 se manca qualche colonna
        col_max = self._max.reindex(cols, fill_value=1.0)   #reindicizza i valori massimi, riempiendo con 1.0 se manca qualche colonna
        den = (col_max - col_min).clip(lower=1e-10, upper=None) #calcola il denominatore (max - min) e lo clippa per evitare valori troppo piccoli

        dataframe.tab.numerical = (numerical_data - col_min) / den  #applica la normalizzazione min-max alle colonne numeriche, cioè sottrae il minimo e divide per (max - min)
        state[self.dataframe_out] = dataframe   #salva il DataFrame normalizzato nello State usando la chiave specificata in dataframe_out
        
        
# Quando usare l’uno o l’altro?

# StandardScale: per modelli che assumono dati centrati/normalizzati (es. modelli lineari, PCA, reti neurali). 
# Robusto agli outlier grazie alla pre-scala.

# MinMaxScale: utile quando vuoi range fisso [0,1] (es. reti con attivazioni sensibili alla scala, 
# o quando certe feature devono essere tutte comparabili in [0,1]). Più sensibile agli outlier.
=======
        cols = numerical_data.columns
        col_min = self._min.reindex(cols, fill_value=0.0)
        col_max = self._max.reindex(cols, fill_value=1.0)
        den = (col_max - col_min).clip(lower=1e-10, upper=None)

        root.tab.numerical = (numerical_data - col_min) / den
        return {"root": root}
>>>>>>> upstream/main
