from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ...core.step import Step    #classe base di tutti gli step di preprocessing; definisce interfaccia standard (run, requires, provides)
from ...core.state import State #contiene tutti i dati e lo stato della pipeline.
from ...data.tab_accessor import reattach_meta  #funzione per riattaccare informazioni aggiuntive al DataFrame dopo un filtro (utile in Filter).
#DropNulls normalizza e pulisce i dati in un unico step: rimuove tutti i valori problematici per evitare errori negli step successivi
class DropNulls(Step):  
    """Drop all rows that contain null values, including NaN and ±inf."""
    #dropnull ereditando da Step diventa uno step della pipeline che rimuove righe con valori mancanti (NaN) o infiniti (±inf)
    def __init__(
        self,
        in_scope: str = "data",
        out_scope: str = "data",
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            name=name or "drop_nulls",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    @Step.requires(root=pd.DataFrame)
    @Step.provides(root=pd.DataFrame)
    def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:
        root = root.replace([np.inf, -np.inf], np.nan).dropna()
        return {"root": root}

        dataframe = dataframe.replace([np.inf, -np.inf], np.nan).dropna()   #sostituisce infiniti con NaN e rimuove tutte le righe con NaN
        state[self.dataframe_out] = dataframe                       #salva il DataFrame pulito nello State usando la chiave specificata in dataframe_out

#Filter è utile per estrarre sottoinsiemi specifici di dati, senza perdere informazioni dello schema originale.
class Filter(Step):
    """Filter rows using a pandas query string."""

    def __init__(
        self,
        query: str,
        in_scope: str = "data",
        out_scope: str = "data",
        name: Optional[str] = None,
    ) -> None:
        self.query = query

        super().__init__(
            name=name or "filter",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    @Step.requires(root=pd.DataFrame)
    @Step.provides(root=pd.DataFrame)
    def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:
        filtered = root.query(self.query)
        return {"root": reattach_meta(root, filtered)}

        filtered = dataframe.query(self.query)          #applica il filtro  selezionando le righe che soddisfano la condizione
        state[self.dataframe_out] = reattach_meta(dataframe, filtered)  #salva il DataFrame filtrato nello State, riattaccando le informazioni di schema originale

#Log1p trasforma i dati numerici per ridurre l’influenza di outlier estremi e distribuire meglio i valori per i modelli ML.
class Log1p(Step):
    """Apply np.log1p to numerical columns."""

    def __init__(
        self,
        in_scope: str = "data",
        out_scope: str = "data",
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            name=name or "log1p",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    @Step.requires(root=pd.DataFrame)
    @Step.provides(root=pd.DataFrame)
    def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:

        root.tab.numerical = np.log1p(root.tab.numerical)
        return {"root": root}
