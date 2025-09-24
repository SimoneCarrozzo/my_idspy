import numpy as np
import pandas as pd

from ..helpers import validate_instance #funzione helper per controllare che l’oggetto sia un DataFrame
from ...core.state import State         #contiene tutti i dati e lo stato della pipeline.
from ...core.step import Step           #classe base di tutti gli step di preprocessing; definisce interfaccia standard (run, requires, provides)
from ...data.tab_accessor import reattach_meta  #funzione per riattaccare informazioni aggiuntive al DataFrame dopo un filtro (utile in Filter).

#DropNulls normalizza e pulisce i dati in un unico step: rimuove tutti i valori problematici per evitare errori negli step successivi
class DropNulls(Step):  
    """Drop all rows that contain null values, including NaN and ±inf."""
    #dropnull ereditando da Step diventa uno step della pipeline che rimuove righe con valori mancanti (NaN) o infiniti (±inf)
    def __init__(
        self,
        dataframe_in: str = "data.root",        #nome della chiave nello State dove si trova il DataFrame di input
        dataframe_out: str | None = None,       #nome della chiave nello State dove salvare il DataFrame di output (se None, sovrascrive l'input)
        name: str | None = None,                #nome opzionale dello step (se None, usa il nome della classe)
    ) -> None:
        self.dataframe_in = dataframe_in
        self.dataframe_out = dataframe_out or dataframe_in

        super().__init__(                       #chiamata al costruttore della superclasse Step dichiarando:
            name=name or "drop_nulls",          #nome dello step (default "drop_nulls")
            requires=[self.dataframe_in],       #richiede il DataFrame di input / dati di input richiesti dallo step
            provides=[self.dataframe_out],      #fornisce il DataFrame di output / dati di output prodotti dallo step
        )

    def run(self, state: State) -> None:        #implementazione concreta del metodo astratto run di Step
        dataframe = state[self.dataframe_in]    #prende il DataFrame dallo State usando la chiave specificata in dataframe_in
        validate_instance(dataframe, pd.DataFrame, self.name)       #verifica che l'oggetto sia effettivamente un DataFrame, altrimenti solleva un errore

        dataframe = dataframe.replace([np.inf, -np.inf], np.nan).dropna()   #sostituisce infiniti con NaN e rimuove tutte le righe con NaN
        state[self.dataframe_out] = dataframe                       #salva il DataFrame pulito nello State usando la chiave specificata in dataframe_out

#Filter è utile per estrarre sottoinsiemi specifici di dati, senza perdere informazioni dello schema originale.
class Filter(Step):
    """Filter rows using a pandas query string."""

    def __init__(
        self,
        query: str,     #stringa pandas query per filtrare il DataFrame (es. "age > 30 and income < 50000")
        dataframe_in: str = "data.root",
        dataframe_out: str | None = None,
        name: str | None = None,
    ) -> None:
        self.query = query
        self.dataframe_in = dataframe_in
        self.dataframe_out = dataframe_out or dataframe_in

        super().__init__(                   #chiamata al costruttore della superclasse Step dichiarando:
            name=name or "filter",          #nome dello step (default "filter")
            requires=[self.dataframe_in],   #richiede il DataFrame di input / dati di input richiesti dallo step
            provides=[self.dataframe_out],  #fornisce il DataFrame di output / dati di output prodotti dallo step
        )

    def run(self, state: State) -> None:    #implementazione concreta del metodo astratto run di Step
        dataframe = state[self.dataframe_in]    #prende il DataFrame dallo State usando la chiave specificata in dataframe_in
        validate_instance(dataframe, pd.DataFrame, self.name)   #verifica che l'oggetto sia effettivamente un DataFrame, altrimenti solleva un errore

        filtered = dataframe.query(self.query)          #applica il filtro  selezionando le righe che soddisfano la condizione
        state[self.dataframe_out] = reattach_meta(dataframe, filtered)  #salva il DataFrame filtrato nello State, riattaccando le informazioni di schema originale

#Log1p trasforma i dati numerici per ridurre l’influenza di outlier estremi e distribuire meglio i valori per i modelli ML.
class Log1p(Step):
    """Apply np.log1p to numerical columns."""

    def __init__(
        self,
        dataframe_in: str = "data.root",    #nome della chiave nello State dove si trova il DataFrame di input
        dataframe_out: str | None = None,
        name: str | None = None,
    ) -> None:
        self.dataframe_in = dataframe_in
        self.dataframe_out = dataframe_out or dataframe_in

        super().__init__(
            name=name or "log1p",        #nome dello step (default "log1p")
            requires=[self.dataframe_in],
            provides=[self.dataframe_out],
        )

    def run(self, state: State) -> None:
        dataframe = state[self.dataframe_in]
        validate_instance(dataframe, pd.DataFrame, self.name)

        dataframe.tab.numerical = np.log1p(dataframe.tab.numerical)
        state[self.dataframe_out] = dataframe

    #cosa fa il run di questa classe:
    # Prendi il DataFrame e verifica tipo.
    # dataframe.tab.numerical → seleziona tutte le colonne numeriche.
    # np.log1p(...) → calcola log(1+x) su tutti i valori numerici.
    # Salva risultato nello State.