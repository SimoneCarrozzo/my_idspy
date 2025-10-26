from typing import Any, Dict, Optional, Tuple
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



class FilterZeroLabel(Step):
    
    def __init__(
        self,
        target_column: str,
        in_scope: str = "data",
        out_scope: str = "data", 
        name: Optional[str] = None,
       ) -> None:
        self.target_column = target_column
        super().__init__(
            name=name or "filter_zero_labels",
            in_scope=in_scope,
            out_scope=out_scope,
        )
    @Step.requires(root=pd.DataFrame)
    @Step.provides(root=pd.DataFrame)
    def run(self, state:State, root: pd.DataFrame) -> Optional[Dict[str,Any]]:
        col_type = root[self.target_column].dtype

        if pd.api.types.is_numeric_dtype(col_type):
            # filtro numerico: elimino tutte le righe con valore 0, parte inizialmente non richiesta come ex iniziale
            filtered = root[root[self.target_column] != 0]
            dropped_lb = root[self.target_column][root[self.target_column] == 0].value_counts()
            kept_lb = root[self.target_column][root[self.target_column] != 0].value_counts().index
        else: #conto numero di righe prima applicazione filtro
            number = root[root.tab.schema.target].value_counts()
            kept_lb = number[number != 10].index
            dropped_lb = number[number == 0]
            filtered = root[root[self.target_column].isin(kept_lb)]
        
        # logging interno
        print(f"[FilterZeroLabels] Kept labels: {list(kept_lb)}")
        print(f"[FilterZeroLabels] Dropped labels: {dropped_lb.to_dict()}")
        print(f"[FilterZeroLabels] Righe eliminate: {len(root) - len(filtered)}")

        # salvo nel state info utili per debug/analisi
        state.update("kept_classes", list(kept_lb), list)
        state.update("dropped_label_counts", dropped_lb.to_dict(), dict)
        state.update(f"{self.name}.dropped_rows", len(root) - len(filtered), int)

        # Riattacca eventuali metadati e salva nello state
        filtered_root = reattach_meta(root, filtered)
        state.update("data", filtered_root, pd.DataFrame)

        # Restituisce il dict richiesto dalla pipeline
        return {"root": filtered_root}




#aggiunto come compito
class FilterRareLabels(Step):
    
    def __init__(
        self,
        target_column: str,
        min_count: int=3000,
        in_scope: str = "data",
        out_scope: str = "data", 
        name: Optional[str] = None,
       ) -> None:
        self.target_column = target_column
        self.min_count = min_count
        super().__init__(
            name=name or "filter_rare_labels",
            in_scope=in_scope,
            out_scope=out_scope,
        )
    @Step.requires(root=pd.DataFrame)
    @Step.provides(root=pd.DataFrame)
    def run(self, state:State, root: pd.DataFrame) -> Optional[Dict[str,Any]]:
        #conta le occorrenze delle etichette
        counts = root[root.tab.schema.target].value_counts()
        #mantengo solo quelle con almeno min_count esempi
        keep_labels = counts[counts >= self.min_count].index
        
        dropped_labels = counts[counts < self.min_count]
        
        filtered = root[root[self.target_column].isin(keep_labels)]
        
        # logging interno
        print(f"[FilterRareLabels] Kept labels: {list(keep_labels)}")
        print(f"[FilterRareLabels] Dropped labels: {dropped_labels.to_dict()}")
        print(f"[FilterRareLabels] Righe eliminate: {len(root) - len(filtered)}")

        # salvo nel state info utili per debug/analisi
        state.update("kept_classes", list(keep_labels), list)
        state.update("dropped_label_counts", dropped_labels.to_dict(), dict)
        state.update(f"{self.name}.dropped_rows", len(root) - len(filtered), int)

        #state.update("data", reattach_meta(root, filtered), pd.DataFrame)
        #return {"root": reattach_meta(root, filtered)}
        # Riattacca eventuali metadati e salva nello state
        filtered_root = reattach_meta(root, filtered)
        state.update("data", filtered_root, pd.DataFrame)

        # Restituisce il dict richiesto dalla pipeline
        return {"root": filtered_root}


class FeatureGenerator(Step):
    
    def __init__(
        self,
        name: Optional[str] = None,
        in_scope: str = "data",
        out_scope: str = "data",
      ) -> None:
        super().__init__(
            name = name or "feature_generator",
            in_scope = in_scope,
            out_scope = out_scope,
        )
    @Step.requires(root = pd.DataFrame)
    @Step.provides(root = pd.DataFrame)
    def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:
        num_data = root.tab.numerical.copy()
        
        new_features = {}
        
        for col in num_data.columns:
            new_features[col + "_squared"] = num_data[col]**2
            if(num_data[col] >= 0).all():
                new_features[col + "_sqrt"] = np.sqrt(num_data[col])
                new_features[col + "_log1p"] = np.log1p(num_data[col])
        
        new_df = pd.DataFrame(new_features, index=num_data.index)
        num_data = pd.concat([num_data, new_df], axis=1)
        print("[FeatureGenerator][RUN] Create nuove colonne:", list(new_features.keys()))
        print("[FeatureGenerator][RUN] Statistiche descrittive delle nuove colonne:\n",
              new_df.describe().T[["mean", "std", "min", "max"]])

        root.tab.numerical = num_data
        return {"root": root}
    
class CategoricalEncoder(Step):
    def __init__(
        self,
        name: Optional[str] = None,
        in_scope: str = "data",
        out_scope: str = "data"
      ) -> None:
        super().__init__(
            name=name or "categorical_encoder",
            in_scope=in_scope,
            out_scope=out_scope
        )
    @Step.requires(root=pd.DataFrame)
    @Step.provides(root=pd.DataFrame)
    def run(self, state:State, root=pd.DataFrame) -> Optional[Dict[str, Any]]:
        cat = root.tab.categorical
        
        if cat.empty:
            print("[CategoricalEncoder][RUN] Nessuna colonna categorica trovata.")
            return {"root":root}
        print("[CategoricalEncoder][RUN] Colonne categoriche trovate.", list(cat.columns))
        
        encoded = pd.get_dummies(cat, dummy_na=True, prefix_sep="__")
        print(f"[CategoricalEncoder][RUN] Create {encoded.shape[1]} colonne da {cat.shape[1]} originali.")

        print("[CategoricalEncoder][RUN] Esempio delle prime 5 righe dopo encoding:")
        print(encoded.head())
        
        root.tab.categorical = encoded
        return {"root":root}
    
    

class FeatureGenV2(Step):
    def __init__(
        self,
        name: Optional[str] = None,
        in_scope: str = "data",
        out_scope: str = "data",
        ratio_cols: Optional[Tuple[str,str]] = None,
        diff_cols: Optional[Tuple[str,str]] = None,
        soglia_cols: Optional[str] = None,
        soglia: float=0.0,
      ) -> None:
        self.ratio_cols = ratio_cols
        self.diff_cols = diff_cols
        self.soglia_cols = soglia_cols
        self.soglia = soglia
        super().__init__(
            name = name or "feature_gen_v2",
            in_scope = in_scope,
            out_scope = out_scope,
        )
    @Step.requires(root=pd.DataFrame)
    @Step.provides(root=pd.DataFrame)
    def run(self, state:State, root=pd.DataFrame) -> Optional[Dict[str,Any]]:
        
        df = root.copy()
        if self.ratio_cols is not None:
            col1, col2 = self.ratio_cols
            df[f"{col1}_div_{col2}"] = df[col1] / (df[col2] + 1e-6)
            print("[FeatureGenV2][RUN] Creata colonna: {col1}_div_{col2} .\n")
            print(df[f"{col1}_div_{col2}"])

        if self.diff_cols is not None:
            col1, col2 = self.diff_cols
            df[f"{col1}_absdiff_{col2}"] = (df[col1] - df[col2]).abs()
            print("[FeatureGenV2][RUN] Creata colonna: {col1}_absdiff_{col2} .\n")
            print(df[f"{col1}_absdiff_{col2}"])
        
        if self.soglia_cols is not None:
            df[f"{self.soglia_cols}_soglia_{self.soglia}"] = (df[self.soglia_cols] > self.soglia).astype(int)
            print("[FeatureGenV2][RUN] Creata colonna: {self.soglia_cols}_soglia_{self.soglia}.\n")
            print(df[f"{self.soglia_cols}_soglia_{self.soglia}"])

        return {"root": root}