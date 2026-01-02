from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn import logger

from src.idspy.data.tab_accessor import PartitionName   #per definire tipi di dati categorici ordinati in pandas

from src.idspy.core.step import FitAwareStep, Step
from src.idspy.core.state import State
import logging



class LabelMap1(FitAwareStep):
    """Encode `target`: binary with `benign_tag`, else ordinal categories."""
    
    def __init__(
        self,
        benign_tag: Optional[str] = None,
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
        
        self.logger = logging.getLogger(__name__)

    @Step.requires(root=pd.DataFrame)
    def fit_impl(self, state: State, root: pd.DataFrame) -> None:
        #Learn ordered categories for the target col (if not binary).
        # Early exit for binary case
        if self.benign_tag is not None:
            self.cat_types = None
            return

        train_df = root.tab.train
        tgt_col = train_df.tab.schema.target

        vc = train_df[tgt_col].value_counts(dropna=False)
        self.cat_types = CategoricalDtype(categories=vc.index.tolist(), ordered=True)
       
    @Step.requires(root=pd.DataFrame)
    @Step.provides(root=pd.DataFrame, target_mapping=CategoricalDtype | None)
    def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:
        tgt_col = root.tab.schema.target

        prev = root[tgt_col].copy()

        if self.benign_tag is not None: 
            tgt = (prev == self.benign_tag).astype("int32")
            tgt = tgt.where(tgt == 0, 1)
            
        else:
            s = prev.astype(self.cat_types)
            codes = s.cat.codes
            tgt = pd.Series(            #codes + 1
                np.where(codes != -1, codes+1, 0).astype("int32"),
                index=s.index,
                name=tgt_col,
            )   
            # ⭐ LOG quante label sono unseen (ma NON dropparle)
            unseen_count = (codes == -1).sum()
            if unseen_count > 0:
                logger.warning(
                    f"⚠️ LabelMap: {unseen_count} label unseen, mappati a classe 0 (fallback)"
                )
            # # ⭐ AGGIUNGI QUESTO: 
            # unseen_count = (tgt == self.default).sum()
            # if unseen_count > 0:
            #     logger.warning(f"⚠️ LabelMap: {unseen_count} label unseen (default={self.default}), rimozione...")
            #     # Droppa righe con label unseen
            #     mask = tgt != self.default
            #     root = root[mask].copy()
            #     tgt = tgt[mask].copy()
        root[f"original_{tgt_col}"] = prev
        root.tab.target = tgt
        return {"root": root, "target_mapping": self.cat_types}



class FrequencyMapGlobal(FitAwareStep):
    """Mappa le colonne categoriali in base alla frequenza globale, senza richiedere split."""

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
        self.cat_types: Dict[str, CategoricalDtype] = {}
        super().__init__(
            name=name or "frequency_map_global",
            in_scope=in_scope,
            out_scope=out_scope,
        )
        self.logger = logging.getLogger(__name__)

    @Step.requires(root=pd.DataFrame)
    def fit_impl(self, state: State, root: pd.DataFrame) -> None:
        # USA TUTTO IL DATAFRAME RICEVUTO (root), non root.tab.train
        self.cat_types.clear()
        
        # Identifica colonne categoriali (usa l'accessore tab se disponibile, altrimenti select_dtypes)
        cat_cols = root.tab.categorical.columns if hasattr(root, 'tab') else root.select_dtypes(include=['object', 'category']).columns

        for col in cat_cols:
            vc = root[col].value_counts(dropna=False)
            if vc.empty:
                continue

            cats = vc.index.tolist() if self.max_levels is None else vc.head(self.max_levels).index.tolist()
            self.cat_types[col] = CategoricalDtype(categories=cats, ordered=True)
            self.logger.info(f"✅ FrequencyMapGlobal: fittata colonna '{col}' con {len(cats)} livelli.")

    @Step.requires(root=pd.DataFrame)
    @Step.provides(root=pd.DataFrame, cat_mapping=dict)
    def run(self, state: State, root: pd.DataFrame) -> Dict[str, Any]:
        if not self.cat_types:
            return {"root": root, "cat_mapping": {}}

        for col, dtype in self.cat_types.items():
            if col in root.columns:
                s = root[col].astype(dtype)
                codes = s.cat.codes
                # Mappa: unseen/NaN -> default, altri -> codes + 1
                root[col] = np.where(codes != -1, codes + 1, self.default).astype("int32")

        return {"root": root, "cat_mapping": self.cat_types}
    
    

class LabelMapGlobal(FitAwareStep):
    """Codifica il target in modo globale: binario o ordinale per frequenza."""
    
    def __init__(
        self,
        benign_tag: Optional[str] = None,
        default: int = 0,
        in_scope: str = "data",
        out_scope: str = "data",
        name: Optional[str] = None,
    ) -> None:
        self.benign_tag = benign_tag
        self.default = default
        self.cat_types: Optional[CategoricalDtype] = None
        super().__init__(
            name=name or "label_map_global",
            in_scope=in_scope,
            out_scope=out_scope,
        )
        self.logger = logging.getLogger(__name__)

    @Step.requires(root=pd.DataFrame)
    def fit_impl(self, state: State, root: pd.DataFrame) -> None:
        if self.benign_tag is not None:
            self.cat_types = None
            return

        # Recupera il nome della colonna target dallo schema
        tgt_col = root.tab.schema.target
        vc = root[tgt_col].value_counts(dropna=False)
        self.cat_types = CategoricalDtype(categories=vc.index.tolist(), ordered=True)
        self.logger.info(f"✅ LabelMapGlobal: fittata colonna target '{tgt_col}' con {len(vc)} classi.")
    
    @Step.requires(root=pd.DataFrame)
    @Step.provides(root=pd.DataFrame, target_mapping=Optional[CategoricalDtype])
    def run(self, state: State, root: pd.DataFrame) -> Dict[str, Any]:
        tgt_col = root.tab.schema.target
        prev = root[tgt_col].copy()

        if self.benign_tag is not None: 
            # 🎯 FORZATURA BINARIA: 
            # Se è uguale al tag benigno -> 0
            # Altrimenti (qualsiasi altra cosa) -> 1
            tgt = np.where(prev.astype(str) == str(self.benign_tag), 0, 1).astype("int32")
            self.logger.info(f"✅ LabelMapGlobal (Binary): {self.benign_tag} -> 0, Others -> 1")
        else:
            # Logica Multi-classe (ordinal)
            s = prev.astype(self.cat_types)
            codes = s.cat.codes
            tgt = np.where(codes != -1, codes + 1, self.default).astype("int32")
        ############
        self.logger.debug(f"Nome Colonna tgt_col: {tgt_col}")
        ###########
        root[f"original_{tgt_col}"] = prev
        # Molto importante: aggiorniamo la colonna target effettiva
        root[tgt_col] = tgt 
        root.tab.target = root[tgt_col] 
        
        return {"root": root, "target_mapping": self.cat_types}   
    """ @Step.requires(root=pd.DataFrame)
    @Step.provides(root=pd.DataFrame, target_mapping=Optional[CategoricalDtype])
    def run(self, state: State, root: pd.DataFrame) -> Dict[str, Any]:
        tgt_col = root.tab.schema.target
        prev = root[tgt_col].copy()

        if self.benign_tag is not None: 
            # Logica binaria: benigno vs tutto il resto
            # tgt = (prev == self.benign_tag).map({True: 0, False: 1}).astype("int32")
            tgt = np.where(prev == self.benign_tag, 0, 1).astype("int32")
        else:
            s = prev.astype(self.cat_types)
            codes = s.cat.codes
            tgt = np.where(codes != -1, codes + 1, self.default).astype("int32")
            
            unseen_count = (codes == -1).sum()
            if unseen_count > 0:
                self.logger.warning(f"⚠️ LabelMapGlobal: {unseen_count} label unseen mappati a {self.default}")

        root[f"original_{tgt_col}"] = prev
        root.tab.target = pd.Series(tgt, index=root.index, name=tgt_col)
        return {"root": root, "target_mapping": self.cat_types} """