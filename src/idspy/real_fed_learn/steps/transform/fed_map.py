from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn import logger

from src.idspy.data.tab_accessor import PartitionName   #per definire tipi di dati categorici ordinati in pandas

from src.idspy.core.step import FitAwareStep, Step
from src.idspy.core.state import State
import logging

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
        # mapping_output_path: Optional[str] = None  # 🆕
    ) -> None:
        self.benign_tag = benign_tag
        self.default = default
        self.cat_types: Optional[CategoricalDtype] = None

        # self.mapping_output_path = mapping_output_path #new

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
        # Creiamo il tipo categorico ordinato per frequenza
        self.cat_types = CategoricalDtype(categories=vc.index.tolist(), ordered=True)
        self.logger.info(f"✅ LabelMapGlobal: fittata colonna target '{tgt_col}' con {len(vc)} classi.")
        
        
    
    @Step.requires(root=pd.DataFrame)
    # @Step.provides(root=pd.DataFrame, target_mapping=Optional[CategoricalDtype])
    @Step.provides(root=pd.DataFrame, 
                   target_mapping=Optional[CategoricalDtype],
                   training_label_map=dict, #dizionario per training binario federato
                   original_label_map=dict  #dizionario per visualizzazione mapping originale
                   )
    def run(self, state: State, root: pd.DataFrame) -> Dict[str, Any]:
        tgt_col = root.tab.schema.target
        prev = root[tgt_col].copy()

        # 🆕 Dizionario per training (binario o multi-classe)
        training_label_map_dict = {}
        
        # 🆕 Dizionario ORIGINALE (tutti gli attacchi)
        original_label_map_dict = {}

        if self.benign_tag is not None: 
            
            # 1️⃣ Crea il mapping ORIGINALE (tutti gli attacchi)
            unique_labels = prev.astype(str).unique()
            for idx, label in enumerate(sorted(unique_labels)):
                original_label_map_dict[label] = idx
            
            # 2️⃣ Crea il mapping per TRAINING (binario)
            training_label_map_dict = {
                str(self.benign_tag): 0,
                "Attacks": 1
            }
            
            # 3️⃣ Trasforma i target per training (binario) # Se è uguale al tag benigno -> 0, Altrimenti (qualsiasi altra cosa) -> 1
            tgt = np.where(prev.astype(str) == str(self.benign_tag), 0, 1).astype("int32")
            
            self.logger.info(f"✅ LabelMapGlobal (Binary): {self.benign_tag} -> 0, Others -> 1")
            self.logger.info(f"   • Original labels preserved: {len(original_label_map_dict)} classi")

        else:
            # Logica Multi-classe (ordinal)
            s = prev.astype(self.cat_types)
            codes = s.cat.codes
            # Applichiamo lo shift +1 (0 è riservato al default/unknown)
            tgt = np.where(codes != -1, codes + 1, self.default).astype("int32")
            
            # Creiamo il dizionario esplicito che riflette ESATTAMENTE questo shift
            # In questo caso, training = original
            training_label_map_dict["Unknown/Default"] = self.default
            for idx, name in enumerate(self.cat_types.categories):
                training_label_map_dict[str(name)] = int(idx + 1)
            
            original_label_map_dict = training_label_map_dict.copy()
        
        ############
        self.logger.debug(f"Nome Colonna tgt_col: {tgt_col}")
        
        ###########
        root[f"original_{tgt_col}"] = prev
        # Molto importante: aggiorniamo la colonna target effettiva
        root[tgt_col] = tgt 
        root.tab.target = root[tgt_col] 
        
        return {"root": root, 
                "target_mapping": self.cat_types,
                "training_label_map": training_label_map_dict, # NUOVO: passiamo i dizionari training e original
                "original_label_map": original_label_map_dict}  
#======================================
#======================================
#======================================
class CreateOneVsRestLabels(Step):
    """
    Crea colonne binarie One-vs-Rest per ogni tipo di attacco.
    Esempio: is_ddos=1 se attacco è DDoS, altrimenti 0
    """
    
    def __init__(
        self,
        attack_types: List[str] = None,  # ["DDoS", "DoS", "Bot", "Infiltration"]
        original_col: str = "original_Attack",
        in_scope: str = "data",
        out_scope: str = "data",
        name: Optional[str] = None,
    ):
        self.attack_types = attack_types or [
            "DDOS attack-HOIC",  # Nome ESATTO dal tuo dataset
            "DoS attacks-Hulk",
            "Bot", 
            "Infilteration",  # Nota: c'è un typo nel dataset
            "DDOS attacks-LOIC-HTTP",
            "DDOS attack-LOIC-UDP",
            "DoS attacks-GoldenEye"
        ]
        self.original_col = original_col
        
        super().__init__(
            name=name or "create_ovr_labels",
            in_scope=in_scope,
            out_scope=out_scope,
        )
        self.logger = logging.getLogger(__name__)
    
    @Step.requires(root=pd.DataFrame)
    @Step.provides(root=pd.DataFrame, ovr_columns=list)
    def run(self, state: State, root: pd.DataFrame) -> Dict[str, Any]:
        
        if self.original_col not in root.columns:
            raise ValueError(f"Colonna {self.original_col} non trovata!")
        
        ovr_cols = []
        
        for attack in self.attack_types:
            # Crea nome colonna pulito (es: "DDOS attack-HOIC" → "is_ddos")
            col_name = f"is_{attack.lower().replace(' ', '_').replace('-', '_')}"
            
            # Crea colonna binaria
            root[col_name] = (root[self.original_col].astype(str) == attack).astype(np.int32)
            
            count = root[col_name].sum()
            ovr_cols.append(col_name)
            
            self.logger.info(f"✅ Creata colonna '{col_name}': {count} samples positivi")
        
        self.logger.info(f"✅ Totale colonne One-vs-Rest create: {len(ovr_cols)}")
        
        return {"root": root, "ovr_columns": ovr_cols}