import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from src.idspy.core.pipeline import Step, State
import logging

"""
Global Normalization per Federated Learning.
Calcola statistiche su TUTTO il dataset, poi applica uniformemente.
"""

class ComputeGlobalNormalizationStats(Step):
    """
    Calcola mean/std su TUTTO il dataset PRIMA dello split federato.
    Salva le statistiche nello State per uso successivo.
    """
    def __init__(
        self,
        in_scope: str = "data",
        out_scope: str = "data",
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            name=name or "compute_global_norm_stats",
            in_scope=in_scope,
            out_scope=out_scope,
        )
        self.logger = logging.getLogger(__name__)

    @Step.requires(root=pd.DataFrame)
    def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Calcola statistiche globali su colonne numeriche."""
        
        # 1. Identifica colonne numeriche
        numerical_cols = root.select_dtypes(include=[np.number]).columns.tolist()
        
        # Escludi colonne che non vanno normalizzate (es. Attack, port numbers)
        # lista personalizzabile di colonne da escludere
        exclude_cols = ['Attack', 'Label', 'L4_SRC_PORT', 'L4_DST_PORT'] ##########AGGIUNTO LABEL
        numerical_cols = [c for c in numerical_cols if c not in exclude_cols]
        
        if len(numerical_cols) == 0:
            self.logger.warning("No numerical columns to normalize!")
            state.set("global_norm_stats", None)
            return {"root": root}
        
        # 2. Estrai dati numerici
        numerical_data = root[numerical_cols].astype(np.float64, copy=False)
        numerical_data = numerical_data.replace([np.inf, -np.inf], np.nan)
        
        #Aggiunto
        numerical_data = numerical_data.clip(lower=-1e10, upper=1e10)
        
        # 3. Calcola statistiche (overflow-safe come StandardScale)
        abs_max = numerical_data.abs().max(axis=0)
        # scale = abs_max.fillna(0.0).clip(lower=1e-10, upper=None).where(abs_max > 0.0, 1.0)
        scale = abs_max.fillna(0.0).clip(lower=1e-10, upper=1e10).where(abs_max > 0.0, 1.0)
        
        num_scaled = numerical_data / scale
        means_s = num_scaled.mean()
        stds_s = num_scaled.std(ddof=0).clip(lower=1e-10, upper=None)
        
        # 4. Salva nello State (accessibile da tutti gli step successivi)
        stats = {
            'scale': scale,
            'means_s': means_s,
            'stds_s': stds_s,
            'numerical_cols': numerical_cols,
        }
        
        state.set("global_norm_stats", stats, dict)
        
        self.logger.info(f"✅ Global normalization stats computed for {len(numerical_cols)} columns")
        # self.logger.debug(f"Columns: {numerical_cols}")
        
        # ⭐ AGGIUNGI:
        self.logger.debug(f"📊 Stats preview:")
        self.logger.debug(f"   scale range: [{scale.min():.2f}, {scale.max():.2f}]")
        self.logger.debug(f"   means_s range: [{means_s.min():.2f}, {means_s.max():.2f}]")
        self.logger.debug(f"   stds_s range: [{stds_s.min():.2f}, {stds_s.max():.2f}]")

        return {"root": root}


class ApplyGlobalNormalization(Step):
    """
    Applica le statistiche globali calcolate da ComputeGlobalNormalizationStats.
    """
    def __init__(
        self,
        in_scope: str = "data",
        out_scope: str = "data",
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            name=name or "apply_global_normalization",
            in_scope=in_scope,
            out_scope=out_scope,
        )
        self.logger = logging.getLogger(__name__)
    
    @Step.requires(root=pd.DataFrame)
    @Step.provides(root=pd.DataFrame)
    def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Applica normalizzazione usando statistiche globali."""
        
        # 1. Recupera statistiche dallo State
        stats = state.get("global_norm_stats", dict)
        
        if stats is None:
            self.logger.warning("No global normalization stats found! Skipping...")
            return {"root": root}
        
        scale = stats['scale']
        means_s = stats['means_s']
        stds_s = stats['stds_s']
        numerical_cols = stats['numerical_cols']
        
        # 2. Applica normalizzazione
        if len(numerical_cols) == 0:
            return {"root": root}
        
        numerical_data = root[numerical_cols].astype(np.float64, copy=False)
        numerical_data = numerical_data.replace([np.inf, -np.inf], np.nan)
        
        # Usa le statistiche GLOBALI (non locali!)
        # root[numerical_cols] = (numerical_data / scale - means_s) / stds_s ---> sostituito con:
        
        # Usa le statistiche GLOBALI (non locali!)
        normalized = (numerical_data / scale - means_s) / stds_s

        # 🔧 FIX: Clipping per proteggere da outlier estremi
        # ±10 sigma = 99.9999% dei dati per una gaussiana
        normalized = normalized.clip(-10, 10)

        root[numerical_cols] = normalized
        
        self.logger.info(f"✅ Applied global normalization to {len(numerical_cols)} columns")
        
        return {"root": root}