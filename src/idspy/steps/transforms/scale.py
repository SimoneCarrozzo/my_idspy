from typing import Optional, Dict, Any
import warnings #Optional indica che una variabile può essere di un certo tipo oppure None finchè non viene inizializzata

import numpy as np
import pandas as pd

from ...data.tab_accessor import PartitionName

from ...core.step import FitAwareStep, Step
from ...core.state import State


#tale classe standardizza le colonne numeriche del DataFrame usando media 0 e deviazione standard 1 calcolate sul training set evitando over/underflow
class StandardScale(FitAwareStep):  
    """Standardize numerical columns using mean/std with overflow-safe scaling."""

    def __init__(
        self,
        in_scope: str = "data",
        out_scope: str = "data",
        name: Optional[str] = None,
    ) -> None:
        self._scale: Optional[pd.Series] = None     #serie pandas per memorizzare i fattori di scala calcolati durante il fitting
        self._means_s: Optional[pd.Series] = None       #serie pandas per memorizzare le medie calcolate durante il fitting
        self._stds_s: Optional[pd.Series] = None    #serie pandas per memorizzare le deviazioni standard calcolate durante il fitting
                       
        super().__init__(
            name=name or "standard_scale",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    @Step.requires(root=pd.DataFrame)
    def fit_impl(self, state: State, root: pd.DataFrame) -> None:
        #Fit scaling stats on train split (overflow-safe).
        numerical_data = root.tab.train.tab.numerical   #estrae le colonne numeriche del DataFrame della partizione di training
        if numerical_data.shape[1] == 0:                #se non ci sono colonne numeriche, inizializza le serie vuote e ritorna
            self._scale = pd.Series(dtype="float64")        #serie vuota per i fattori di scala
            self._means_s = pd.Series(dtype="float64")      #serie vuota per le medie
            self._stds_s = pd.Series(dtype="float64")   #serie vuota per le deviazioni standard
            return

        numerical_data = numerical_data.astype(np.float64, copy=False).replace( #converte i dati in float64 e sostituisce infiniti con NaN
            [np.inf, -np.inf], np.nan                       
        )

        # Overflow-safe: compute scale and scaled values efficiently
        abs_max = numerical_data.abs().max(axis=0)  #calcola il valore assoluto massimo per ogni colonna numerica
        self._scale = (                             #calcola i fattori di scala per ogni colonna numerica
            abs_max.fillna(0.0).clip(lower=1e-10, upper=None).where(abs_max > 0.0, 1.0) #se una colonna è tutta nan la riempie con 0 
        )   #e se è minore di 1e-10, evita le scale troppo piccole, quindi evita di dividere per 0; se il max è 0 la sostituisce con 1.0                 

        num_scaled = numerical_data / self._scale
        self._means_s = num_scaled.mean()
        self._stds_s = num_scaled.std(ddof=0).clip(lower=1e-10, upper=None)
    """ @Step.requires(root=pd.DataFrame)
    def fit_impl(self, state: State, root: pd.DataFrame) -> None:
        #LEGGERA MODIFICA:
        if root.tab.has_partitions and root.tab.has_partition(PartitionName.TRAIN.value):
            numerical_data = root.tab.train.tab.numerical
        else: # Nessuna partizione → usa direttamente il DataFrame
            numerical_data = root.tab.numerical
            #numerical_data = root.tab.train.tab.numerical   
        
        if numerical_data.shape[1] == 0:                
            self._scale = pd.Series(dtype="float64")    
            self._means_s = pd.Series(dtype="float64")  
            self._stds_s = pd.Series(dtype="float64")   
            return

        numerical_data = numerical_data.astype(np.float64, copy=False).replace(
            [np.inf, -np.inf], np.nan                       
        )

        # Overflow-safe: compute scale and scaled values efficiently
        abs_max = numerical_data.abs().max(axis=0)  
        self._scale = (                             
            abs_max.fillna(0.0).clip(lower=1e-10, upper=None).where(abs_max > 0.0, 1.0)  
        )                    

        num_scaled = numerical_data / self._scale
        self._means_s = num_scaled.mean()
        self._stds_s = num_scaled.std(ddof=0).clip(lower=1e-10, upper=None)"""
 
    @Step.requires(root=pd.DataFrame)
    @Step.provides(root=pd.DataFrame)
    def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Apply standardization to numerical columns."""

        numerical_data = root.tab.numerical
        if numerical_data.shape[1] == 0:
            return {"root": root}

        numerical_data = numerical_data.astype(np.float64, copy=False).replace( #converte i dati in float64 e sostituisce infiniti con NaN
            [np.inf, -np.inf], np.nan
        )

        cols = numerical_data.columns
        scale = self._scale.reindex(cols, fill_value=1.0)
        means_s = self._means_s.reindex(cols, fill_value=0.0)
        stds_s = self._stds_s.reindex(cols, fill_value=1.0)

        root.tab.numerical = (numerical_data / scale - means_s) / stds_s
        
        return {"root": root}
   

class MinMaxScale(FitAwareStep):
    """Scale numerical columns to [0, 1] via min/max."""

    def __init__(
        self,
        in_scope: str = "data",
        out_scope: str = "data",
        name: Optional[str] = None,
    ) -> None:
        self._min: Optional[pd.Series] = None
        self._max: Optional[pd.Series] = None

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
            self._min = pd.Series(dtype="float32")
            self._max = pd.Series(dtype="float32")
            return

        numerical_data = numerical_data.astype(np.float32, copy=False).replace(
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

        numerical_data = numerical_data.astype(np.float32, copy=False).replace(
            [np.inf, -np.inf], np.nan
        )

        cols = numerical_data.columns
        col_min = self._min.reindex(cols, fill_value=0.0)
        col_max = self._max.reindex(cols, fill_value=1.0)
        den = (col_max - col_min).clip(lower=1e-10, upper=None)

        root.tab.numerical = (numerical_data - col_min) / den
        return {"root": root}



#==================================================================================================
warnings.filterwarnings("ignore", category=FutureWarning)

class StandardScaleMemoryEfficient(FitAwareStep):
    """
    Standardize numerical columns usando mean/std con overflow-safe scaling.
    Versione MEMORY-EFFICIENT: processa colonna-per-colonna per gestire 100M+ righe.
    
    🔧 IDENTICO alla StandardScale originale, ma ottimizzato per grandi dataset.
    """
    def __init__(
        self,
        in_scope: str = "data",
        out_scope: str = "data",
        chunk_size: int = 1_000_000,  # Processa 1M righe alla volta
        name: Optional[str] = None,
    ) -> None:
        self._scale: Optional[pd.Series] = None
        self._means_s: Optional[pd.Series] = None
        self._stds_s: Optional[pd.Series] = None
        self._chunk_size = chunk_size
        
        super().__init__(
            name=name or "standard_scale_memory_efficient",
            in_scope=in_scope,
            out_scope=out_scope,
        )
    
    @Step.requires(root=pd.DataFrame)
    def fit_impl(self, state: State, root: pd.DataFrame) -> None:
        """Fit scaling stats on train split (overflow-safe + memory-efficient)."""
        numerical_data = root.tab.train.tab.numerical
        
        if numerical_data.shape[1] == 0:
            self._scale = pd.Series(dtype="float64")
            self._means_s = pd.Series(dtype="float64")
            self._stds_s = pd.Series(dtype="float64")
            return
        
        print("\n" + "="*80)
        print("🔧 STANDARD SCALE FIT (Memory-Efficient)")
        print("="*80)
        print(f"Train samples: {len(numerical_data):,}")
        print(f"Num features: {numerical_data.shape[1]}")
        print(f"Processing strategy: column-by-column\n")
        
        # 🔧 STRATEGIA: Calcola statistiche colonna-per-colonna
        scales = []
        means = []
        stds = []
        cols = []
        
        for i, col in enumerate(numerical_data.columns):
            # ═══════════════════════════════════════════════════════
            # STEP 1: Converti e pulisci (IDENTICO all'originale)
            # ═══════════════════════════════════════════════════════
            col_data = numerical_data[col].values.astype(np.float64)
            
            # Sostituisci inf con NaN (IDENTICO)
            col_data = np.where(np.isfinite(col_data), col_data, np.nan)
            
            # ═══════════════════════════════════════════════════════
            # STEP 2: Calcola scale (IDENTICO all'originale)
            # ═══════════════════════════════════════════════════════
            abs_max = np.nanmax(np.abs(col_data))
            
            # IDENTICO alla logica originale:
            # - Se abs_max è NaN → scale = 1.0
            # - Se abs_max < 1e-10 → scale = 1.0 (evita divisione per 0)
            # - Altrimenti → scale = abs_max
            if np.isnan(abs_max) or abs_max == 0.0:
                scale = 1.0
            else:
                scale = max(abs_max, 1e-10)
            
            # ═══════════════════════════════════════════════════════
            # STEP 3: Scala e calcola mean/std (IDENTICO)
            # ═══════════════════════════════════════════════════════
            col_scaled = col_data / scale
            
            # Mean e std su dati scalati (IDENTICO)
            mean = np.nanmean(col_scaled)
            std = np.nanstd(col_scaled, ddof=0)  # ddof=0 come pandas.std()
            
            # Clip std per evitare divisione per 0 (IDENTICO)
            std = max(std, 1e-10)
            
            # Salva
            scales.append(scale)
            means.append(mean)
            stds.append(std)
            cols.append(col)
            
            # Progress ogni 10 colonne
            if (i + 1) % 10 == 0:
                print(f"   Processed {i+1}/{numerical_data.shape[1]} features...")
        
        # Converti in Series (come l'originale)
        self._scale = pd.Series(scales, index=cols)
        self._means_s = pd.Series(means, index=cols)
        self._stds_s = pd.Series(stds, index=cols)
        
        print(f"\n✅ Fit completato:")
        print(f"   Scale range: [{self._scale.min():.6e}, {self._scale.max():.6e}]")
        print(f"   Means range: [{self._means_s.min():.6f}, {self._means_s.max():.6f}]")
        print(f"   Stds range: [{self._stds_s.min():.6f}, {self._stds_s.max():.6f}]")
        print("="*80 + "\n")
    
    @Step.requires(root=pd.DataFrame)
    @Step.provides(root=pd.DataFrame)
    def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Apply standardization to numerical columns (memory-efficient)."""
        numerical_data = root.tab.numerical
        
        if numerical_data.shape[1] == 0:
            return {"root": root}
        
        print(f"[StandardScale] Applying to {len(numerical_data):,} samples...")
        
        # ═══════════════════════════════════════════════════════════
        # VERSIONE MEMORY-EFFICIENT: Processa colonna-per-colonna
        # ═══════════════════════════════════════════════════════════
        # Per dataset grandi (100M+), evitiamo di creare DataFrame temporanei
        # Modifichiamo direttamente le colonne in-place
        
        for col in numerical_data.columns:
            # Recupera parametri di scaling
            scale = self._scale.get(col, 1.0)
            mean = self._means_s.get(col, 0.0)
            std = self._stds_s.get(col, 1.0)
            
            # Converti e pulisci (IDENTICO)
            col_data = numerical_data[col].values.astype(np.float64)
            col_data = np.where(np.isfinite(col_data), col_data, np.nan)
            
            # Applica standardization (IDENTICO alla formula originale)
            col_transformed = (col_data / scale - mean) / std
            
            # Aggiorna in-place (memory-efficient)
            numerical_data[col] = col_transformed
        
        root.tab.numerical = numerical_data
        
        print(f"[StandardScale] Done. Range: [{numerical_data.min().min():.2f}, {numerical_data.max().max():.2f}]")
        
        return {"root": root}


class StandardScale2(FitAwareStep):  
    """Standardize numerical columns using mean/std with memory-efficient and stable scaling."""

    def __init__(
        self,
        in_scope: str = "data",
        out_scope: str = "data",
        name: Optional[str] = None,
    ) -> None:
        self._scale: Optional[pd.Series] = None
        self._means: Optional[pd.Series] = None
        self._stds: Optional[pd.Series] = None
                       
        super().__init__(
            name=name or "standard_scale",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    @Step.requires(root=pd.DataFrame)
    def fit_impl(self, state: State, root: pd.DataFrame) -> None:
        """Fit scaling stats on train split (memory-efficient + numerically stable)."""
        numerical_data = root.tab.train.tab.numerical
        
        if numerical_data.shape[1] == 0:
            self._scale = pd.Series(dtype="float64")
            self._means = pd.Series(dtype="float64")
            self._stds = pd.Series(dtype="float64")
            return

        # 🔍 TEST RAPIDO: Verifica colonne con valori sospetti PRIMA del processing
        print("\n" + "="*80)
        print("🔍 DEBUG PRE-SCALING: Analisi valori estremi")
        print("="*80)
        suspicious_cols = []
        for col in numerical_data.columns:
            abs_max = numerical_data[col].abs().max()
            if abs_max > 1e6:
                suspicious_cols.append((col, abs_max))
                print(f"⚠️  {col}: max={abs_max:.2e}")
        
        if not suspicious_cols:
            print("✅ Nessuna colonna con valori estremi rilevata")
        else:
            print(f"\n⚠️  Trovate {len(suspicious_cols)} colonne con valori > 1e6")
        print("="*80 + "\n")
        
        # 🔧 FIX CRITICO: Processa colonna per colonna per evitare OOM
        scales = []
        means = []
        stds = []
        cols = []
        
        for col in numerical_data.columns:
            # Lavora su una colonna alla volta (molto più memory-efficient)
            col_data = numerical_data[col].values  # Usa numpy array direttamente
            
            # Converti a float64 in-place
            if col_data.dtype != np.float64:
                col_data = col_data.astype(np.float64)
            
            # Rimuovi inf/nan
            col_data = np.where(np.isfinite(col_data), col_data, np.nan)
            
            # 🔧 FIX STABILITÀ: Usa log-scale per valori estremi
            abs_max = np.nanmax(np.abs(col_data))
            
            # Se i valori sono troppo grandi (>1e10), usa log-transform
            if abs_max > 1e10:
                print(f"⚠️  Colonna '{col}' ha valori estremi (max={abs_max:.2e}), applico log-transform")
                # Log-transform per stabilizzare (aggiungi 1 per evitare log(0))
                col_data = np.sign(col_data) * np.log1p(np.abs(col_data))
                abs_max = np.nanmax(np.abs(col_data))
            
            scale = max(abs_max, 1e-10)  # Evita divisione per zero
            
            # Scala e calcola statistiche
            col_scaled = col_data / scale
            mean = np.nanmean(col_scaled)
            std = np.nanstd(col_scaled)
            std = max(std, 1e-10)  # Evita divisione per zero
            
            scales.append(scale)
            means.append(mean)
            stds.append(std)
            cols.append(col)
        
        self._scale = pd.Series(scales, index=cols)
        self._means = pd.Series(means, index=cols)
        self._stds = pd.Series(stds, index=cols)
        
        print(f"[StandardScale] Fit completato:")
        print(f"  Scale range: [{self._scale.min():.6e}, {self._scale.max():.6e}]")
        print(f"  Means range: [{self._means.min():.6f}, {self._means.max():.6f}]")
        print(f"  Stds range: [{self._stds.min():.6f}, {self._stds.max():.6f}]")

    @Step.requires(root=pd.DataFrame)
    @Step.provides(root=pd.DataFrame)
    def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Apply standardization to numerical columns."""
        numerical_data = root.tab.numerical
        
        if numerical_data.shape[1] == 0:
            return {"root": root}

        # 🔧 Processa colonna per colonna (memory-efficient)
        transformed = {}
        
        for col in numerical_data.columns:
            col_data = numerical_data[col].values.astype(np.float64)
            
            # Rimuovi inf/nan
            col_data = np.where(np.isfinite(col_data), col_data, np.nan)
            
            # Recupera parametri di scaling
            scale = self._scale.get(col, 1.0)
            mean = self._means.get(col, 0.0)
            std = self._stds.get(col, 1.0)
            
            # 🔧 Se questa colonna aveva valori estremi nel fit, applica lo stesso log-transform
            if scale > 1e10:  # Euristica: se scale era grande, era stato fatto log-transform
                col_data = np.sign(col_data) * np.log1p(np.abs(col_data))
            
            # Standardizza
            col_transformed = (col_data / scale - mean) / std
            
            # 🔧 Clipping moderato (±10 è sufficiente, ±100 è troppo)
            col_transformed = np.clip(col_transformed, -10.0, 10.0)
            
            # Riempi NaN con 0
            col_transformed = np.nan_to_num(col_transformed, nan=0.0)
            
            transformed[col] = col_transformed
        
        # Crea nuovo DataFrame
        root.tab.numerical = pd.DataFrame(transformed, index=numerical_data.index)
        
        final_data = root.tab.numerical
        print(f"[StandardScale] Run completato:")
        print(f"  Range valori: [{final_data.min().min():.6f}, {final_data.max().max():.6f}]")
        print(f"  NaN totali: {final_data.isna().sum().sum():,}")
        
        return {"root": root}
# class StandardScale2(FitAwareStep):  
#     """Standardize numerical columns using mean/std with overflow-safe scaling."""

#     def __init__(
#         self,
#         in_scope: str = "data",
#         out_scope: str = "data",
#         name: Optional[str] = None,
#     ) -> None:
#         self._scale: Optional[pd.Series] = None
#         self._means_s: Optional[pd.Series] = None
#         self._stds_s: Optional[pd.Series] = None
                       
#         super().__init__(
#             name=name or "standard_scale",
#             in_scope=in_scope,
#             out_scope=out_scope,
#         )

#     @Step.requires(root=pd.DataFrame)
#     def fit_impl(self, state: State, root: pd.DataFrame) -> None:
#         """Fit scaling stats on train split (overflow-safe)."""
#         numerical_data = root.tab.train.tab.numerical
        
#         if numerical_data.shape[1] == 0:
#             # 🔧 Mantieni float64 per coerenza
#             self._scale = pd.Series(dtype="float64")
#             self._means_s = pd.Series(dtype="float64")
#             self._stds_s = pd.Series(dtype="float64")
#             return

#         # 🔧 NON convertire a float32, mantieni float64 per calcoli
#         numerical_data = numerical_data.astype(np.float64, copy=False).replace(
#             [np.inf, -np.inf], np.nan                       
#         )

#         # Overflow-safe: compute scale and scaled values efficiently
#         abs_max = numerical_data.abs().max(axis=0)
#         self._scale = (
#             abs_max.fillna(0.0).clip(lower=1e-10, upper=None).where(abs_max > 0.0, 1.0)
#         )

#         num_scaled = numerical_data / self._scale
#         self._means_s = num_scaled.mean()
#         self._stds_s = num_scaled.std(ddof=0).clip(lower=1e-10, upper=None)
        
#         print(f"[StandardScale] Fit completato:")
#         print(f"  Scale range: [{self._scale.min():.6f}, {self._scale.max():.6f}]")
#         print(f"  Means range: [{self._means_s.min():.6f}, {self._means_s.max():.6f}]")
#         print(f"  Stds range: [{self._stds_s.min():.6f}, {self._stds_s.max():.6f}]")

#     @Step.requires(root=pd.DataFrame)
#     @Step.provides(root=pd.DataFrame)
#     def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:
#         """Apply standardization to numerical columns."""

#         numerical_data = root.tab.numerical
#         if numerical_data.shape[1] == 0:
#             return {"root": root}

#         # 🔧 Mantieni float64 per calcoli stabili
#         numerical_data = numerical_data.astype(np.float64, copy=False).replace(
#             [np.inf, -np.inf], np.nan
#         )

#         cols = numerical_data.columns
#         scale = self._scale.reindex(cols, fill_value=1.0)
#         means_s = self._means_s.reindex(cols, fill_value=0.0)
#         stds_s = self._stds_s.reindex(cols, fill_value=1.0)

#         # Applica standardizzazione in float64 (stabile)
#         root.tab.numerical = (numerical_data / scale - means_s) / stds_s
        
#         # 🔧 POST-PROCESSING: Clipping per sicurezza (evita inf/nan nel backprop)
#         final_data = root.tab.numerical
        
#         # Clipping aggressivo di valori estremi
#         final_data = final_data.clip(lower=-100.0, upper=100.0)
        
#         # Riempi NaN residui con 0
#         final_data = final_data.fillna(0.0)
        
#         root.tab.numerical = final_data
        
#         print(f"[StandardScale] Run completato:")
#         print(f"  Valori finali range: [{final_data.min().min():.6f}, {final_data.max().max():.6f}]")
#         print(f"  NaN totali: {final_data.isna().sum().sum():,}")
#         print(f"  Inf totali: {np.isinf(final_data.values).sum():,}")
        
#         return {"root": root}

#==================================================================================

# class StandardScale2(FitAwareStep):  
#     """Standardize numerical columns using mean/std with overflow-safe scaling and chunk processing."""

#     def __init__(
#         self,
#         chunk_size: int = 300000,
#         in_scope: str = "data",
#         out_scope: str = "data",
#         name: Optional[str] = None,
#     ) -> None:
#         self._chunk_size = chunk_size
#         self._scale: Optional[pd.Series] = None
#         self._means_s: Optional[pd.Series] = None
#         self._stds_s: Optional[pd.Series] = None
                       
#         super().__init__(
#             name=name or "standard_scale2",
#             in_scope=in_scope,
#             out_scope=out_scope,
#         )

#     def _process_chunk_fit(self, chunk: pd.DataFrame) -> tuple:
#         """Process a single chunk during fitting phase."""
#         if chunk.shape[1] == 0:
#             return None, None, None, 0
            
#         # Converti a float64 e gestisci valori infiniti
#         chunk = chunk.astype(np.float64, copy=False).replace(
#             [np.inf, -np.inf], np.nan                       
#         )
        
#         # Calcola statistiche per questo chunk
#         abs_max = chunk.abs().max(axis=0, skipna=True)
#         chunk_scale = (
#             abs_max.fillna(0.0).clip(lower=1e-10, upper=None).where(abs_max > 0.0, 1.0)
#         )
        
#         chunk_scaled = chunk / chunk_scale
#         chunk_sum = chunk_scaled.sum(axis=0, skipna=True)
#         chunk_sum_sq = (chunk_scaled ** 2).sum(axis=0, skipna=True)
#         chunk_count = chunk_scaled.count(axis=0)
        
#         return chunk_scale, chunk_sum, chunk_sum_sq, chunk_count

#     @Step.requires(root=pd.DataFrame)
#     def fit_impl(self, state: State, root: pd.DataFrame) -> None:
#         """Fit scaling stats on train split using chunked processing (overflow-safe)."""
#         numerical_data = root.tab.train.tab.numerical
#         print(f"[StandardScale2] Inizio fitting su {numerical_data.shape[0]:,} righe e {numerical_data.shape[1]} colonne")

#         if numerical_data.shape[1] == 0:
#             self._scale = pd.Series(dtype="float64")
#             self._means_s = pd.Series(dtype="float64")
#             self._stds_s = pd.Series(dtype="float64")
#             return

#         # Inizializza accumulatori per le statistiche
#         total_scale = None
#         total_sum = None
#         total_sum_sq = None
#         total_count = None
        
#         # Processa il dataset in chunk
#         for start_idx in range(0, len(numerical_data), self._chunk_size):
#             end_idx = min(start_idx + self._chunk_size, len(numerical_data))
#             chunk = numerical_data.iloc[start_idx:end_idx]
            
#             chunk_scale, chunk_sum, chunk_sum_sq, chunk_count = self._process_chunk_fit(chunk)
            
#             if chunk_scale is None:
#                 continue
                
#             # Accumula le statistiche
#             if total_scale is None:
#                 total_scale = chunk_scale.copy()
#                 total_sum = chunk_sum.copy()
#                 total_sum_sq = chunk_sum_sq.copy()
#                 total_count = chunk_count.copy()
#             else:
#                 # CORREZIONE: Usa np.maximum invece di pd.concat per combinare le scale
#                 total_scale = pd.Series(
#                     np.maximum(total_scale.values, chunk_scale.reindex(total_scale.index, fill_value=0).values),
#                     index=total_scale.index,
#                     name=total_scale.name
#                 )
                
#                 # Assicurati che gli indici siano allineati per le somme
#                 total_sum = total_sum.add(chunk_sum, fill_value=0)
#                 total_sum_sq = total_sum_sq.add(chunk_sum_sq, fill_value=0)
#                 total_count = total_count.add(chunk_count, fill_value=0)

#         # Salva le scale finali
#         self._scale = total_scale
        
#         # Calcola media e deviazione standard dalle statistiche accumulate
#         # Evita divisione per zero sostituendo 0 con NaN
#         valid_counts = total_count.replace(0, np.nan)
#         self._means_s = (total_sum / valid_counts).fillna(0.0)
        
#         # Calcola varianza: E[X²] - E[X]²
#         mean_sq = (total_sum_sq / valid_counts).fillna(0.0)
#         variance = (mean_sq - (self._means_s ** 2)).clip(lower=1e-20)
#         self._stds_s = np.sqrt(variance).clip(lower=1e-10)
        
#         # Gestisci casi edge: se non ci sono osservazioni valide, usa valori default
#         self._stds_s = self._stds_s.fillna(1.0)
#         print(f"[StandardScale2] Scale range: [{self._scale.min():.6f}, {self._scale.max():.6f}]")
#         print(f"[StandardScale2] Means range: [{self._means_s.min():.6f}, {self._means_s.max():.6f}]")
#         print(f"[StandardScale2] Stds range: [{self._stds_s.min():.6f}, {self._stds_s.max():.6f}]")
#         print("[StandardScale2] Fitting completato")

#     def _process_chunk_transform(self, chunk: pd.DataFrame, cols: pd.Index, 
#                                 scale: pd.Series, means_s: pd.Series, stds_s: pd.Series) -> pd.DataFrame:
#         """Process a single chunk during transformation phase."""
#         if chunk.shape[1] == 0:
#             return chunk
            
#         # Converti a float64 e gestisci valori infiniti
#         chunk = chunk.astype(np.float64, copy=False).replace(
#             [np.inf, -np.inf], np.nan
#         )
        
#         # Applica la trasformazione con reindexing sicuro
#         chunk_scale = scale.reindex(chunk.columns, fill_value=1.0)
#         chunk_means_s = means_s.reindex(chunk.columns, fill_value=0.0)
#         chunk_stds_s = stds_s.reindex(chunk.columns, fill_value=1.0)
        
#         # Applica la standardizzazione: (X/scale - mean) / std
#         transformed = (chunk / chunk_scale - chunk_means_s) / chunk_stds_s
        
#         # Gestisci eventuali risultati infiniti o NaN dalla trasformazione
#         return transformed.replace([np.inf, -np.inf], np.nan)

#     @Step.requires(root=pd.DataFrame)
#     @Step.provides(root=pd.DataFrame)
#     def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:
#         """Apply standardization to numerical columns using chunked processing."""

#         numerical_data = root.tab.numerical
#         print(f"[StandardScale2] Inizio run su {numerical_data.shape[0]:,} righe e {numerical_data.shape[1]} colonne")

#         if numerical_data.shape[1] == 0:
#             print("[StandardScale2] Nessuna colonna numerica da trasformare")
#             return {"root": root}

#         # Verifica che le statistiche siano state calcolate durante il fit
#         if self._scale is None or self._means_s is None or self._stds_s is None:
#             raise ValueError("StandardScale2 must be fitted before transformation")

#         cols = numerical_data.columns
#         scale = self._scale.reindex(cols, fill_value=1.0)
#         means_s = self._means_s.reindex(cols, fill_value=0.0)
#         stds_s = self._stds_s.reindex(cols, fill_value=1.0)
        
        
#         # SOLUZIONE MEMORY-EFFICIENT: Trasforma in-place chunk per chunk
#         # invece di accumulare tutti i chunk in memoria
#         total_rows = len(numerical_data)
        
#         for start_idx in range(0, total_rows, self._chunk_size):
#             end_idx = min(start_idx + self._chunk_size, total_rows)
            
#             # Ottieni il chunk corrente
#             chunk_indices = numerical_data.index[start_idx:end_idx]
#             chunk = numerical_data.iloc[start_idx:end_idx]
            
#             # Trasforma il chunk
#             transformed_chunk = self._process_chunk_transform(chunk, cols, scale, means_s, stds_s)
            
#             # Aggiorna direttamente il DataFrame originale (in-place)
#             # Questo evita di tenere tutti i chunk in memoria contemporaneamente
#             # root.tab.numerical.loc[chunk_indices, chunk.columns] = transformed_chunk.values
#             col_positions = [root.tab.numerical.columns.get_loc(col) for col in chunk.columns]
#             root.tab.numerical.iloc[start_idx:end_idx, col_positions] = transformed_chunk.values
            
#             # Opzionale: forza garbage collection ogni N chunk per liberare memoria
#             if (start_idx // self._chunk_size) % 10 == 0:
#                 import gc
#                 gc.collect()
            
#             final_data = root.tab.numerical
#             print(f"[StandardScale2] Trasformazione completata:")
#             print(f"[StandardScale2] - Valori finali range: [{final_data.min().min():.6f}, {final_data.max().max():.6f}]")
#             print(f"[StandardScale2] - Media delle medie per colonna: {final_data.mean().mean():.6f}")
#             print(f"[StandardScale2] - Media delle std per colonna: {final_data.std().mean():.6f}")
#             print(f"[StandardScale2] - NaN totali: {final_data.isna().sum().sum():,}")
        
#         return {"root": root}
    
    
class ZScaler(FitAwareStep):
    def __init__(
        self, 
        name: Optional[str] = None,
        in_scope: str = "data",
        out_scope: str = "data",
       ) -> None:
        self._scale: Optional[pd.Series] = None
        self._mean_s: Optional[pd.Series] = None
        self._stds_s: Optional[pd.Series] = None
        super().__init__(
            name = name or "zscale",
            in_scope = in_scope,
            out_scope = out_scope,
        )
    @Step.requires(root = pd.DataFrame)
    def fit_impl(self, state: State, root: pd.DataFrame) -> None:
        numerical_data = root.tab.train.tab.numerical
        if numerical_data.shape[1] == 0:
            self._scale = pd.Series(dtype="float64")
            self._mean_s = pd.Series(dtype="float64")
            self._stds_s = pd.Series(dtype = "float64")
            return
        
        numerical_data = numerical_data.astype(np.float64, copy = False).replace(
            [np.inf, -np.inf], np.nan
        )

        abs_max = numerical_data.abs().max(axis=0)
        self._scale = (
            abs_max.fillna(0.0).clip(lower=1e-10, upper=None).where(abs_max > 0.0, 1.0)
        )
        num_scaled = numerical_data / self._scale
        self._mean_s = num_scaled.mean()
        self._stds_s = num_scaled.std(ddof = 0).clip(lower=1e-10, upper=None)
        
    
    
    @Step.requires(root = pd.DataFrame)
    @Step.provides(root = pd.DataFrame)
    def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:
        numerical_data = root.tab.numerical
        if numerical_data.shape[1] == 0:
            return {"root":root}
        
        numerical_data = numerical_data.astype(np.float64, copy=False).replace(
            [np.inf, -np.inf], np.nan
        )
        
        cols = numerical_data.columns
        scale = self._scale.reindex(cols, fill_value=1.0)
        means_s = self._mean_s.reindex(cols, fill_value=0.0)
        stds_s = self._stds_s.reindex(cols, fill_value=1.0)
        
        root.tab.numerical = (numerical_data / scale - means_s) / stds_s
        
        clip_threshold = 3.0
        zscored = (numerical_data / scale - means_s) / stds_s

        # Conta righe outlier
        outliers = (zscored.abs() > clip_threshold).sum()
        print("[ZScaler] Outlier per colonna:", outliers.to_dict())

        print("[ZScaler] Mean prima del clipping:", zscored.mean().to_dict())
        print("[ZScaler] Std prima del clipping:", zscored.std(ddof=0).to_dict())
        print("[ZScaler] Min/Max prima del clipping:", {col: (zscored[col].min(), zscored[col].max()) for col in zscored.columns})

        # Clipping
        zscored = zscored.clip(lower=-clip_threshold, upper=clip_threshold)

        zscored = zscored.clip(lower=-clip_threshold, upper=clip_threshold)
        print("[ZScaler] Min/Max dopo clipping:", {col: (zscored[col].min(), zscored[col].max()) for col in zscored.columns})

        root.tab.numerical = zscored

        return {"root":root}
    
class MissingValueImputer(FitAwareStep):
    
    def __init__(
        self,
        name: Optional[str] = None,
        in_scope: str = "data",
        out_scope: str = "data",
       ) -> None:
        self._scale: Optional[pd.Series] = None
        self._mean_s: Optional[pd.Series] = None
        self._mode_s: Optional[pd.Series] = None
        super().__init__(
            name=name or "missing_value_imputer",
            in_scope=in_scope,
            out_scope=out_scope,
        )
        
    @Step.requires(root= pd.DataFrame)
    def fit_impl(self, state: State, root: pd.DataFrame) -> None:
        
        cat_data = root.tab.train.tab.categorical
        numerical_data = root.tab.train.tab.numerical   #estrae le colonne numeriche del DataFrame della partizione di training
        if numerical_data.shape[1] == 0:                #se non ci sono colonne numeriche, inizializza le serie vuote e ritorna
            self._scale = pd.Series(dtype="float64")        #serie vuota per i fattori di scala
            self._mean_s = pd.Series(dtype="float64")      #serie vuota per le medie
            return

        # categoriche
        if cat_data.shape[1] > 0:
            self._mode_s = cat_data.mode().iloc[0]
        else:
            self._mode_s = pd.Series(dtype="object")
        
        numerical_data = numerical_data.astype(np.float64, copy=False).replace( #converte i dati in float64 e sostituisce infiniti con NaN
            [np.inf, -np.inf], np.nan                       
        )

        # Overflow-safe: compute scale and scaled values efficiently
        abs_max = numerical_data.abs().max(axis=0)  #calcola il valore assoluto massimo per ogni colonna numerica
        self._scale = (                             #calcola i fattori di scala per ogni colonna numerica
            abs_max.fillna(0.0).clip(lower=1e-10, upper=None).where(abs_max > 0.0, 1.0) #se una colonna è tutta nan la riempie con 0 
        )   #e se è minore di 1e-10, evita le scale troppo piccole, quindi evita di dividere per 0; se il max è 0 la sostituisce con 1.0                 

        num_scaled = numerical_data / self._scale
        self._mean_s = num_scaled.mean()
        self._mode_s = num_scaled.mode().iloc[0]   
        print("[MissingValueImputer][FIT] Medie colonne numeriche:", self._mean_s.to_dict())
        print("[MissingValueImputer][FIT] Moda colonne categoriche:", self._mode_s.to_dict())
     
    
    @Step.requires(root=pd.DataFrame)
    @Step.provides(root=pd.DataFrame)
    def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:
        cat_data = root.tab.categorical
        numerical_data = root.tab.numerical
        if numerical_data.shape[1] == 0 and cat_data.shape[1]==0:
            return {"root":root}
        
        numerical_data = numerical_data.astype(np.float64, copy=False).replace(
            [np.inf, -np.inf], np.nan
        )
        
        print("[MissingValueImputer][RUN] NaN prima imputazione numeriche:", numerical_data.isna().sum().to_dict())
        print("[MissingValueImputer][RUN] NaN prima imputazione categoriche:", cat_data.isna().sum().to_dict())

        mean_s = self._mean_s.reindex(numerical_data.columns, fill_value=0.0)
        mode_s = self._mode_s.reindex(cat_data.columns, fill_value="unknown")
       
        root.tab.numerical = numerical_data.fillna(mean_s)
        root.tab.categorical = cat_data.fillna(mode_s)
        print("[MissingValueImputer][RUN] NaN dopo imputazione numeriche:", root.tab.numerical.isna().sum().to_dict())
        print("[MissingValueImputer][RUN] NaN dopo imputazione categoriche:", root.tab.categorical.isna().sum().to_dict())

        return {"root": root}


class OutlierRemover(FitAwareStep):
    
    def __init__(
        self,
        name: Optional[str] = None,
        in_scope = "data",
        out_scope = "data",
      ) -> None:
        self._medians = None
        self._q1 = None 
        self._q3 = None
        self._iqr = None
        super().__init__(
            name=name or "outlier_remover",
            in_scope=in_scope,
            out_scope=out_scope
        )
    @Step.requires(root=pd.DataFrame)
    def fit_impl(self, state:State, root:pd.DataFrame) -> Optional[Dict[str, Any]]:
        numerical_data = root.tab.train.tab.numerical
        # if numerical_data.shape[1] == 0:
        self._medians = numerical_data.median()
        self._q1 = numerical_data.quantile(0.25)
        self._q3 = numerical_data.quantile(0.75)
        self._iqr = self._q3 - self._q1
            # return
        
        numerical_data = numerical_data.astype(np.float64, copy = False).replace(
            [np.inf, -np.inf], np.nan
        )

        abs_max = numerical_data.abs().max(axis=0)
        self._scale = (
            abs_max.fillna(0.0).clip(lower=1e-10, upper=None).where(abs_max > 0.0, 1.0)
        )

        self._medians = numerical_data.median()
        print("[OutlierRemover][FIT] Q1:", self._q1.to_dict())
        print("[OutlierRemover][FIT] Q3:", self._q3.to_dict())
        print("[OutlierRemover][FIT] IQR:", self._iqr.to_dict())
             
    @Step.requires(root=pd.DataFrame)
    @Step.provides(root=pd.DataFrame)
    def run(self, state:State, root:pd.DataFrame) -> Optional[Dict[str, Any]]:
        
        numerical_data = root.tab.numerical
        if numerical_data.shape[1] == 0:
            return {"root":root}
        
        numerical_data = numerical_data.astype(np.float64, copy=False).replace(
            [np.inf, -np.inf], np.nan
        )
        
        lower = self._q1 - 1.5*self._iqr
        upper = self._q3 + 1.5*self._iqr
        
        mask_outliers = (numerical_data < lower) | (numerical_data > upper)
        outliers_count = mask_outliers.sum()
        
        print("[OutlierRemover][RUN] outlier count per column:", outliers_count.to_dict())
        
        for col in numerical_data.columns:
            numerical_data.loc[mask_outliers[col], col] = self._medians[col]
            
        root.tab.numerical = numerical_data
        
        return {"root": root}