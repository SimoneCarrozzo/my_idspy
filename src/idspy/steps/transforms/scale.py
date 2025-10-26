from typing import Optional, Dict, Any
import warnings #Optional indica che una variabile può essere di un certo tipo oppure None finchè non viene inizializzata

import numpy as np
import pandas as pd

from ...core.step import FitAwareStep, Step
from ...core.state import State

################    HO AGGIUNTO IO

import gc

# Disabilita i warning di overflow per non inquinare i log
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")


class StandardScale1(FitAwareStep):
    """Ultra-robust standardization that handles extreme values and memory constraints."""

    def __init__(
        self, 
        chunk_size: int = 300_000,  # Aumentato come richiesto
        dtype: str = "float32",     # Usa float32 per ridurre memoria
        in_scope: str = "data", 
        out_scope: str = "data", 
        name: Optional[str] = None
    ) -> None:
        self.chunk_size = chunk_size
        self.dtype = getattr(np, dtype)
        self._scale: Optional[pd.Series] = None
        self._means: Optional[pd.Series] = None
        self._stds: Optional[pd.Series] = None
        
        super().__init__(
            name=name or "standard_scale",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    def _sanitize_chunk(self, chunk: pd.DataFrame) -> np.ndarray:
        """Pulisce e converte chunk in numpy array sicuro."""
        # Converte a numpy direttamente per evitare operazioni pandas costose
        chunk_array = chunk.values.astype(self.dtype, copy=False)
        
        # Sostituisce inf e valori estremi con NaN
        chunk_array = np.where(
            (np.abs(chunk_array) > 1e30) | (~np.isfinite(chunk_array)), 
            np.nan, 
            chunk_array
        )
        
        return chunk_array

    def _compute_chunk_stats(self, chunk_array: np.ndarray) -> Dict[str, np.ndarray]:
        """Calcola statistiche robuste per un chunk."""
        n_rows, n_cols = chunk_array.shape
        
        # Maschera per valori validi
        valid_mask = ~np.isnan(chunk_array)
        
        # Conteggi
        counts = np.sum(valid_mask, axis=0)
        
        # Somme (usa nansum per sicurezza)
        sums = np.nansum(chunk_array, axis=0)
        
        # Somme dei quadrati - usa clipping per evitare overflow
        chunk_clipped = np.clip(chunk_array, -1e15, 1e15)  # Limita range
        sum_squares = np.nansum(chunk_clipped ** 2, axis=0)
        
        # Massimi assoluti
        abs_max = np.nanmax(np.abs(chunk_array), axis=0)
        abs_max = np.nan_to_num(abs_max, nan=0.0)
        
        return {
            'count': counts.astype(np.int64),
            'sum': sums.astype(np.float64),
            'sum_sq': sum_squares.astype(np.float64), 
            'abs_max': abs_max.astype(self.dtype)
        }

    def _iter_chunks_safe(self, df: pd.DataFrame):
        """Iterator sicuro per chunk grandi."""
        n_rows = len(df)
        for start in range(0, n_rows, self.chunk_size):
            end = min(start + self.chunk_size, n_rows)
            
            # Estrai chunk come numpy array direttamente 
            chunk = df.iloc[start:end]
            
            # Pulisci attrs per evitare deepcopy issues
            if hasattr(chunk, 'attrs'):
                chunk.attrs.clear()
                
            yield chunk
            
            # Memory cleanup ogni 20 chunk (meno frequente)
            if (start // self.chunk_size) % 20 == 19:
                gc.collect()

    @Step.requires(root=pd.DataFrame)
    def fit_impl(self, state: State, root: pd.DataFrame) -> None:
        """Fit robusto con gestione overflow."""
        
        numerical_data = root.tab.train.tab.numerical
        
        if numerical_data.shape[1] == 0:
            self._scale = pd.Series(dtype=self.dtype)
            self._means = pd.Series(dtype=self.dtype)
            self._stds = pd.Series(dtype=self.dtype)
            return

        print(f"[StandardScale] Fitting su {len(numerical_data):,} righe, {len(numerical_data.columns)} colonne")
        print(f"[StandardScale] Chunk size: {self.chunk_size:,}")
        
        n_cols = numerical_data.shape[1]
        col_names = numerical_data.columns
        
        # Accumulatori robusti
        total_count = np.zeros(n_cols, dtype=np.int64)
        total_sum = np.zeros(n_cols, dtype=np.float64)
        total_sum_sq = np.zeros(n_cols, dtype=np.float64)
        global_abs_max = np.zeros(n_cols, dtype=self.dtype)
        
        # Processa chunk
        chunk_idx = 0
        for chunk in self._iter_chunks_safe(numerical_data):
            chunk_idx += 1
            
            # Progress ogni 50 chunk (meno verbose)
            if chunk_idx % 50 == 0:
                print(f"[StandardScale] Chunk {chunk_idx} processato...")
            
            # Sanitizza e calcola stats
            chunk_array = self._sanitize_chunk(chunk)
            stats = self._compute_chunk_stats(chunk_array)
            
            # Accumula in modo sicuro
            total_count += stats['count']
            total_sum += stats['sum']
            total_sum_sq += stats['sum_sq'] 
            global_abs_max = np.maximum(global_abs_max, stats['abs_max'])
        
        # Calcola statistiche finali robuste
        
        # Scale: usa percentile se abs_max è troppo grande
        scale_array = np.maximum(global_abs_max, 1e-8)  # Soglia più permissiva
        scale_array = np.clip(scale_array, 1e-8, 1e10)  # Limita range estremo
        self._scale = pd.Series(scale_array, index=col_names, dtype=self.dtype)
        
        # Media robusta
        valid_counts = np.maximum(total_count, 1)
        means_array = np.divide(total_sum, valid_counts, 
                               out=np.zeros(n_cols, dtype=np.float64),
                               where=valid_counts>0)
        means_array = np.nan_to_num(means_array, nan=0.0)
        means_array = np.clip(means_array, -1e6, 1e6)  # Clamp medie estreme
        self._means = pd.Series(means_array, index=col_names, dtype=self.dtype)
        
        # Std robusta  
        variance = np.divide(total_sum_sq, valid_counts,
                            out=np.zeros(n_cols, dtype=np.float64),
                            where=valid_counts>0) - means_array**2
        variance = np.maximum(variance, 0.0)  # Forza non-negativo
        variance = np.clip(variance, 0.0, 1e10)  # Limita varianza estrema
        
        std_array = np.sqrt(variance)
        std_array = np.maximum(std_array, 1e-8)  # Evita std=0
        std_array = np.minimum(std_array, 1e6)   # Limita std estremi
        self._stds = pd.Series(std_array, index=col_names, dtype=self.dtype)
        
        print(f"[StandardScale] Fit completato.")
        print(f"Scale range: [{self._scale.min():.6f}, {self._scale.max():.6f}]")
        print(f"Std range: [{self._stds.min():.6f}, {self._stds.max():.6f}]")

    @Step.requires(root=pd.DataFrame)
    @Step.provides(root=pd.DataFrame)
    def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Applica scaling in modo memory-safe."""
        
        numerical_data = root.tab.numerical
        if numerical_data.shape[1] == 0:
            return {"root": root}
        
        print(f"[StandardScale] Applying scaling to {len(numerical_data):,} rows")
        
        # Allinea parametri alle colonne correnti
        cols = numerical_data.columns
        scale = self._scale.reindex(cols, fill_value=1.0).values.astype(self.dtype)
        means = self._means.reindex(cols, fill_value=0.0).values.astype(self.dtype)
        stds = self._stds.reindex(cols, fill_value=1.0).values.astype(self.dtype)
        
        # Applica scaling chunk per chunk con numpy puro (più veloce)
        result_chunks = []
        chunk_idx = 0
        
        for chunk in self._iter_chunks_safe(numerical_data):
            chunk_idx += 1
            
            # Progress meno frequente
            if chunk_idx % 100 == 0:
                print(f"[StandardScale] Scaling chunk {chunk_idx}...")
            
            # Converte a numpy e pulisce
            chunk_array = self._sanitize_chunk(chunk)
            
            # Applica standardizzazione con numpy (evita operazioni pandas)
            chunk_scaled = (chunk_array / scale - means) / stds
            
            # Clamp valori estremi post-scaling
            chunk_scaled = np.clip(chunk_scaled, -10.0, 10.0)
            chunk_scaled = np.nan_to_num(chunk_scaled, nan=0.0)
            
            # Ricrea DataFrame con attributi puliti
            chunk_df = pd.DataFrame(
                chunk_scaled, 
                columns=cols,
                dtype=self.dtype
            )
            
            result_chunks.append(chunk_df)
        
        print("[StandardScale] Concatenating results...")
        
        # Concatenazione più efficiente
        try:
            numerical_scaled = pd.concat(result_chunks, axis=0, ignore_index=True, copy=False)
        except Exception as e:
            print(f"[StandardScale] Errore concatenazione: {e}")
            # Fallback: concatena chunk più piccoli
            print("[StandardScale] Fallback: concatenazione in batch...")
            batched_chunks = []
            batch_size = 10
            
            for i in range(0, len(result_chunks), batch_size):
                batch = result_chunks[i:i+batch_size]
                batched_chunk = pd.concat(batch, axis=0, ignore_index=True, copy=False)
                batched_chunks.append(batched_chunk)
                del batch
                gc.collect()
            
            numerical_scaled = pd.concat(batched_chunks, axis=0, ignore_index=True, copy=False)
            del batched_chunks
        
        # Assegna risultato
        root.tab.numerical = numerical_scaled
        
        # Cleanup completo
        del result_chunks
        gc.collect()
        
        print("[StandardScale] Scaling completato!")
        return {"root": root}



########### FINE AGGIUNTA

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
        """Fit scaling stats on train split (overflow-safe)."""
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










warnings.filterwarnings("ignore", category=FutureWarning)


class StandardScale2(FitAwareStep):  
    """Standardize numerical columns using mean/std with overflow-safe scaling and chunk processing."""

    def __init__(
        self,
        chunk_size: int = 300000,
        in_scope: str = "data",
        out_scope: str = "data",
        name: Optional[str] = None,
    ) -> None:
        self._chunk_size = chunk_size
        self._scale: Optional[pd.Series] = None
        self._means_s: Optional[pd.Series] = None
        self._stds_s: Optional[pd.Series] = None
                       
        super().__init__(
            name=name or "standard_scale2",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    def _process_chunk_fit(self, chunk: pd.DataFrame) -> tuple:
        """Process a single chunk during fitting phase."""
        if chunk.shape[1] == 0:
            return None, None, None, 0
            
        # Converti a float64 e gestisci valori infiniti
        chunk = chunk.astype(np.float64, copy=False).replace(
            [np.inf, -np.inf], np.nan                       
        )
        
        # Calcola statistiche per questo chunk
        abs_max = chunk.abs().max(axis=0, skipna=True)
        chunk_scale = (
            abs_max.fillna(0.0).clip(lower=1e-10, upper=None).where(abs_max > 0.0, 1.0)
        )
        
        chunk_scaled = chunk / chunk_scale
        chunk_sum = chunk_scaled.sum(axis=0, skipna=True)
        chunk_sum_sq = (chunk_scaled ** 2).sum(axis=0, skipna=True)
        chunk_count = chunk_scaled.count(axis=0)
        
        return chunk_scale, chunk_sum, chunk_sum_sq, chunk_count

    @Step.requires(root=pd.DataFrame)
    def fit_impl(self, state: State, root: pd.DataFrame) -> None:
        """Fit scaling stats on train split using chunked processing (overflow-safe)."""
        numerical_data = root.tab.train.tab.numerical
        print(f"[StandardScale2] Inizio fitting su {numerical_data.shape[0]:,} righe e {numerical_data.shape[1]} colonne")

        if numerical_data.shape[1] == 0:
            self._scale = pd.Series(dtype="float64")
            self._means_s = pd.Series(dtype="float64")
            self._stds_s = pd.Series(dtype="float64")
            return

        # Inizializza accumulatori per le statistiche
        total_scale = None
        total_sum = None
        total_sum_sq = None
        total_count = None
        
        # Processa il dataset in chunk
        for start_idx in range(0, len(numerical_data), self._chunk_size):
            end_idx = min(start_idx + self._chunk_size, len(numerical_data))
            chunk = numerical_data.iloc[start_idx:end_idx]
            
            chunk_scale, chunk_sum, chunk_sum_sq, chunk_count = self._process_chunk_fit(chunk)
            
            if chunk_scale is None:
                continue
                
            # Accumula le statistiche
            if total_scale is None:
                total_scale = chunk_scale.copy()
                total_sum = chunk_sum.copy()
                total_sum_sq = chunk_sum_sq.copy()
                total_count = chunk_count.copy()
            else:
                # CORREZIONE: Usa np.maximum invece di pd.concat per combinare le scale
                total_scale = pd.Series(
                    np.maximum(total_scale.values, chunk_scale.reindex(total_scale.index, fill_value=0).values),
                    index=total_scale.index,
                    name=total_scale.name
                )
                
                # Assicurati che gli indici siano allineati per le somme
                total_sum = total_sum.add(chunk_sum, fill_value=0)
                total_sum_sq = total_sum_sq.add(chunk_sum_sq, fill_value=0)
                total_count = total_count.add(chunk_count, fill_value=0)

        # Salva le scale finali
        self._scale = total_scale
        
        # Calcola media e deviazione standard dalle statistiche accumulate
        # Evita divisione per zero sostituendo 0 con NaN
        valid_counts = total_count.replace(0, np.nan)
        self._means_s = (total_sum / valid_counts).fillna(0.0)
        
        # Calcola varianza: E[X²] - E[X]²
        mean_sq = (total_sum_sq / valid_counts).fillna(0.0)
        variance = (mean_sq - (self._means_s ** 2)).clip(lower=1e-20)
        self._stds_s = np.sqrt(variance).clip(lower=1e-10)
        
        # Gestisci casi edge: se non ci sono osservazioni valide, usa valori default
        self._stds_s = self._stds_s.fillna(1.0)
        print(f"[StandardScale2] Scale range: [{self._scale.min():.6f}, {self._scale.max():.6f}]")
        print(f"[StandardScale2] Means range: [{self._means_s.min():.6f}, {self._means_s.max():.6f}]")
        print(f"[StandardScale2] Stds range: [{self._stds_s.min():.6f}, {self._stds_s.max():.6f}]")
        print("[StandardScale2] Fitting completato")

    def _process_chunk_transform(self, chunk: pd.DataFrame, cols: pd.Index, 
                                scale: pd.Series, means_s: pd.Series, stds_s: pd.Series) -> pd.DataFrame:
        """Process a single chunk during transformation phase."""
        if chunk.shape[1] == 0:
            return chunk
            
        # Converti a float64 e gestisci valori infiniti
        chunk = chunk.astype(np.float64, copy=False).replace(
            [np.inf, -np.inf], np.nan
        )
        
        # Applica la trasformazione con reindexing sicuro
        chunk_scale = scale.reindex(chunk.columns, fill_value=1.0)
        chunk_means_s = means_s.reindex(chunk.columns, fill_value=0.0)
        chunk_stds_s = stds_s.reindex(chunk.columns, fill_value=1.0)
        
        # Applica la standardizzazione: (X/scale - mean) / std
        transformed = (chunk / chunk_scale - chunk_means_s) / chunk_stds_s
        
        # Gestisci eventuali risultati infiniti o NaN dalla trasformazione
        return transformed.replace([np.inf, -np.inf], np.nan)

    @Step.requires(root=pd.DataFrame)
    @Step.provides(root=pd.DataFrame)
    def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Apply standardization to numerical columns using chunked processing."""

        numerical_data = root.tab.numerical
        print(f"[StandardScale2] Inizio run su {numerical_data.shape[0]:,} righe e {numerical_data.shape[1]} colonne")

        if numerical_data.shape[1] == 0:
            print("[StandardScale2] Nessuna colonna numerica da trasformare")
            return {"root": root}

        # Verifica che le statistiche siano state calcolate durante il fit
        if self._scale is None or self._means_s is None or self._stds_s is None:
            raise ValueError("StandardScale2 must be fitted before transformation")

        cols = numerical_data.columns
        scale = self._scale.reindex(cols, fill_value=1.0)
        means_s = self._means_s.reindex(cols, fill_value=0.0)
        stds_s = self._stds_s.reindex(cols, fill_value=1.0)
        
        
        # SOLUZIONE MEMORY-EFFICIENT: Trasforma in-place chunk per chunk
        # invece di accumulare tutti i chunk in memoria
        total_rows = len(numerical_data)
        
        for start_idx in range(0, total_rows, self._chunk_size):
            end_idx = min(start_idx + self._chunk_size, total_rows)
            
            # Ottieni il chunk corrente
            chunk_indices = numerical_data.index[start_idx:end_idx]
            chunk = numerical_data.iloc[start_idx:end_idx]
            
            # Trasforma il chunk
            transformed_chunk = self._process_chunk_transform(chunk, cols, scale, means_s, stds_s)
            
            # Aggiorna direttamente il DataFrame originale (in-place)
            # Questo evita di tenere tutti i chunk in memoria contemporaneamente
            # root.tab.numerical.loc[chunk_indices, chunk.columns] = transformed_chunk.values
            col_positions = [root.tab.numerical.columns.get_loc(col) for col in chunk.columns]
            root.tab.numerical.iloc[start_idx:end_idx, col_positions] = transformed_chunk.values
            
            # Opzionale: forza garbage collection ogni N chunk per liberare memoria
            if (start_idx // self._chunk_size) % 10 == 0:
                import gc
                gc.collect()
            
            final_data = root.tab.numerical
            print(f"[StandardScale2] Trasformazione completata:")
            print(f"[StandardScale2] - Valori finali range: [{final_data.min().min():.6f}, {final_data.max().max():.6f}]")
            print(f"[StandardScale2] - Media delle medie per colonna: {final_data.mean().mean():.6f}")
            print(f"[StandardScale2] - Media delle std per colonna: {final_data.std().mean():.6f}")
            print(f"[StandardScale2] - NaN totali: {final_data.isna().sum().sum():,}")
        
        return {"root": root}
    
    
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