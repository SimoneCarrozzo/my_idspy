from typing import Optional, Dict, List, Tuple, Any

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from ...core.step import Step
from ...core.state import State

import logging
import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


class ClassificationMetrics(Step):
    """Compute metrics for multiclass classification."""

    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_prefix: str = "test",
        in_scope: str = "test",
        out_scope: str = "test",
        name: Optional[str] = None,
    ) -> None:
        self.writer: Optional[SummaryWriter] = (
            SummaryWriter(log_dir) if log_dir else None
        )
        self.log_prefix = log_prefix

        super().__init__(
            name=name or "multiclass_classification_metrics",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    def compute_metrics(self, y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """Compute classification metrics."""
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            confusion_matrix,
        )

        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_micro = f1_score(y_true, y_pred, average="micro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")

        cm = confusion_matrix(y_true, y_pred)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "f1_weighted": f1_weighted,
            "f1_per_class": f1_per_class,
            "confusion_matrix": cm,
        }

        return metrics

    @Step.requires(
        predictions=np.ndarray,
        targets=np.ndarray,
    )
    @Step.provides(metrics=dict)
    def run(self, state: State, predictions: np.ndarray, targets: np.ndarray) -> None:
        metrics = self.compute_metrics(predictions, targets)

        for name, value in metrics.items():
            if isinstance(value, (int, float)) and self.writer is not None:
                self.writer.add_scalar(f"{self.log_prefix}/{name}", value)

        if self.writer is not None:
            self.writer.close()

        return {"metrics": metrics}





"""
Step per Visualizzazione Metriche con Matplotlib
=================================================
Questo step legge le metriche salvate durante il training e genera
grafici professionali usando Matplotlib.

Caratteristiche:
- Grafici multi-metrica (Accuracy, Precision, Recall, F1)
- Stile pubblicazione scientifica
- Salvataggio in alta risoluzione (300 DPI)
- Supporto per metriche per classe e globali
"""
logger = logging.getLogger(__name__)

class PlotMetrics(Step):
    """
    Step per generare grafici delle metriche di training e validazione.
    
    Parametri
    ---------
    metrics_dir : str o Path
        Directory contenente i file delle metriche salvate
    output_dir : str o Path
        Directory dove salvare i grafici generati
    metrics_to_plot : list of str, optional
        Lista delle metriche da plottare. Default: ["accuracy", "precision", "recall", "f1"]
    figsize : tuple, default=(12, 8)
        Dimensione della figura in pollici (larghezza, altezza)
    dpi : int, default=300
        Risoluzione per il salvataggio (DPI - dots per inch)
    style : str, default="seaborn-v0_8-darkgrid"
        Stile di Matplotlib da usare
    show_plots : bool, default=False
        Se True, mostra i plot interattivamente (oltre a salvarli)
    """
    
    def __init__(
        self,
        metrics_dir: str,
        output_dir: str,
        metrics_to_plot: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 300,
        style: str = "seaborn-v0_8-darkgrid",
        show_plots: bool = False,
        in_scope: str = "test",          
        out_scope: str = "plot",         
        name: Optional[str] = None,       
    ) -> None:
        super().__init__(
            name=name or "plot_metrics",  
            in_scope=in_scope,            
            out_scope=out_scope,          
        )
        self.metrics_dir = Path(metrics_dir)
        self.output_dir = Path(output_dir)
        self.metrics_to_plot = metrics_to_plot or ["accuracy", "precision", "recall", "f1_micro", "f1_macro", "f1_weighted"]
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        self.show_plots = show_plots
        
        # Crea la directory di output se non esiste
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurazione estetica
        self._setup_plot_style()
    
    def _setup_plot_style(self):
        """
        Configura lo stile dei grafici per qualità pubblicazione.
        """
        try:
            plt.style.use(self.style)
        except:
            logger.warning(f"Stile '{self.style}' non disponibile, uso default")
        
        # Configurazione globale per font e dimensioni
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 12,
            'figure.titlesize': 18,
            'lines.linewidth': 2,
            'lines.markersize': 8,
        })
    
    def _load_metrics_from_files(self) -> Dict[int, Dict]:
        """
        Carica le metriche da file TensorBoard, JSON o Pickle.
        
        Supporta:
        - TensorBoard events files (events.out.tfevents.*)
        - JSON (.json)
        - Pickle (.pkl, .pickle, .0, .1, .2, ecc.)
        """
        metrics_by_epoch = {}
        
        # 1️⃣ Prova prima TensorBoard
        event_files = list(self.metrics_dir.glob("events.out.tfevents.*"))
        
        if event_files:
            logger.info(f"🔍 Trovati {len(event_files)} file TensorBoard")
            
            try:
                from tensorboard.backend.event_processing import event_accumulator
                
                for event_file in sorted(event_files):
                    try:
                        ea = event_accumulator.EventAccumulator(str(event_file))
                        ea.Reload()
                        
                        # Estrai metriche scalari
                        for tag in ea.Tags().get('scalars', []):
                            events = ea.Scalars(tag)
                            
                            for event in events:
                                epoch = event.step
                                
                                if epoch not in metrics_by_epoch:
                                    metrics_by_epoch[epoch] = {}
                                
                                # Pulisci il nome della metrica
                                metric_name = tag.replace("test/", "").replace("val/", "").replace("train/", "")
                                metrics_by_epoch[epoch][metric_name] = event.value
                        
                        logger.debug(f"✅ Caricato {event_file.name}")
                    
                    except Exception as e:
                        logger.warning(f"⚠️ Errore caricando {event_file.name}: {e}")
            
            except ImportError:
                logger.warning("⚠️ TensorBoard non installato. Installa con: pip install tensorboard")
        
        # 2️⃣ Se non ci sono metriche TensorBoard, prova JSON/Pickle
        if not metrics_by_epoch:
            logger.info("🔍 Cerco file JSON/Pickle...")
            
            all_files = [f for f in self.metrics_dir.glob("*") if f.is_file() and not f.name.startswith("events")]
            
            for file_path in sorted(all_files):
                try:
                    # JSON
                    if file_path.suffix == '.json':
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                    
                    # Pickle
                    else:
                        import pickle
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                    
                    # Estrai epoca e metriche
                    if isinstance(data, dict):
                        epoch_num = self._extract_epoch_number(file_path.stem, data)
                        if epoch_num is not None:
                            metrics_by_epoch[epoch_num] = data
                            logger.debug(f"✅ Caricato {file_path.name}")
                
                except Exception as e:
                    logger.debug(f"⚠️ Errore caricando {file_path.name}: {e}")
        
        if not metrics_by_epoch:
            logger.error("❌ Nessuna metrica trovata in nessun formato!")
        else:
            logger.info(f"✅ Caricate metriche per {len(metrics_by_epoch)} epoche")
        
        return metrics_by_epoch
        
    def _extract_epoch_number(self, filename: str, data: Dict) -> Optional[int]:
        """
        Estrae il numero dell'epoca dal nome del file o dai dati.
        """
        # Prova dal nome del file
        import re
        match = re.search(r'epoch[_\s]*(\d+)', filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        # Prova dai dati
        if 'epoch' in data:
            return int(data['epoch'])
        
        # Fallback: usa il timestamp o la posizione
        return None
    
    def _extract_metric_values(self, metrics_by_epoch: Dict) -> pd.DataFrame:
        """
        Estrae i valori delle metriche e li organizza in un DataFrame.
        """
        records = []
        
        for epoch, metrics_dict in sorted(metrics_by_epoch.items()):
            for metric_name in self.metrics_to_plot:
                # 🆕 Prova diverse varianti del nome
                possible_keys = [
                    metric_name,                    # es. "accuracy"
                    f"test/{metric_name}",          # TensorBoard format
                    f"test_{metric_name}",          # es. "test_accuracy"
                    f"macro_{metric_name}",         
                    f"{metric_name}_macro",         
                    f"weighted_{metric_name}",      
                    f"{metric_name}_weighted",      
                    f"micro_{metric_name}",         
                    f"{metric_name}_micro",
                ]
                
                value = None
                for key in possible_keys:
                    if key in metrics_dict:
                        value = metrics_dict[key]
                        break
                
                if value is not None:
                    try:
                        records.append({
                            'epoch': epoch,
                            'metric': metric_name,
                            'value': float(value)
                        })
                    except (ValueError, TypeError):
                        logger.warning(f"⚠️ Valore non numerico per {metric_name}: {value}")
        
        if not records:
            logger.warning("⚠️ Nessuna metrica estratta!")
            # 🆕 STAMPA le chiavi disponibili per debug
            if metrics_by_epoch:
                first_epoch_data = next(iter(metrics_by_epoch.values()))
                logger.info(f"📋 Chiavi disponibili: {list(first_epoch_data.keys())}")
        
        df = pd.DataFrame(records)
        if not df.empty:
            logger.info(f"✅ Estratte {len(df)} righe")
            logger.info(f"   Metriche: {df['metric'].unique().tolist()}")
        return df
        
    def _plot_single_metric(self, df: pd.DataFrame, metric_name: str) -> plt.Figure:
        """
        Crea un grafico per una singola metrica.
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con le metriche
        metric_name : str
            Nome della metrica da plottare
        Returns
        -------
        plt.Figure
            Figura Matplotlib
        """
        # Filtra per la metrica specifica
        metric_data = df[df['metric'] == metric_name].sort_values('epoch')
        
        if metric_data.empty:
            logger.warning(f"⚠️ Nessun dato per metrica '{metric_name}'")
            return None
        
        # Crea la figura
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot della metrica
        epochs = metric_data['epoch'].values
        values = metric_data['value'].values
        
        ax.plot(epochs, values, marker='o', label=metric_name.upper(), 
                color='#2E86AB', linewidth=2.5, markersize=8)
        
        # Aggiungi linea del massimo
        max_value = values.max()
        max_epoch = epochs[values.argmax()]
        ax.axhline(y=max_value, color='#A23B72', linestyle='--', 
                   linewidth=1.5, alpha=0.7, 
                   label=f'Max: {max_value:.4f} (epoch {max_epoch})')
        
        # Configurazione assi
        ax.set_xlabel('Epoca', fontweight='bold')
        ax.set_ylabel(metric_name.upper(), fontweight='bold')
        ax.set_title(f'Andamento {metric_name.upper()} durante il Training', 
                     fontweight='bold', pad=20)
        
        # Griglia
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Legenda
        ax.legend(loc='best', framealpha=0.9)
        
        # Limiti asse Y (0-1 per metriche percentuali)
        ax.set_ylim([0, 1.05])
        
        # Aggiungi annotazione del punto massimo
        ax.annotate(f'{max_value:.4f}',
                   xy=(max_epoch, max_value),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        return fig
    
    def _plot_all_metrics_combined(self, df: pd.DataFrame) -> plt.Figure:
        """
        Crea un grafico unico con tutte le metriche.
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con le metriche
        
        Returns
        -------
        plt.Figure
            Figura Matplotlib
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Colori per ciascuna metrica
        colors = {
            'accuracy': '#2E86AB',
            'precision': '#A23B72',
            'recall': '#F18F01',
            'f1_micro': '#C73E1D',      
            'f1_macro': '#06A77D',      
            'f1_weighted': '#D4AC0D',
        }
        
        # Plot di ogni metrica
        for metric_name in self.metrics_to_plot:
            metric_data = df[df['metric'] == metric_name].sort_values('epoch')
            
            if metric_data.empty:
                continue
            
            epochs = metric_data['epoch'].values
            values = metric_data['value'].values
            color = colors.get(metric_name, '#000000')
            
            ax.plot(epochs, values, marker='o', label=metric_name.upper(),
                   color=color, linewidth=2.5, markersize=7, alpha=0.8)
        
        # Configurazione
        ax.set_xlabel('Epoca', fontweight='bold')
        ax.set_ylabel('Valore Metrica', fontweight='bold')
        ax.set_title('Confronto Metriche di Classificazione', 
                     fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(loc='best', framealpha=0.9)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        return fig
    
    @Step.requires()  
    @Step.provides(plots_generated=bool, output_dir=str)  
    def run(self, state: State) -> Optional[Dict[str, Any]]:
        """
        Esegue la generazione dei grafici.
        Flusso
        ------
        1. Carica le metriche dai file 
        2. Organizza i dati in DataFrame
        3. Genera grafici individuali per ogni metrica
        4. Genera grafico combinato
        5. Salva tutto in alta risoluzione
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 Generazione Grafici Metriche")
        logger.info(f"{'='*60}\n")
        
        # 1️⃣ Carica le metriche
        metrics_by_epoch = self._load_metrics_from_files()
        
        if not metrics_by_epoch:
            logger.error("❌ Nessuna metrica da plottare!")
            return
        
        # 2️⃣ Estrai i valori
        df = self._extract_metric_values(metrics_by_epoch)
        
        if df.empty:
            logger.error("❌ DataFrame metriche vuoto!")
            return
        
        # 3️⃣ Genera grafici individuali
        for metric_name in self.metrics_to_plot:
            fig = self._plot_single_metric(df, metric_name)
            
            if fig is not None:
                output_path = self.output_dir / f"{metric_name}_plot.png"
                fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"✅ Salvato: {output_path}")
                
                if self.show_plots:
                    plt.show()
                
                plt.close(fig)
        
        # 4️⃣ Genera grafico combinato
        fig_combined = self._plot_all_metrics_combined(df)
        output_path_combined = self.output_dir / "all_metrics_combined.png"
        fig_combined.savefig(output_path_combined, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"✅ Salvato grafico combinato: {output_path_combined}")
        
        if self.show_plots:
            plt.show()
        
        plt.close(fig_combined)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🎉 Generazione grafici completata!")
        logger.info(f"   Output directory: {self.output_dir}")
        logger.info(f"{'='*60}\n")
        return{
            "plots_generated": True,
            "output_dir": str(self.output_dir)
        }