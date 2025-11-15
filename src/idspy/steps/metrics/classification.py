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

import io
from PIL import Image

class ClassificationMetrics(Step):
    """Compute metrics for multiclass classification. --> 13/11 ho riscritto la classe"""

    def __init__(
        self,
        log_dir: Optional[str] = None,
        save_confusion_matrix: bool = True, #AGGIUNTO
        save_classification_report: bool = True,   #AGGIUNTO
        save_f1_per_class_plot: bool = True,
        log_prefix: str = "test",
        class_names: Optional[List[str]] = None, #--> aggiunto per salvare i nomi di ogni classe
        in_scope: str = "test",
        out_scope: str = "test",
        name: Optional[str] = None,
    ) -> None:
        self.writer: Optional[SummaryWriter] = (
            SummaryWriter(f"{log_dir}") if log_dir else None  #/{log_prefix} ← Aggiunto log_prefix
        )
        # self.log_prefix = log_prefix
        # self.class_names = class_names #aggiunto
        
        # aggiunto
        self.log_dir = Path(log_dir) if log_dir else Path("./metrics")
        self.log_prefix = log_prefix
        self.class_names = class_names or []
        self.save_confusion_matrix = save_confusion_matrix
        self.save_classification_report = save_classification_report
        self.save_f1_per_class_plot = save_f1_per_class_plot #aggiunto
        
        # Crea directory se necessario
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Sottodirectory per metriche per epoca
        self.metrics_per_epoch_dir = self.log_dir / "metrics_per_epoch"
        self.metrics_per_epoch_dir.mkdir(parents=True, exist_ok=True)
        
        # Confusion Matrix
        self.confusion_matrix_dir = self.metrics_per_epoch_dir / "confusion_matrix"
        self.confusion_matrix_dir.mkdir(parents=True, exist_ok=True)
        
        # Classification Report
        self.classification_report_dir = self.metrics_per_epoch_dir / "classification_report"
        self.classification_report_dir.mkdir(parents=True, exist_ok=True)
        
        # F1 per Classe
        self.f1_per_class_dir = self.metrics_per_epoch_dir / "f1_per_classe"
        self.f1_per_class_dir.mkdir(parents=True, exist_ok=True)
        ####
        
        super().__init__(
            name=name or "multiclass_classification_metrics",
            in_scope=in_scope,
            out_scope=out_scope,
        )
    
    #Aggiunto nuovo metodo per plottare f1Xcls:
    def _plot_f1_per_class(self, f1_per_class: list, epoch: int) -> None:
        """
        Crea e salva un grafico a barre con F1 per ogni classe.
        
        Parameters
        ----------
        f1_per_class : list
            Lista di F1 score per ogni classe
        epoch : int
            Numero dell'epoca
        """
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Etichette classi
        labels = (
            self.class_names 
            if self.class_names 
            else [f"Class {i}" for i in range(len(f1_per_class))]
        )
        
        # Colori: verde se F1 > 0.7, giallo se > 0.5, rosso altrimenti
        colors = []
        for f1_val in f1_per_class:
            if f1_val >= 0.8:
                colors.append("#1FB804")  # Verde
            elif f1_val >= 0.5:
                colors.append('#F18F01')  # Arancione
            else:
                colors.append("#FF2301F4")  # Rosso
        
        # Grafico a barre
        bars = ax.bar(range(len(labels)), f1_per_class, color=colors, 
                     alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Aggiungi valori sopra le barre
        for bar, f1_val in zip(bars, f1_per_class):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{f1_val:.4f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Linea di riferimento (F1 = 0.7 e 0.5)
        ax.axhline(y=0.8, color='green', linestyle='--', linewidth=1.5, 
                  alpha=0.5, label='Target threshold: 0.8')
        ax.axhline(y=0.5, color="#F18D01", linestyle='--', linewidth=1.5, 
                  alpha=0.5, label='Warning threshold: 0.5')
        
        # Media
        mean_f1 = np.mean(f1_per_class)
        ax.axhline(y=mean_f1, color='blue', linestyle='-', linewidth=2, 
                label=f'Media: {mean_f1:.3f}')
        
        # Configurazione
        ax.set_xlabel('Classi', fontweight='bold', fontsize=14)
        ax.set_ylabel('F1 Score', fontweight='bold', fontsize=14)
        ax.set_title(f'F1 Score per Classe - Epoch {epoch }', 
                    fontweight='bold', fontsize=16, pad=20)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.legend(loc='upper right', fontsize=11)
        
        plt.tight_layout()
        
        # Salva
        output_path = self.f1_per_class_dir / f"f1_per_classe_epoch_{epoch }.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        logging.info(f"💾 Grafico F1 per classe salvato: {output_path}")
        
        plt.close(fig)
    
    # Aggiunto NUOVO metodo per plottare cm nella classe:
    def _plot_confusion_matrix_to_image(self, cm: np.ndarray, normalize: bool = False) -> Image.Image:
        """
        Crea immagine confusion matrix per TensorBoard usando solo Matplotlib.
        
        Parameters
        ----------
        cm : np.ndarray
            Confusion matrix (shape: num_classes x num_classes)
        normalize : bool, default=False
            Se True, mostra percentuali invece di count
        
        Returns
        -------
        PIL.Image
            Immagine della confusion matrix
        """
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Etichette classi
        labels = self.class_names if self.class_names else [f"Class {i}" for i in range(len(cm))]
        
        # Normalizza se richiesto
        if normalize:
            cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            vmin, vmax = 0, 1
            fmt = '.2%'
        else:
            cm_display = cm
            vmin, vmax = None, None
            fmt = 'd'
        
        # Crea heatmap
        im = ax.imshow(cm_display, interpolation='nearest', cmap='Blues', vmin=vmin, vmax=vmax)
        
        # Colorbar
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        label_text = 'Percentage' if normalize else 'Count'
        cbar.ax.set_ylabel(label_text, rotation=-90, va="bottom", fontsize=12, fontweight='bold')
        
        # Configura assi
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        
        # Ruota etichette
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
        # Etichette assi
        ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold', labelpad=10)
        ax.set_ylabel('True Label', fontsize=13, fontweight='bold', labelpad=10)
        
        title = 'Confusion Matrix (Normalized)' if normalize else 'Confusion Matrix'
        ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
        
        # Griglia tra celle
        ax.set_xticks(np.arange(len(labels)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(labels)) - 0.5, minor=True)
        ax.grid(which="minor", color="lightgray", linestyle='-', linewidth=0.5)
        
        # Aggiungi numeri nelle celle
        thresh = cm_display.max() / 2.0
        for i in range(len(labels)):
            for j in range(len(labels)):
                text_color = "white" if cm_display[i, j] > thresh else "black"
                
                # Formatta testo
                if normalize:
                    text = f"{cm_display[i, j]:.1%}"
                else:
                    text = f"{cm_display[i, j]:d}"
                
                ax.text(j, i, text,
                    ha="center", va="center",
                    color=text_color, fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        # Converti in PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        image = Image.open(buf)
        
        plt.close(fig)
        
        return image
    
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

    @Step.requires(predictions=np.ndarray, targets=np.ndarray, epoch=int)  # aggiunto epoch
    @Step.provides(metrics=dict)
    def run(
        self, 
        state: State, 
        predictions: np.ndarray, 
        targets: np.ndarray,
        epoch: int = 0,  # aggiunto epoch
    ) -> None:
        import torchvision.transforms as transforms
        from sklearn.metrics import classification_report
        metrics = self.compute_metrics(predictions, targets)
    
        # 1️⃣ Salva metriche scalari 
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and self.writer is not None:
                self.writer.add_scalar(f"{self.log_prefix}/{name}", value, epoch)
        
        # 2️⃣ 🆕 Salva F1 per classe
        if self.writer is not None and 'f1_per_class' in metrics:
            f1_per_class = metrics['f1_per_class']
            
            for class_idx, f1_value in enumerate(f1_per_class):
                class_label = (
                    self.class_names[class_idx] 
                    if self.class_names and class_idx < len(self.class_names)
                    else f"class_{class_idx}"
                )
                
                self.writer.add_scalar(
                    f"{self.log_prefix}/f1_per_class/{class_label}", 
                    f1_value, 
                    epoch
                )
        
        # 3️⃣ 🆕 Salva Confusion Matrix come immagine
        if self.writer is not None and 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            # Salva versione con count
            cm_image_count = self._plot_confusion_matrix_to_image(cm, normalize=False)
            cm_tensor = transforms.ToTensor()(cm_image_count)
            self.writer.add_image(f"{self.log_prefix}/confusion_matrix_count", cm_tensor, epoch)
            
            # Salva versione normalizzata
            cm_image_norm = self._plot_confusion_matrix_to_image(cm, normalize=True)
            cm_tensor_norm = transforms.ToTensor()(cm_image_norm)
            self.writer.add_image(f"{self.log_prefix}/confusion_matrix_normalized", cm_tensor_norm, epoch)
        
        
        # 🆕 SALVA Confusion Matrix anche come PNG file
            if self.save_confusion_matrix:
                # Count version
                cm_image_count.save(
                    self.confusion_matrix_dir / f"confusion_matrix_count_epoch_{epoch}.png"
                )
                # Normalized version
                cm_image_norm.save(
                    self.confusion_matrix_dir / f"confusion_matrix_normalized_epoch_{epoch}.png"
                )
                logging.info(
                    f"💾 Confusion matrices salvate in: {self.confusion_matrix_dir} "
                    f"epoch_{epoch }"
                )
        
        # 4️⃣ 🆕 SALVA F1 per classe come PNG
        if self.save_f1_per_class_plot and 'f1_per_class' in metrics:
            f1_per_class = metrics['f1_per_class']
            self._plot_f1_per_class(f1_per_class, epoch)
        
        # 4️⃣ 🆕 SALVA Classification Report come file TXT
        if self.save_classification_report:
            report = classification_report(
                targets,
                predictions,
                target_names=self.class_names if self.class_names else None,
                zero_division=0,
                digits=4
            )
            
            report_path = (
                self.classification_report_dir / 
                f"classification_report_epoch_{epoch}.txt"
            )
            
            with open(report_path, 'w') as f:
                f.write(f"Classification Report - Epoch {epoch}\n")
                f.write(f"{'='*80}\n\n")
                f.write(report)
            
            logging.info(f"💾 Classification Report salvato: {report_path}")
        
        if self.writer is not None:
            self.writer.flush()  # Forza scrittura
            # NON chiudo qui: self.writer.close()
        
        return {"metrics": metrics}

# class ClassificationMetrics(Step):
#     """Compute metrics for multiclass classification."""

#     def __init__(
#         self,
#         log_dir: Optional[str] = None,
#         log_prefix: str = "test",
#         in_scope: str = "test",
#         out_scope: str = "test",
#         name: Optional[str] = None,
#     ) -> None:
#         self.writer: Optional[SummaryWriter] = (
#             SummaryWriter(log_dir) if log_dir else None
#         )
#         self.log_prefix = log_prefix

#         super().__init__(
#             name=name or "multiclass_classification_metrics",
#             in_scope=in_scope,
#             out_scope=out_scope,
#         )

#     def compute_metrics(self, y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
#         """Compute classification metrics."""
#         from sklearn.metrics import (
#             accuracy_score,
#             f1_score,
#             precision_score,
#             recall_score,
#             confusion_matrix,
#         )

#         f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
#         f1_macro = f1_score(y_true, y_pred, average="macro")
#         f1_micro = f1_score(y_true, y_pred, average="micro")
#         f1_weighted = f1_score(y_true, y_pred, average="weighted")

#         cm = confusion_matrix(y_true, y_pred)

#         metrics = {
#             "accuracy": accuracy_score(y_true, y_pred),
#             "precision": precision_score(
#                 y_true, y_pred, average="macro", zero_division=0
#             ),
#             "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
#             "f1_macro": f1_macro,
#             "f1_micro": f1_micro,
#             "f1_weighted": f1_weighted,
#             "f1_per_class": f1_per_class,
#             "confusion_matrix": cm,
#         }

#         return metrics

#     @Step.requires(
#         predictions=np.ndarray,
#         targets=np.ndarray,
#     )
#     @Step.provides(metrics=dict)
#     def run(self, state: State, predictions: np.ndarray, targets: np.ndarray) -> None:
#         metrics = self.compute_metrics(predictions, targets)

#         for name, value in metrics.items():
#             if isinstance(value, (int, float)) and self.writer is not None:
#                 self.writer.add_scalar(f"{self.log_prefix}/{name}", value)

#         if self.writer is not None:
#             self.writer.close()

#         return {"metrics": metrics}





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
        metrics_dir: Optional[str] = None, # prima questi due erano solo str 13/11
        output_dir: Optional[str] = None, 
        metrics_to_plot: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 300,
        show_plots: bool = False,
        style: str = "seaborn-v0_8-darkgrid",
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
        
        # 1️⃣ Prova prima TensorBoard         event_files = list(self.metrics_dir.glob("events.out.tfevents.*"))

        event_files = list(self.metrics_dir.rglob("events.out.tfevents.*"))
        
        if event_files:
            logger.info(f"🔍 Trovati {len(event_files)} file TensorBoard")
            
            try:
                from tensorboard.backend.event_processing import event_accumulator
                
                for event_file in sorted(event_files):
                    try:
                        logger.info(f"   📂 Caricamento: {event_file.relative_to(self.metrics_dir)}")

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
        else:
            logger.warning(f"⚠️ Nessun file TensorBoard trovato in: {self.metrics_dir}")
            logger.info("   Cerco nelle sottodirectory...")
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
        fig, ax = plt.subplots(figsize=(14,8))
        
        # Plot della metrica
        epochs = metric_data['epoch'].values
        values = metric_data['value'].values
        
        # Grafico a barre
        ax.bar(epochs, values, label=metric_name.upper(), 
            color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Aggiungi linea del massimo
        max_value = values.max()
        max_epoch = epochs[values.argmax()]
        ax.axhline(y=max_value, color='#A23B72', linestyle='--', 
                linewidth=1.5, alpha=0.7)
        
        # Configurazione assi
        ax.set_xlabel('Epoca', fontweight='bold', fontsize=13)
        ax.set_ylabel(metric_name.upper(), fontweight='bold', fontsize=13)
        
        # MODIFICA: Titolo più in basso e centrato - riduci pad
        ax.set_title(f'Andamento {metric_name.upper()} durante il Training', 
                    fontweight='bold', pad=25, fontsize=16)
        
        # Griglia
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
        
        # MODIFICA: Legenda a destra, più distante dal titolo
        legend_text = (
            f"Max: {max_value:.4f} (epoca {max_epoch})\n"
            f"Min: {values.min():.4f}\n"
            f"Media: {values.mean():.4f}"
        )

        # Posiziona la legenda a DESTRA in alto ma non sovrapposta
        ax.text(0.98, 1.01, legend_text,
                transform=ax.transAxes,
                fontsize=9,
                fontweight='bold',
                ha='right',
                va='bottom',  # MODIFICA: va='top' invece di 'bottom'
                bbox=dict(boxstyle='round,pad=0.6', 
                        facecolor='yellow',
                        edgecolor='black', 
                        linewidth=1.5,
                        alpha=0.9))
        
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
    # def _plot_single_metric(self, df: pd.DataFrame, metric_name: str) -> plt.Figure:
    #     """
    #     Crea un grafico per una singola metrica.
    #     Parameters
    #     ----------
    #     df : pd.DataFrame
    #         DataFrame con le metriche
    #     metric_name : str
    #         Nome della metrica da plottare
    #     Returns
    #     -------
    #     plt.Figure
    #         Figura Matplotlib
    #     """
    #     # Filtra per la metrica specifica
    #     metric_data = df[df['metric'] == metric_name].sort_values('epoch')
        
    #     if metric_data.empty:
    #         logger.warning(f"⚠️ Nessun dato per metrica '{metric_name}'")
    #         return None
        
    #     # Crea la figura
    #     fig, ax = plt.subplots(figsize=(14,8)) #figsize=self.figsize
        
    #     # Plot della metrica
    #     epochs = metric_data['epoch'].values
    #     values = metric_data['value'].values
        
    #     # ax.plot(epochs, values, marker='o', label=metric_name.upper(), 
    #     #         color='#2E86AB', linewidth=2.5, markersize=8)
    #     # 🆕 Grafico a barre invece di linea -------------------> 13/11 
    #     ax.bar(epochs, values, label=metric_name.upper(), 
    #     color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.2)
        
        
    #     # Aggiungi linea del massimo
    #     max_value = values.max()
    #     max_epoch = epochs[values.argmax()]
    #     ax.axhline(y=max_value, color='#A23B72', linestyle='--', 
    #                linewidth=1.5, alpha=0.7, 
    #                label=f'Max: {max_value:.4f} (epoch {max_epoch})')
        
    #     # Configurazione assi
    #     ax.set_xlabel('Epoca', fontweight='bold')
    #     ax.set_ylabel(metric_name.upper(), fontweight='bold')
    #     ax.set_title(f'Andamento {metric_name.upper()} durante il Training', 
    #                  fontweight='bold', pad=50, fontsize=14)
        
    #     # Griglia
    #     ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
        
    #     # Legenda
    #     #ax.legend(loc='best', framealpha=0.9)
    #     legend_text = (
    #         f"Max: {max_value:.4f} (epoca {max_epoch})\n"  # +1 per display umano
    #         f"Min: {values.min():.4f}\n"
    #         f"Media: {values.mean():.4f}"
    #     )
    
    #     # Posiziona la legenda SOPRA il grafico, centrata
    #     # 🆕 Posiziona la legenda a DESTRA in alto
    #     ax.text(0.97, 1.02, legend_text,
    #             transform=ax.transAxes,  # Usa coordinate relative (0-1)
    #             fontsize=9,
    #             fontweight='bold',
    #             ha='right',      # ← Allineamento a destra
    #             va='bottom',        # ← Allineamento dall'alto
    #             bbox=dict(boxstyle='round,pad=0.6', 
    #                     facecolor='yellow',  # Giallo chiaro
    #                     edgecolor='black', 
    #                     linewidth=1.5,
    #                     alpha=0.9))
        
    #     # Limiti asse Y (0-1 per metriche percentuali)
    #     ax.set_ylim([0, 1.05])
        
    #     # Aggiungi annotazione del punto massimo
    #     ax.annotate(f'{max_value:.4f}',
    #                xy=(max_epoch, max_value),
    #                xytext=(10, 10), textcoords='offset points',
    #                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
    #                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
    #     plt.tight_layout(rect=[0.12, 0, 0.88, 0.88])
    #     return fig
    
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
        fig, ax = plt.subplots(figsize=(16, 9))
        
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
            
            # ax.plot(epochs, values, marker='o', label=metric_name.upper(),
            #        color=color, linewidth=2.5, markersize=7, alpha=0.8)
            # 🆕 Grafico a barre raggruppate -----------------------------> 13/11
            bar_width = 0.13
            num_metrics = len(self.metrics_to_plot)
            offset = (list(self.metrics_to_plot).index(metric_name) - num_metrics/2) * bar_width
        
            ax.bar(epochs + offset, values, width=bar_width, 
                label=metric_name.upper(), color=color, alpha=0.85, 
                edgecolor='black', linewidth=0.8)
            # offset = (list(self.metrics_to_plot).index(metric_name) - len(self.metrics_to_plot)/2) * bar_width
            # ax.bar(epochs + offset, values, width=bar_width, 
            #     label=metric_name.upper(), color=color, alpha=0.8, 
            #     edgecolor='black', linewidth=0.8)
        
        # Configurazione
        ax.set_xlabel('Epoca', fontweight='bold', fontsize=13)
        ax.set_ylabel('Valore Metrica', fontweight='bold', fontsize=13)
        ax.set_title('Confronto Metriche di Classificazione', 
                     fontweight='bold', pad=80, fontsize=16)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
        ax.legend(loc='best', framealpha=0.9)
        ax.set_ylim([0, 1.05])
        
        #LEGENDA ORIZZONTALE SOTTO IL TITOLO
        ax.legend(
            loc='upper center',        # Posiziona in alto al centro
            bbox_to_anchor=(0.5, 1.12), # Sposta sopra il grafico (y > 1)
            ncol=6,                     # 6 colonne (una per metrica)
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=11,
            framealpha=0.95,
            edgecolor='black',
            facecolor='white'
        )
        
        plt.tight_layout()
        return fig

    def _plot_metrics_separate_bars(self, df: pd.DataFrame) -> plt.Figure:
        """
        Crea un grafico con BARRE SEPARATE per ogni metrica.
        Una figura con sottografici (subplot) - uno per metrica.
        Ideale come figura supplementare in appendice.
        """
        num_metrics = len(self.metrics_to_plot)
        
        # Calcola numero di righe e colonne per i subplot
        ncols = 2
        nrows = (num_metrics + ncols - 1) // ncols  # Arrotonda per eccesso
        
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                                 figsize=(16, 5 * nrows))
        
        # Appiattisci axes se è un array 2D
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]
        
        colors = {
            'accuracy': '#2E86AB',
            'precision': '#A23B72',
            'recall': '#F18F01',
            'f1_micro': '#C73E1D',      
            'f1_macro': '#06A77D',      
            'f1_weighted': '#D4AC0D',
        }
        
        # Crea un subplot per ogni metrica
        for idx, metric_name in enumerate(self.metrics_to_plot):
            ax = axes[idx]
            metric_data = df[df['metric'] == metric_name].sort_values('epoch')
            
            if metric_data.empty:
                ax.text(0.5, 0.5, f'Nessun dato per {metric_name}',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{metric_name.upper()} - ❌ Dati non disponibili')
                continue
            
            epochs = metric_data['epoch'].values
            values = metric_data['value'].values
            color = colors.get(metric_name, '#000000')
            
            # Barre separate (non raggruppate)
            bars = ax.bar(epochs, values, color=color, alpha=0.8, 
                         edgecolor='black', linewidth=1.2, width=0.6)
            
            # Linea del massimo
            max_value = values.max()
            
            # Aggiungi valori sopra le barre
            for bar, value, epoch in zip(bars, values, epochs):
                height = bar.get_height()
                # Se barra è molto alta (> 0.95), metti testo sotto
                if height > 0.90:
                    y_pos = height - 0.05  # Dentro la barra
                    color_text = 'white'
                    va_align = 'top'
                else:
                    y_pos = max_value + 0.02  # Sopra la barra
                    color_text = 'black'
                    va_align = 'bottom'
                ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                   f'{value:.3f}',
                   ha='center', va=va_align, 
                   fontsize=9, fontweight='bold',
                   color=color_text)
                # ax.text(bar.get_x() + bar.get_width()/2., height,
                #        f'{value:.3f}',
                #        ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Linea del massimo
            ax.axhline(y=max_value, color='red', linestyle='--', 
                      linewidth=1.5, alpha=0.6)
            
            # Info in alto a destra, FUORI dal plot
            info_text = f"Max: {max_value:.4f}"
            ax.text(0.98, 1.03, info_text,
                    transform=ax.transAxes,
                    fontsize=10, fontweight='bold',
                    ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', 
                            facecolor='yellow', 
                            alpha=0.8,
                            edgecolor='black'))
            
            # Configurazione subplot
            ax.set_xlabel('Epoca', fontweight='bold', fontsize=11)
            ax.set_ylabel('Valore', fontweight='bold', fontsize=11)
            
            ax.set_title(f'{metric_name.upper()}', fontweight='bold', fontsize=12, pad=8)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax.set_ylim([0, 1.05])
            # ax.legend(loc='upper right', fontsize=9)
        
        # Nascondi i subplot vuoti
        for idx in range(len(self.metrics_to_plot), len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle('Metriche - Analisi Dettagliata per Epoca', 
            fontsize=17, fontweight='bold', y=0.955, x=0.5, ha='center')
        plt.tight_layout(rect=[0.10, 0, 0.90, 0.95])
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
        
        # 5️⃣ AGGIUNTO 13/11 --> Genera grafico con barre separate (figura supplementare)
        fig_separate = self._plot_metrics_separate_bars(df)
        output_path_separate = self.output_dir / "all_metrics_separated_bars.png"
        fig_separate.savefig(output_path_separate, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"✅ Salvato grafico separate bars (appendice): {output_path_separate}")
        
        if self.show_plots:
            plt.show()
        
        plt.close(fig_separate)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🎉 Generazione grafici completata!")
        logger.info(f"   Output directory: {self.output_dir}")
        logger.info(f"{'='*60}\n")
        return{
            "plots_generated": True,
            "output_dir": str(self.output_dir)
        }