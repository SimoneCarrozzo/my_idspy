from typing import Optional, Dict, List, Tuple, Any

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.idspy.core.step import Step
from src.idspy.core.state import State

import logging
import json
from pathlib import Path
import pandas as pd

import io
from PIL import Image

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class FederatedRoundMetrics_alt(Step):
    """
    Salva metriche per training federato.
    Si attiva SOLO nei round con valutazione globale.
    """
    
    def __init__(
        self,
        log_dir: str,
        class_names: Optional[List[str]] = None,
        save_confusion_matrix: bool = True,
        save_classification_report: bool = True,
        save_f1_per_class: bool = True,
        save_client_distribution: bool = True,
        in_scope: str = "federated",
        out_scope: str = "federated",
        name: Optional[str] = None,
    ):
        self.log_dir = Path(log_dir)
        self.class_names = class_names or []
        self.save_confusion_matrix = save_confusion_matrix
        self.save_classification_report = save_classification_report
        self.save_f1_per_class = save_f1_per_class
        self.save_client_distribution = save_client_distribution
        
        # Directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.confusion_matrix_dir = self.log_dir / "confusion_matrix"
        self.classification_report_dir = self.log_dir / "classification_report"
        self.f1_per_class_dir = self.log_dir / "f1_per_classe"
        self.client_metrics_dir = self.log_dir / "client_distribution"
        
        for d in [self.confusion_matrix_dir, self.classification_report_dir,
                  self.f1_per_class_dir, self.client_metrics_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        super().__init__(
            name=name or "federated_round_metrics",
            in_scope=in_scope,
            out_scope=out_scope,
        )
    
    @Step.requires(
        round_num=int,
        # predictions=np.ndarray,
        # targets=np.ndarray,
        predictions=(np.ndarray, type(None)),  # ← Ora accetta None
        targets=(np.ndarray, type(None)),      # ← Ora accetta None
        round_results=dict,
    )
    @Step.provides(metrics_saved=bool)
    def run(
        self,
        state: State,
        round_num: int,
        predictions: np.ndarray,
        targets: np.ndarray,
        round_results: dict,
    ) -> Dict[str, Any]:
        
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            confusion_matrix,
        )
        
        # 🆕EVENTUALE FALLBACK: Recupera class_names dallo State se non forniti all'init
        if not self.class_names:
            try:
                original_label_map = state.get("data.original_label_map", dict)
                if original_label_map and isinstance(original_label_map, dict):
                    # Crea lista ordinata per indice
                    max_idx = max(original_label_map.values())
                    self.class_names = [None] * (max_idx + 1)
                    for name, idx in original_label_map.items():
                        self.class_names[idx] = name
                    logger.info(f"✅ Class names recuperati: {len(self.class_names)} classi")
            except Exception as e:
                logger.warning(f"⚠️ Impossibile recuperare class_names: {e}")
                self.class_names = []
        
        logger.info(f"\n📊 Salvataggio metriche Round {round_num}...")
        
        
        # ✅ CHECK: Se non abbiamo predictions, salva solo metriche aggregate
        has_predictions = predictions is not None and targets is not None

        if not has_predictions:
            logger.info("   ℹ️ Valutazione cluster: salvo solo metriche aggregate")
            self._save_aggregated_metrics_only(round_results, round_num)
            return {"metrics_saved": True}
        
        
        
        # 1️⃣ Confusion Matrix
        if self.save_confusion_matrix:
            cm = confusion_matrix(targets, predictions)
            self._save_confusion_matrix(cm, round_num)
        
        # 2️⃣ Classification Report
        if self.save_classification_report:
            self._save_classification_report(targets, predictions, round_num)
        
        # 3️⃣ F1 per classe (grafico)
        if self.save_f1_per_class:
            f1_per_class = f1_score(targets, predictions, average=None, zero_division=0)
            self._save_f1_per_class(f1_per_class, round_num)
        
        # 4️⃣ Distribuzione loss client
        if self.save_client_distribution:
            self._save_client_distribution(round_results, round_num)
        
        logger.info(f"✅ Metriche Round {round_num} salvate in: {self.log_dir}")
        
        return {"metrics_saved": True}
    
    def _save_confusion_matrix(self, cm: np.ndarray, round_num: int):
        """Salva confusion matrix come immagine."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        labels = self.class_names if self.class_names else [f"C{i}" for i in range(len(cm))]
        
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        
        ax.set_xlabel('Predicted', fontweight='bold', fontsize=12)
        ax.set_ylabel('True', fontweight='bold', fontsize=12)
        ax.set_title(f'Confusion Matrix - Round {round_num}', fontweight='bold', fontsize=14)
        
        # Aggiungi valori nelle celle
        thresh = cm.max() / 2.0
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f'{cm[i, j]}',
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=8)
        
        plt.tight_layout()
        fig.savefig(
            self.confusion_matrix_dir / f"confusion_matrix_round_{round_num}.png",
            dpi=300, bbox_inches='tight'
        )
        plt.close(fig)
    
    def _save_classification_report(self, targets: np.ndarray, predictions: np.ndarray, round_num: int):
        from sklearn.metrics import classification_report

        """Salva classification report come file di testo."""
        report = classification_report(
            targets,
            predictions,
            target_names=self.class_names if self.class_names else None,
            zero_division=0,
            digits=4
        )
        
        report_path = self.classification_report_dir / f"classification_report_round_{round_num}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Classification Report - Round {round_num}\n")
            f.write(f"{'='*80}\n\n")
            f.write(report)
        
        logger.info(f"💾 Classification Report salvato: {report_path.name}")
    
    def _save_f1_per_class(self, f1_per_class: np.ndarray, round_num: int):
        """Salva grafico F1 per classe."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        labels = self.class_names if self.class_names else [f"C{i}" for i in range(len(f1_per_class))]
        
        colors = ['#1FB804' if f1 >= 0.8 else '#F18F01' if f1 >= 0.5 else '#FF2301' 
                  for f1 in f1_per_class]
        
        bars = ax.bar(range(len(labels)), f1_per_class, color=colors, 
                     alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Valori sopra barre
        for bar, f1_val in zip(bars, f1_per_class):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{f1_val:.4f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Threshold lines
        ax.axhline(y=0.8, color='green', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1.5, alpha=0.5)
        
        # Media
        mean_f1 = np.mean(f1_per_class)
        ax.axhline(y=mean_f1, color='blue', linestyle='-', linewidth=2, 
                  label=f'Media: {mean_f1:.3f}')
        
        ax.set_xlabel('Classi', fontweight='bold', fontsize=13)
        ax.set_ylabel('F1 Score', fontweight='bold', fontsize=13)
        ax.set_title(f'F1 Score per Classe - Round {round_num}', fontweight='bold', fontsize=15)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        plt.tight_layout()
        fig.savefig(
            self.f1_per_class_dir / f"f1_per_classe_round_{round_num}.png",
            dpi=300, bbox_inches='tight'
        )
        plt.close(fig)
        
        
    def _save_aggregated_metrics_only(self, round_results: dict, round_num: int):
        """Salva solo metriche aggregate (senza confusion matrix)."""
        
        # Cerca metriche aggregate nei risultati
        agg_metrics = round_results.get('aggregated_test_metrics', {})
        cluster_metrics = round_results.get('cluster_test_metrics', {})
        
        if not agg_metrics:
            logger.warning("⚠️ Nessuna metrica aggregata trovata")
            return
        
        # Salva report testuale
        report_path = self.classification_report_dir / f"cluster_metrics_round_{round_num}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Cluster Evaluation Report - Round {round_num}\n")
            f.write(f"{'='*80}\n\n")
            
            # Metriche aggregate
            f.write("📊 AGGREGATE METRICS (Weighted Average)\n")
            f.write(f"{'-'*80}\n")
            f.write(f"Accuracy:  {agg_metrics.get('accuracy', 0):.4f}\n")
            f.write(f"Precision: {agg_metrics.get('precision', 0):.4f}\n")
            f.write(f"Recall:    {agg_metrics.get('recall', 0):.4f}\n")
            f.write(f"F1-Score:  {agg_metrics.get('f1', 0):.4f}\n")
            f.write(f"Total Samples: {agg_metrics.get('total_samples', 0)}\n")
            f.write(f"Clusters Evaluated: {agg_metrics.get('num_clusters_evaluated', 0)}\n\n")
            
            # Metriche per cluster
            f.write("🎯 PER-CLUSTER METRICS\n")
            f.write(f"{'-'*80}\n")
            for cluster_id, metrics in cluster_metrics.items():
                f.write(f"\nCluster {cluster_id} ({metrics.get('target_column', 'unknown')})\n")
                f.write(f"  Samples:   {metrics.get('num_samples', 0)}\n")
                f.write(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}\n")
                f.write(f"  Precision: {metrics.get('precision', 0):.4f}\n")
                f.write(f"  Recall:    {metrics.get('recall', 0):.4f}\n")
                f.write(f"  F1-Score:  {metrics.get('f1', 0):.4f}\n")
        
        logger.info(f"💾 Cluster metrics salvate: {report_path.name}")
        
        # Salva distribuzione client (sempre disponibile)
        if 'client_updates' in round_results:
            self._save_client_distribution(round_results, round_num)

    
    
    def _save_client_distribution(self, round_results: dict, round_num: int):
        """Salva distribuzione loss tra client (specifica del federated)."""
        client_updates = round_results['client_updates']
        client_ids = [u['client_id'] for u in client_updates]
        
        # client_losses = [u['metrics']['final_loss'] for u in client_updates]
        # --- Controllo di sicurezza ---
        if not client_updates:
            logger.warning("⚠️ Nessun dato per la distribuzione loss client.")
            return
        client_ids = [str(u['client_id']) for u in client_updates]
        client_losses = [u['metrics']['final_loss'] for u in client_updates]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(range(len(client_ids)), client_losses, color='steelblue', 
                     alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Statistiche
        mean_loss = np.mean(client_losses)
        std_loss = np.std(client_losses)
        
        ax.axhline(y=mean_loss, color='red', linestyle='--', linewidth=2,
                  label=f'Media: {mean_loss:.4f}')
        ax.fill_between(range(len(client_ids)),
                        mean_loss - std_loss, mean_loss + std_loss,
                        alpha=0.2, color='red', label=f'± 1 STD: {std_loss:.4f}')
        
        # Valori sopra barre
        for bar, loss in zip(bars, client_losses):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{loss:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Client ID', fontweight='bold', fontsize=12)
        ax.set_ylabel('Final Loss', fontweight='bold', fontsize=12)
        ax.set_title(f'Distribuzione Loss tra Client - Round {round_num}', 
                    fontweight='bold', fontsize=14)
        ax.set_xticks(range(len(client_ids)))
        ax.set_xticklabels(client_ids, rotation=45)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        fig.savefig(
            self.client_metrics_dir / f"client_loss_distribution_round_{round_num}.png",
            dpi=300, bbox_inches='tight'
        )
        plt.close(fig)        
        
########################

class FederatedPlotMetrics_alt(Step):
    """
    Genera grafici finali aggregando tutti i round.
    Legge i dati da round_history.
    """
    
    def __init__(
        self,
        output_dir: str,
        rounds_history: List[Dict],
        figsize: tuple = (14, 8),
        dpi: int = 300,
        in_scope: str = "federated",
        out_scope: str = "federated",
        name: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.rounds_history = rounds_history
        self.figsize = figsize
        self.dpi = dpi
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        super().__init__(
            name=name or "federated_plot_metrics",
            in_scope=in_scope,
            out_scope=out_scope,
        )
    
    @Step.requires()
    @Step.provides(plots_generated=bool)
    def run(self, state: State) -> Dict[str, Any]:
        
        logger.info(f"\n{'='*60}")
        logger.info("📊 Generazione Grafici Federati")
        logger.info(f"{'='*60}\n")
        
        # Estrai dati
        data = self._extract_data_from_history()
        
        if not data['eval_rounds']:
            logger.warning("⚠️ Nessun round con valutazione trovato!")
            return {"plots_generated": False}
        
        # Genera grafici
        self._plot_global_metrics(data)
        self._plot_loss_statistics(data)
        self._plot_client_participation(data)
        self._plot_combined_overview(data)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ Grafici salvati in: {self.output_dir}")
        logger.info(f"{'='*60}\n")
        
        return {"plots_generated": True}
    
    def _extract_data_from_history(self) -> Dict:
        """Estrae dati strutturati da round_history."""
        
        data = {
            'all_rounds': [],
            'eval_rounds': [],
            'accuracies': [],
            'f1_scores': [],
            'avg_losses': [],
            'std_losses': [],
            'num_clients': [],
        }
        
        for r in self.rounds_history:
            round_num = r['round_num']
            data['all_rounds'].append(round_num)
            
            # Metriche sempre disponibili
            data['avg_losses'].append(r['round_metrics']['weighted_avg_loss'])
            data['std_losses'].append(r['round_metrics']['std_loss'])
            data['num_clients'].append(r['num_clients_participated'])
            
            # Metriche di valutazione (solo alcuni round)
            if 'global_test_metrics' in r:
                data['eval_rounds'].append(round_num)
                data['accuracies'].append(r['global_test_metrics']['accuracy'])
                data['f1_scores'].append(r['global_test_metrics']['f1'])
        
        return data
    
    def _plot_global_metrics(self, data: Dict):
        """Plot 1: Accuracy e F1 sui round valutati."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        rounds = data['eval_rounds']
        
        # Accuracy
        ax1.plot(rounds, data['accuracies'], marker='o', linewidth=2.5, 
                markersize=8, color='#2E86AB', label='Accuracy')
        ax1.axhline(np.mean(data['accuracies']), color='red', linestyle='--', 
                   label=f'Media: {np.mean(data["accuracies"]):.4f}')
        ax1.set_xlabel('Round', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
        ax1.set_title('Accuracy su Test Globale', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim([0, 1.05])
        
        # F1
        ax2.plot(rounds, data['f1_scores'], marker='s', linewidth=2.5, 
                markersize=8, color='#A23B72', label='F1 Score')
        ax2.axhline(np.mean(data['f1_scores']), color='red', linestyle='--',
                   label=f'Media: {np.mean(data["f1_scores"]):.4f}')
        ax2.set_xlabel('Round', fontweight='bold', fontsize=12)
        ax2.set_ylabel('F1 Score', fontweight='bold', fontsize=12)
        ax2.set_title('F1 Score su Test Globale', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim([0, 1.05])
        
        plt.tight_layout()
        fig.savefig(self.output_dir / "global_metrics.png", dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        logger.info("✅ Salvato: global_metrics.png")
    
    def _plot_loss_statistics(self, data: Dict):
        """Plot 2: Loss media e STD (mostra convergenza Non-IID)."""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        rounds = data['all_rounds']
        
        # Loss Media
        ax1.plot(rounds, data['avg_losses'], marker='o', linewidth=2, 
                color='#F18F01', label='Loss Media Client')
        ax1.fill_between(rounds, 
                         np.array(data['avg_losses']) - np.array(data['std_losses']),
                         np.array(data['avg_losses']) + np.array(data['std_losses']),
                         alpha=0.2, color='#F18F01')
        ax1.set_xlabel('Round', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Loss', fontweight='bold', fontsize=12)
        ax1.set_title('Loss Media Client (± 1 STD)', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Loss STD (indicatore Non-IID)
        ax2.plot(rounds, data['std_losses'], marker='s', linewidth=2, 
                color='#C73E1D', label='Deviazione Standard Loss')
        ax2.axhline(np.mean(data['std_losses']), color='blue', linestyle='--',
                   label=f'Media STD: {np.mean(data["std_losses"]):.4f}')
        ax2.set_xlabel('Round', fontweight='bold', fontsize=12)
        ax2.set_ylabel('STD Loss', fontweight='bold', fontsize=12)
        ax2.set_title('Eterogeneità tra Client (Non-IID Indicator)', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        fig.savefig(self.output_dir / "loss_statistics.png", dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        logger.info("✅ Salvato: loss_statistics.png")
    
    def _plot_client_participation(self, data: Dict):
        """Plot 3: Numero client partecipanti per round."""
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        rounds = data['all_rounds']
        
        ax.bar(rounds, data['num_clients'], color='#06A77D', alpha=0.8, 
              edgecolor='black', linewidth=1.2)
        ax.axhline(np.mean(data['num_clients']), color='red', linestyle='--',
                  linewidth=2, label=f'Media: {np.mean(data["num_clients"]):.1f}')
        
        ax.set_xlabel('Round', fontweight='bold', fontsize=12)
        ax.set_ylabel('N° Client', fontweight='bold', fontsize=12)
        ax.set_title('Client Partecipanti per Round', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        plt.tight_layout()
        fig.savefig(self.output_dir / "client_participation.png", dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        logger.info("✅ Salvato: client_participation.png")
    
    def _plot_combined_overview(self, data: Dict):
        """Plot 4: Overview completo (4 subplot)."""
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        
        rounds_all = data['all_rounds']
        rounds_eval = data['eval_rounds']
        
        # 1. Accuracy
        ax1.plot(rounds_eval, data['accuracies'], marker='o', linewidth=2, color='#2E86AB')
        ax1.set_title('Accuracy Globale', fontweight='bold')
        ax1.set_ylabel('Accuracy', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])
        
        # 2. F1
        ax2.plot(rounds_eval, data['f1_scores'], marker='s', linewidth=2, color='#A23B72')
        ax2.set_title('F1 Score Globale', fontweight='bold')
        ax2.set_ylabel('F1 Score', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.05])
        
        # 3. Loss STD
        ax3.plot(rounds_all, data['std_losses'], marker='o', linewidth=2, color='#C73E1D')
        ax3.set_title('Eterogeneità Client (Loss STD)', fontweight='bold')
        ax3.set_xlabel('Round', fontweight='bold')
        ax3.set_ylabel('STD Loss', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Client Participation
        ax4.bar(rounds_all, data['num_clients'], color='#06A77D', alpha=0.8)
        ax4.set_title('Client Partecipanti', fontweight='bold')
        ax4.set_xlabel('Round', fontweight='bold')
        ax4.set_ylabel('N° Client', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle('Federated Learning - Overview Completo', fontsize=18, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        fig.savefig(self.output_dir / "overview_combined.png", dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        logger.info("✅ Salvato: overview_combined.png")
        

#=============================================================================#
#=============================================================================#
#=============================================================================#
#=============================================================================#
class ClusterRoundMetrics_alt(Step):
    """
    Salva metriche per ogni cluster separatamente.
    Itera su cluster_test_metrics e genera confusion matrix + report per ognuno.
    """
    
    def __init__(
        self,
        log_dir: str,
        class_names: Optional[List[str]] = None,
        save_confusion_matrix: bool = True,
        save_classification_report: bool = True,
        save_f1_per_class: bool = True,
        in_scope: str = "federated",
        out_scope: str = "federated",
        name: Optional[str] = None,
    ):
        self.log_dir = Path(log_dir)
        self.class_names = class_names or []
        self.save_confusion_matrix = save_confusion_matrix
        self.save_classification_report = save_classification_report
        self.save_f1_per_class = save_f1_per_class
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        from collections import defaultdict
        # Storico metriche per cluster
        self.cluster_history = defaultdict(lambda: {
            'rounds': [], 'accuracy': [], 'f1': [],
            'precision': [], 'recall': []
        })

        # 🆕 Storico F1 per singola classe (ogni cluster ha 1-2 target)
        self.f1_class_history = defaultdict(lambda: defaultdict(list))
        
        super().__init__(
            name=name or "cluster_round_metrics",
            in_scope=in_scope,
            out_scope=out_scope,
        )
    
    @Step.requires(
        round_num=int,
        cluster_results_detailed=dict,  # Contiene predictions/targets per cluster
    )
    @Step.provides(metrics_saved=bool)
    # def run(self, state: State, round_num: int, cluster_results: dict) -> Dict[str, Any]:
    #     from sklearn.metrics import confusion_matrix, f1_score, classification_report
    #     from collections import defaultdict
        
    #     logger.info(f"\n📊 Salvataggio metriche cluster - Round {round_num}")
        
    #     # --- FIX MEMORIA: Recupera storico dallo STATE, non da self ---
    #     # Usiamo un helper per ricostruire la struttura defaultdict se non esiste
    #     saved_history = state.get("cluster_metrics_history", {})
    #     # Convertiamo in defaultdict per facilità d'uso
    #     self.cluster_history = defaultdict(lambda: {
    #         'rounds': [], 'accuracy': [], 'f1': [], 'precision': [], 'recall': []
    #     }, saved_history)
        
    #     # Recupera storico F1 per classe (o crea nuovo)
    #     self.f1_class_history = state.get("cluster_f1_class_history", defaultdict(lambda: defaultdict(list)))
    #     # ---
        
    #     # Recupera class_names se non forniti
    #     if not self.class_names:
    #         original_label_map = state.get("data.original_label_map", dict)
    #         if original_label_map:
    #             max_idx = max(original_label_map.values())
    #             self.class_names = [None] * (max_idx + 1)
    #             for name, idx in original_label_map.items():
    #                 self.class_names[idx] = name
        
    #     # Itera su ogni cluster
    #     # Itera su ogni cluster
    #     for cluster_id, result in cluster_results.items():
    #         # Crea struttura per questo cluster
    #         cluster_dir = self.log_dir / f"cluster_{cluster_id}"
    #         cm_dir = cluster_dir / "confusion_matrix"
    #         report_dir = cluster_dir / "classification_report"
    #         f1_dir = cluster_dir / "f1_per_classe"
    #         evolution_dir = cluster_dir / "evolution"

    #         for d in [cm_dir, report_dir, f1_dir, evolution_dir]:
    #             d.mkdir(parents=True, exist_ok=True)
                    
    #         predictions = result.get('predictions')
    #         targets = result.get('targets')
            
    #         if predictions is None or targets is None:
    #             continue
            
    #         # 🎯 Recupera nomi attacchi target per questo cluster
    #         target_to_cluster = state.get('federated.stable_target_to_cluster', dict)
    #         cluster_targets = [
    #             name.replace('is_', '')  # Rimuovi prefisso
    #             for name, cid in target_to_cluster.items() 
    #             if cid == cluster_id
    #         ]
            
    #         if not cluster_targets:
    #             logger.warning(f"⚠️ Cluster {cluster_id}: nessun target trovato!")
    #             cluster_targets = [f"Unknown_{cluster_id}"]
            
    #         # 🔢 Aggiorna storico con metriche dal result
    #         self.cluster_history[cluster_id]['rounds'].append(round_num)
    #         self.cluster_history[cluster_id]['accuracy'].append(result.get('accuracy', 0.0))
    #         self.cluster_history[cluster_id]['f1'].append(result.get('f1', 0.0))
    #         self.cluster_history[cluster_id]['precision'].append(result.get('precision', 0.0))
    #         self.cluster_history[cluster_id]['recall'].append(result.get('recall', 0.0))
            
    #         # 1. Confusion Matrix
    #         if self.save_confusion_matrix:
    #             cm = confusion_matrix(targets, predictions)
    #             self._save_confusion_matrix(cm, round_num, cluster_id, cluster_targets, cm_dir)
            
    #         # 2. Classification Report
    #         if self.save_classification_report:
    #             self._save_classification_report(targets, predictions, round_num, 
    #                                             cluster_id, cluster_targets, report_dir)
            
    #         # 3. F1 Evolution (Line Plot)
    #         if self.save_f1_per_class:
    #             self._save_f1_evolution(round_num, cluster_id, cluster_targets, f1_dir)
            
    #         # 4. Metrics Evolution (2x2 Subplot)
    #         self._save_metrics_evolution(round_num, cluster_id, cluster_targets, evolution_dir)
        
        
    #     logger.info(f"✅ Metriche cluster salvate in: {self.log_dir}")
    #     return {"metrics_saved": True}
    def run(self, state: State, round_num: int, cluster_results_detailed: dict) -> Dict[str, Any]:
        from sklearn.metrics import confusion_matrix
        from collections import defaultdict
        import numpy as np
        
        logger.info(f"\n📊 [Round {round_num}] Elaborazione grafici e storico metriche...")

        # ═══════════════════════════════════════════════════════════════════
        # 1. GESTIONE PERSISTENZA STATO (Evita KeyError al Round 0)
        # ═══════════════════════════════════════════════════════════════════
        
        from collections import defaultdict

        # --- Gestione cluster_metrics_history ---
        if state.has("federated.cluster_metrics_history"):
            # Recupera storico esistente
            saved_history = state.get("federated.cluster_metrics_history", dict)
            
            # Ricostruisci defaultdict da dict normale
            self.cluster_history = defaultdict(
                lambda: {'rounds': [], 'accuracy': [], 'f1': [], 'precision': [], 'recall': []},
                saved_history  # Direttamente il dict salvato
            )
        else:
            # Round 0: inizializza vuoto
            self.cluster_history = defaultdict(lambda: {
                'rounds': [], 'accuracy': [], 'f1': [], 'precision': [], 'recall': []
            })

        # --- Gestione cluster_f1_class_history ---
        if state.has("federated.cluster_f1_class_history"):
            saved_f1_history = state.get("federated.cluster_f1_class_history", dict)
            
            # Ricostruisci nested defaultdict
            self.f1_class_history = defaultdict(
                lambda: defaultdict(list),
                {k: defaultdict(list, v) for k, v in saved_f1_history.items()}
            )
        else:
            # Round 0: inizializza vuoto
            self.f1_class_history = defaultdict(lambda: defaultdict(list))


        # ═══════════════════════════════════════════════════════════════════
        # 2. ITERAZIONE SUI RISULTATI DI OGNI CLUSTER
        # ═══════════════════════════════════════════════════════════════════
        
        for cluster_id, result in cluster_results_detailed.items():
            # Prepariamo le cartelle di output
            cluster_dir = self.log_dir / f"cluster_{cluster_id}"
            dirs = {
                "cm": cluster_dir / "confusion_matrix",
                "f1": cluster_dir / "f1_per_classe",
                "evol": cluster_dir / "evolution"
            }
            for d in dirs.values():
                d.mkdir(parents=True, exist_ok=True)

            # Estrazione dati (forniti da EvaluateClusterModels)
            predictions = result.get('predictions')
            targets = result.get('targets')
            target_col_name = result.get('target_column', f"Cluster_{cluster_id}")
            clean_target_name = target_col_name.replace('is_', '')

            # --- AGGIORNAMENTO MEMORIA ---
            # Aggiungiamo i dati solo se non abbiamo già registrato questo round 
            # (utile in caso di ri-esecuzione dello stesso step)
            if round_num not in self.cluster_history[cluster_id]['rounds']:
                self.cluster_history[cluster_id]['rounds'].append(round_num)
                self.cluster_history[cluster_id]['accuracy'].append(result.get('accuracy', 0.0))
                self.cluster_history[cluster_id]['f1'].append(result.get('f1', 0.0))
                self.cluster_history[cluster_id]['precision'].append(result.get('precision', 0.0))
                self.cluster_history[cluster_id]['recall'].append(result.get('recall', 0.0))

            # ═══════════════════════════════════════════════════════════════════
            # 3. GENERAZIONE GRAFICI
            # ═══════════════════════════════════════════════════════════════════
            
            # A. Confusion Matrix
            if self.save_confusion_matrix and predictions is not None:
                cm = confusion_matrix(targets, predictions)
                self._save_confusion_matrix(cm, round_num, cluster_id, [clean_target_name], dirs["cm"])
            
            # B. F1 Evolution (Linea con quadratini)
            if self.save_f1_per_class:
                self._save_f1_evolution(round_num, cluster_id, [clean_target_name], dirs["f1"])
            
            # C. Metrics Evolution (Barre consecutive 2x2)
            self._save_metrics_evolution(round_num, cluster_id, [clean_target_name], dirs["evol"])

        # ═══════════════════════════════════════════════════════════════════
        # 4. SALVATAGGIO FINALE NELLO STATE
        # ═══════════════════════════════════════════════════════════════════
        
        # Converti defaultdict → dict normale per serializzazione
        cluster_history_dict = {k: dict(v) for k, v in self.cluster_history.items()}
        f1_history_dict = {k: dict(v) for k, v in self.f1_class_history.items()}

        # Salva nello State con .set()
        state.set("federated.cluster_metrics_history", cluster_history_dict, dict)
        state.set("federated.cluster_f1_class_history", f1_history_dict, dict)

        logger.info(f"✅ Grafici salvati con successo in: {self.log_dir}")
        return {"metrics_saved": True}
        
    """ def _save_classification_report(self, targets, predictions, round_num, cluster_id, output_dir):
        from sklearn.metrics import classification_report
        
        report = classification_report(targets, predictions, 
                                       target_names=self.class_names if self.class_names else None,
                                       zero_division=0, digits=4)
        
        report_path = output_dir / f"report_cluster_{cluster_id}_round_{round_num}.txt"
        with open(report_path, 'w') as f:
            f.write(f"Classification Report - Cluster {cluster_id} - Round {round_num}\n")
            f.write(f"{'='*80}\n\n")
            f.write(report) """
    def _save_classification_report(self, targets, predictions, round_num, cluster_id, cluster_targets, output_dir):
        from sklearn.metrics import classification_report
        
        targets_str = ', '.join(cluster_targets)
        report = classification_report(targets, predictions, 
                                    target_names=self.class_names if self.class_names else None,
                                    zero_division=0, digits=4)
        
        report_path = output_dir / f"report_cluster_{cluster_id}_round_{round_num}.txt"
        with open(report_path, 'w') as f:
            f.write(f"Classification Report - Cluster {cluster_id} ({targets_str}) - Round {round_num}\n")
            f.write(f"{'='*80}\n\n")
            f.write(report)
    
    # def _save_confusion_matrix(self, cm, round_num, cluster_id, cluster_targets, output_dir):
    #     ##Identico al tuo, ma con cluster_id nel titolo/filename
    #     fig, ax = plt.subplots(figsize=(12, 10))
    #     labels = self.class_names if self.class_names else [f"C{i}" for i in range(len(cm))]
        
    #     im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    #     ax.figure.colorbar(im, ax=ax)
    #     ax.set_xticks(np.arange(len(labels)))
    #     ax.set_yticks(np.arange(len(labels)))
    #     ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    #     ax.set_yticklabels(labels, fontsize=9)
    #     ax.set_xlabel('Predicted', fontweight='bold')
    #     ax.set_ylabel('True', fontweight='bold')
        
    #     # CAMBIA SOLO IL TITOLO
    #     targets_str = ', '.join(cluster_targets)
    #     ax.set_title(f'Confusion Matrix - Cluster {cluster_id} ({targets_str}) - Round {round_num}', 
    #                 fontweight='bold', fontsize=14)
    #     # ax.set_title(f'Confusion Matrix - Cluster {cluster_id} - Round {round_num}', 
    #     #              fontweight='bold', fontsize=14)
        
    #     thresh = cm.max() / 2.0
    #     for i in range(len(labels)):
    #         for j in range(len(labels)):
    #             ax.text(j, i, f'{cm[i, j]}', ha="center", va="center",
    #                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
        
    #     plt.tight_layout()
    #     fig.savefig(output_dir / f"cm_cluster_{cluster_id}_round_{round_num}.png", 
    #                dpi=300, bbox_inches='tight')
    #     plt.close(fig)
    # def _save_f1_evolution(self, round_num, cluster_id, cluster_targets, output_dir):
    #     """
    #     Line plot: evoluzione F1 per ogni attacco target del cluster.
    #     Ogni cluster specializza su 1-2 attacchi → 1-2 linee nel grafico.
    #     """
    #     # Se è il primo round, inizializza storico per ogni target
    #     if round_num == 0 or not self.f1_class_history[cluster_id]:
    #         for target_name in cluster_targets:
    #             self.f1_class_history[cluster_id][target_name] = []
        
    #     # ⚠️ CALCOLO F1 PER SINGOLO TARGET
    #     # (Nota: questo richiede di avere predictions/targets con label encoding)
    #     # Per ora usiamo l'F1 aggregato del cluster come placeholder
    #     # TODO: Implementare F1 per-class extraction se necessario
        
    #     # Placeholder: usa F1 aggregato per tutte le target del cluster
    #     cluster_f1 = self.cluster_history[cluster_id]['f1'][-1]
    #     for target_name in cluster_targets:
    #         self.f1_class_history[cluster_id][target_name].append(cluster_f1)
        
    #     # 📊 Plot linea spezzata
    #     fig, ax = plt.subplots(figsize=(14, 8))
        
    #     colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_targets)))
    #     for idx, target_name in enumerate(cluster_targets):
    #         f1_values = self.f1_class_history[cluster_id][target_name]
    #         rounds = self.cluster_history[cluster_id]['rounds'][:len(f1_values)]
            
    #         ax.plot(rounds, f1_values, 
    #                 marker='o', label=target_name, 
    #                 color=colors[idx], linewidth=2.5, markersize=8)
        
    #     ax.set_xlabel('Rounds', fontweight='bold', fontsize=12)
    #     ax.set_ylabel('F1-Score', fontweight='bold', fontsize=12)
    #     ax.set_title(f'F1 Evolution - Cluster {cluster_id}', fontweight='bold', fontsize=15)
    #     ax.legend(title='Attack Types', fontsize=11, title_fontsize=12)
    #     ax.grid(True, alpha=0.3, linestyle='--')
    #     ax.set_ylim([0, 1.05])
        
    #     plt.tight_layout()
    #     fig.savefig(output_dir / f"f1_evolution_cluster_{cluster_id}.png", 
    #             dpi=300, bbox_inches='tight')
    #     plt.close(fig)
    
    # def _save_metrics_evolution(self, round_num, cluster_id, cluster_targets, output_dir):
    #     """
    #     Genera grafico 2x2 con evoluzione di Acc/F1/Prec/Recall 
    #     per il cluster corrente fino al round attuale.
    #     """
    #     targets_str = ', '.join(cluster_targets)
        
    #     fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    #     metrics = ['accuracy', 'f1', 'precision', 'recall']
    #     titles = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
    #     colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
        
    #     for ax, metric, title, color in zip(axes.flat, metrics, titles, colors):
    #         rounds = self.cluster_history[cluster_id]['rounds']
    #         values = self.cluster_history[cluster_id][metric]
            
    #         ax.plot(rounds, values, marker='o', linewidth=2.5, 
    #             markersize=8, color=color, label=title)
    #         ax.set_title(f'{title} - Cluster {cluster_id}', fontweight='bold', fontsize=14)
    #         ax.set_xlabel('Rounds', fontweight='bold')
    #         ax.set_ylabel(title, fontweight='bold')
    #         ax.grid(True, alpha=0.3, linestyle='--')
    #         ax.set_ylim([0, 1.05])
    #         ax.legend()
        
    #     fig.suptitle(f'Metrics Evolution - Cluster {cluster_id} ({targets_str})', 
    #                 fontweight='bold', fontsize=16, y=0.995)
    #     plt.tight_layout()
    #     fig.savefig(output_dir / f"metrics_evolution_cluster_{cluster_id}.png", 
    #             dpi=300, bbox_inches='tight')
    #     plt.close(fig)
    def _save_metrics_evolution(self, round_num, cluster_id, cluster_targets, output_dir):
        """Crea i 4 grafici a barre (Acc, F1, Prec, Rec) con barre consecutive."""
        
        # ✅ AGGIUNGI questo check
        if cluster_id not in self.cluster_history or len(self.cluster_history[cluster_id]['rounds']) == 0:
            logger.warning(f"⚠️ Cluster {cluster_id}: Nessun dato storico, skip plot metrics")
            return
        
        # ✅ Verifica lunghezza minima per plot significativo
        if len(self.cluster_history[cluster_id]['rounds']) < 1:
            logger.debug(f"📊 Cluster {cluster_id}: Solo {len(self.cluster_history[cluster_id]['rounds'])} round, plot comunque")
        
            
        targets_str = ', '.join(cluster_targets)
        
        fig, axes = plt.subplots(2, 2, figsize=(22, 18))
        metrics = ['accuracy', 'f1', 'precision', 'recall']
        titles = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
        
        # Recuperiamo la storia completa salvata nello state
        history = self.cluster_history[cluster_id]
        rounds_data = history['rounds']
        
        # TRUCCO: Convertiamo i round in stringhe per avere barre consecutive e centrate
        rounds_labels = [f"R{r}" for r in rounds_data]
        
        for ax, metric, title, color in zip(axes.flat, metrics, titles, colors):
            values = history[metric]
            
            # Creazione barre
            bars = ax.bar(rounds_labels, values, color=color, alpha=0.8, width=0.6, edgecolor='black')
            # --- FONT ASSE X e Y ---
            ax.tick_params(axis='both', which='major', labelsize=16) # Numeri sui lati più grandi
            
            ax.set_title(f'{title} Evolution', fontweight='bold', fontsize=20, pad=13)
            ax.set_ylim([0, 1.15])
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            
            # Aggiungiamo i valori numerici sopra ogni barra
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=18)
        
        fig.suptitle(f'Metrics History - Cluster {cluster_id} ({targets_str})', 
                     fontweight='bold', fontsize=25, y=0.98)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(output_dir / f"metrics_evolution_cluster_{cluster_id}.png", dpi=300)
        plt.close(fig)

    # def _save_f1_evolution(self, round_num, cluster_id, cluster_targets, output_dir):
    #     """Crea il grafico a linee per l'F1-Score storico del cluster."""
    #     fig, ax = plt.subplots(figsize=(12, 7))
        
    #     # Recuperiamo la storia
    #     history = self.cluster_history[cluster_id]
    #     rounds = history['rounds']
    #     f1_values = history['f1']
        
    #     # Prepariamo il nome dell'attacco (es. "bot")
    #     label_name = cluster_targets[0] if cluster_targets else f"Cluster {cluster_id}"
        
    #     # Disegniamo la linea con i quadratini (marker='s')
    #     ax.plot(rounds, f1_values, 
    #             marker='s',           # Quadratino come l'immagine 3
    #             linestyle='-',        # Linea continua
    #             linewidth=2.5, 
    #             markersize=10, 
    #             color='#1f77b4',      # Blu standard
    #             label=f"Attack: {label_name}")

    #     ax.set_title(f'F1 Evolution - Cluster {cluster_id}', fontweight='bold', fontsize=15)
    #     ax.set_xlabel('Rounds', fontweight='bold')
    #     ax.set_ylabel('F1-Score', fontweight='bold')
        
    #     # Assicuriamoci che l'asse X mostri solo numeri interi di round
    #     ax.set_xticks(rounds)
        
    #     ax.set_ylim([0, 1.05])
    #     ax.grid(True, alpha=0.3, linestyle='--')
    #     ax.legend(loc='upper left', frameon=True)
        
    #     plt.tight_layout()
    #     fig.savefig(output_dir / f"f1_evolution_cluster_{cluster_id}.png", dpi=300)
    #     plt.close(fig)
    def _save_f1_evolution(self, round_num, cluster_id, cluster_targets, output_dir):
        """
        Line plot: evoluzione F1 per ogni attacco target del cluster.
        NOTA: Questa funzione SOLO plotta - non modifica lo storico.
        """
        # ✅ Verifica che ci siano dati da plottare
        if cluster_id not in self.cluster_history or len(self.cluster_history[cluster_id]['rounds']) == 0:
            logger.warning(f"⚠️ Cluster {cluster_id}: Nessun dato storico, skip plot F1")
            return
        
        # ✅ AGGIORNA lo storico F1 per-target (SOLO se questo round non è già stato registrato)
        # Controlliamo se dobbiamo aggiungere il dato per questo round
        cluster_f1 = self.cluster_history[cluster_id]['f1'][-1]  # F1 dell'ultimo round
        
        for target_name in cluster_targets:
            # Inizializza se non esiste
            if target_name not in self.f1_class_history[cluster_id]:
                self.f1_class_history[cluster_id][target_name] = []
            
            # Aggiungi SOLO se non abbiamo già questo round
            # (confronta lunghezza con storico principale)
            if len(self.f1_class_history[cluster_id][target_name]) < len(self.cluster_history[cluster_id]['f1']):
                self.f1_class_history[cluster_id][target_name].append(cluster_f1)
        
        # 📊 Plot linea spezzata
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_targets)))
        for idx, target_name in enumerate(cluster_targets):
            f1_values = self.f1_class_history[cluster_id][target_name]
            rounds = self.cluster_history[cluster_id]['rounds'][:len(f1_values)]
            
            ax.plot(rounds, f1_values, 
                    marker='o', label=target_name, 
                    color=colors[idx], linewidth=2.5, markersize=8)
        
        ax.set_xlabel('Rounds', fontweight='bold', fontsize=12)
        ax.set_ylabel('F1-Score', fontweight='bold', fontsize=12)
        ax.set_title(f'F1 Evolution - Cluster {cluster_id} - {target_name}', fontweight='bold', fontsize=15)
        ax.set_xticks(all_rounds := self.cluster_history[cluster_id]['rounds'])
        ax.set_yticks(np.arange(0, 1.05, 0.05))
        ax.legend(title='Attack Types', fontsize=11, title_fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0, 1.05])
                
        plt.tight_layout()
        fig.savefig(output_dir / f"f1_evolution_cluster_{cluster_id}.png", 
                dpi=300, bbox_inches='tight')
        plt.close(fig)
        
    def _save_confusion_matrix(self, cm, round_num, cluster_id, cluster_targets, output_dir):
        """
        MODIFICATO:
        - Aggiunto padding al titolo (pad=20) per evitare sovrapposizioni.
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        labels = self.class_names if self.class_names else [f"C{i}" for i in range(len(cm))]
        
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        
        ax.set_xlabel('Predicted', fontweight='bold')
        ax.set_ylabel('True', fontweight='bold')
        
        targets_str = ', '.join(cluster_targets)
        # MODIFICA QUI: pad=20 sposta il titolo più in alto
        ax.set_title(f'Confusion Matrix - Cluster {cluster_id} ({targets_str}) - Round {round_num}', 
                     fontweight='bold', fontsize=14, pad=20)
        
        thresh = cm.max() / 2.0
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f'{cm[i, j]}', ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=8)
        
        plt.tight_layout()
        fig.savefig(output_dir / f"cm_cluster_{cluster_id}_round_{round_num}.png", 
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
    """ def run(self, state: State, round_num: int, cluster_results: dict) -> Dict[str, Any]:
        from sklearn.metrics import confusion_matrix, f1_score, classification_report
        
        logger.info(f"\n📊 Salvataggio metriche cluster - Round {round_num}")
        
        # Recupera class_names se non forniti
        if not self.class_names:
            original_label_map = state.get("data.original_label_map", dict)
            if original_label_map:
                max_idx = max(original_label_map.values())
                self.class_names = [None] * (max_idx + 1)
                for name, idx in original_label_map.items():
                    self.class_names[idx] = name
        
        # Itera su ogni cluster
        for cluster_id, result in cluster_results.items():
            # Crea struttura per questo cluster
            cluster_dir = self.log_dir / f"cluster_{cluster_id}"
            cm_dir = cluster_dir / "confusion_matrix"
            report_dir = cluster_dir / "classification_report"
            f1_dir = cluster_dir / "f1_per_classe"

            for d in [cm_dir, report_dir, f1_dir]:
                d.mkdir(parents=True, exist_ok=True)
                
                
            predictions = result.get('predictions')
            targets = result.get('targets')
            
            if predictions is None or targets is None:
                continue
            
            cluster_dir = self.log_dir / f"cluster_{cluster_id}"
            cluster_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Confusion Matrix
            if self.save_confusion_matrix:
                cm = confusion_matrix(targets, predictions)
                self._save_confusion_matrix(cm, round_num, cluster_id, cm_dir)
            
            # 2. Classification Report
            if self.save_classification_report:
                self._save_classification_report(targets, predictions, round_num, 
                                                 cluster_id, report_dir)
            
            # 3. F1 per classe
            if self.save_f1_per_class:
                f1_per_class = f1_score(targets, predictions, average=None, zero_division=0)
                self._save_f1_per_class(f1_per_class, round_num, cluster_id, f1_dir)
        
        
        logger.info(f"✅ Metriche cluster salvate in: {self.log_dir}")
        return {"metrics_saved": True}
    
    def _save_confusion_matrix(self, cm, round_num, cluster_id, output_dir):
        ##Identico al tuo, ma con cluster_id nel titolo/filename
        fig, ax = plt.subplots(figsize=(12, 10))
        labels = self.class_names if self.class_names else [f"C{i}" for i in range(len(cm))]
        
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Predicted', fontweight='bold')
        ax.set_ylabel('True', fontweight='bold')
        ax.set_title(f'Confusion Matrix - Cluster {cluster_id} - Round {round_num}', 
                     fontweight='bold', fontsize=14)
        
        thresh = cm.max() / 2.0
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f'{cm[i, j]}', ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black", fontsize=8)
        
        plt.tight_layout()
        fig.savefig(output_dir / f"cm_cluster_{cluster_id}_round_{round_num}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def _save_classification_report(self, targets, predictions, round_num, cluster_id, output_dir):
        from sklearn.metrics import classification_report
        
        report = classification_report(targets, predictions, 
                                       target_names=self.class_names if self.class_names else None,
                                       zero_division=0, digits=4)
        
        report_path = output_dir / f"report_cluster_{cluster_id}_round_{round_num}.txt"
        with open(report_path, 'w') as f:
            f.write(f"Classification Report - Cluster {cluster_id} - Round {round_num}\n")
            f.write(f"{'='*80}\n\n")
            f.write(report)
    
    def _save_f1_per_class(self, f1_per_class, round_num, cluster_id, output_dir):
        ##Identico al tuo, ma con cluster_id nel titolo
        fig, ax = plt.subplots(figsize=(14, 8))
        labels = self.class_names if self.class_names else [f"C{i}" for i in range(len(f1_per_class))]
        
        colors = ['#1FB804' if f1 >= 0.8 else '#F18F01' if f1 >= 0.5 else '#FF2301' 
                  for f1 in f1_per_class]
        bars = ax.bar(range(len(labels)), f1_per_class, color=colors, alpha=0.8)
        
        for bar, f1_val in zip(bars, f1_per_class):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{f1_val:.4f}', ha='center', va='bottom', fontsize=9)
        
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5)
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5)
        mean_f1 = np.mean(f1_per_class)
        ax.axhline(y=mean_f1, color='blue', linestyle='-', linewidth=2, 
                  label=f'Media: {mean_f1:.3f}')
        
        ax.set_xlabel('Classi', fontweight='bold')
        ax.set_ylabel('F1 Score', fontweight='bold')
        ax.set_title(f'F1 per Classe - Cluster {cluster_id} - Round {round_num}', 
                     fontweight='bold', fontsize=15)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        plt.tight_layout()
        fig.savefig(output_dir / f"f1_cluster_{cluster_id}_round_{round_num}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close(fig) """

#=============================================================================#
#=============================================================================#
class ClusterPlotMetrics_alt(Step):
    """
    Genera grafici aggregati finali confrontando performance tra cluster.
    """
    
    def __init__(
        self,
        output_dir: str,
        rounds_history: List[Dict],
        in_scope: str = "federated",
        out_scope: str = "federated",
        name: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.rounds_history = rounds_history
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        super().__init__(
            name=name or "cluster_plot_metrics",
            in_scope=in_scope,
            out_scope=out_scope,
        )
    
    @Step.requires()
    @Step.provides(plots_generated=bool)
    def run(self, state: State) -> Dict[str, Any]:
        logger.info("\n📊 Generazione Grafici Cluster Comparativi")
        
        data = self._extract_cluster_data()
        
        if not data['eval_rounds']:
            logger.warning("⚠️ Nessun round con valutazione cluster!")
            return {"plots_generated": False}
        
        self._plot_cluster_comparison(data)
        self._plot_cluster_evolution(data)
        
        logger.info(f"✅ Grafici salvati in: {self.output_dir}")
        return {"plots_generated": True}
    
    def _extract_cluster_data(self) -> Dict:
        """Estrae metriche per cluster da round_history"""
        data = {
            'eval_rounds': [],
            'cluster_ids': set(),
            'cluster_metrics': {}  # {cluster_id: {'rounds': [], 'acc': [], 'f1': []}}
        }
        
        for r in self.rounds_history:
            if 'cluster_test_metrics' not in r:
                continue
            
            round_num = r['round_num']
            data['eval_rounds'].append(round_num)
            
            for cluster_id, metrics in r['cluster_test_metrics'].items():
                data['cluster_ids'].add(cluster_id)
                
                if cluster_id not in data['cluster_metrics']:
                    data['cluster_metrics'][cluster_id] = {
                        'rounds': [], 'accuracy': [], 'f1': []
                    }
                
                data['cluster_metrics'][cluster_id]['rounds'].append(round_num)
                data['cluster_metrics'][cluster_id]['accuracy'].append(metrics['accuracy'])
                data['cluster_metrics'][cluster_id]['f1'].append(metrics['f1'])
        
        data['cluster_ids'] = sorted(data['cluster_ids'])
        return data
    
    def _plot_cluster_comparison(self, data: Dict):
        """Confronto finale accuracy/F1 tra cluster (ultimo round)"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        cluster_ids = data['cluster_ids']
        last_round_metrics = {
            cid: {
                'acc': data['cluster_metrics'][cid]['accuracy'][-1],
                'f1': data['cluster_metrics'][cid]['f1'][-1]
            }
            for cid in cluster_ids
        }
        
        # Accuracy comparison
        acc_values = [last_round_metrics[cid]['acc'] for cid in cluster_ids]
        ax1.bar(range(len(cluster_ids)), acc_values, color='steelblue', alpha=0.8)
        ax1.set_xticks(range(len(cluster_ids)))
        ax1.set_xticklabels([f"C{cid}" for cid in cluster_ids])
        ax1.set_ylabel('Accuracy', fontweight='bold')
        ax1.set_title('Accuracy per Cluster (Ultimo Round)', fontweight='bold')
        ax1.set_ylim([0, 1.05])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # F1 comparison
        f1_values = [last_round_metrics[cid]['f1'] for cid in cluster_ids]
        ax2.bar(range(len(cluster_ids)), f1_values, color='#A23B72', alpha=0.8)
        ax2.set_xticks(range(len(cluster_ids)))
        ax2.set_xticklabels([f"C{cid}" for cid in cluster_ids])
        ax2.set_ylabel('F1 Score', fontweight='bold')
        ax2.set_title('F1 Score per Cluster (Ultimo Round)', fontweight='bold')
        ax2.set_ylim([0, 1.05])
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        fig.savefig(self.output_dir / "cluster_comparison.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_cluster_evolution(self, data: Dict):
        """Evoluzione accuracy/F1 nel tempo per ogni cluster"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(data['cluster_ids'])))
        
        for idx, cluster_id in enumerate(data['cluster_ids']):
            metrics = data['cluster_metrics'][cluster_id]
            
            ax1.plot(metrics['rounds'], metrics['accuracy'], 
                    marker='o', label=f"Cluster {cluster_id}", 
                    color=colors[idx], linewidth=2)
            
            ax2.plot(metrics['rounds'], metrics['f1'], 
                    marker='s', label=f"Cluster {cluster_id}", 
                    color=colors[idx], linewidth=2)
        
        ax1.set_ylabel('Accuracy', fontweight='bold')
        ax1.set_title('Evoluzione Accuracy per Cluster', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])
        
        ax2.set_xlabel('Round', fontweight='bold')
        ax2.set_ylabel('F1 Score', fontweight='bold')
        ax2.set_title('Evoluzione F1 per Cluster', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.05])
        
        plt.tight_layout()
        fig.savefig(self.output_dir / "cluster_evolution.png", dpi=300, bbox_inches='tight')
        plt.close(fig)