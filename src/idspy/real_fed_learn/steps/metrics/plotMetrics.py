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

class ClusterRoundMetrics_C(Step):
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
        
        # ✅ NUOVO: Un solo target comune per tutti i cluster
        target_name = cluster_targets[0]  # Nel nuovo approccio c'è UN solo attacco
        cluster_f1 = self.cluster_history[cluster_id]['f1'][-1]

        # Inizializza se non esiste
        if target_name not in self.f1_class_history[cluster_id]:
            self.f1_class_history[cluster_id][target_name] = []

        # Aggiungi SOLO se non abbiamo già questo round
        if len(self.f1_class_history[cluster_id][target_name]) < len(self.cluster_history[cluster_id]['f1']):
            self.f1_class_history[cluster_id][target_name].append(cluster_f1)
        
        # 📊 Plot linea spezzata (UN SOLO TARGET)
        fig, ax = plt.subplots(figsize=(16, 9))

        target_name = cluster_targets[0]  # Attacco comune
        f1_values = self.f1_class_history[cluster_id][target_name]
        rounds = self.cluster_history[cluster_id]['rounds'][:len(f1_values)]

        # Linea con marker grandi
        ax.plot(rounds, f1_values, 
                marker='o', label=f'{target_name}', 
                color='#E63946', linewidth=3.5, markersize=12, 
                markerfacecolor='white', markeredgewidth=2.5)

        # Aggiungi valori numerici sopra ogni punto
        for i, (r, f1) in enumerate(zip(rounds, f1_values)):
            ax.text(r, f1 + 0.02, f'{f1:.3f}', 
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax.set_xlabel('Rounds', fontweight='bold', fontsize=18)
        ax.set_ylabel('F1-Score', fontweight='bold', fontsize=18)
        ax.set_title(f'F1 Evolution - Cluster {cluster_id} ({target_name})', 
                    fontweight='bold', fontsize=22, pad=15)
        ax.set_xticks(self.cluster_history[cluster_id]['rounds'])
        ax.set_xticklabels([f"R{r}" for r in self.cluster_history[cluster_id]['rounds']], 
                            fontsize=16)
        ax.set_yticks(np.arange(0, 1.05, 0.1))
        ax.tick_params(axis='y', labelsize=16)
        ax.legend(fontsize=15, frameon=True, shadow=True, loc='lower right')
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.2)
        ax.set_ylim([0, 1.08])
                
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
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=16)
        ax.set_yticklabels(labels, fontsize=16)

        ax.set_xlabel('Predicted', fontweight='bold', fontsize=18)
        ax.set_ylabel('True', fontweight='bold', fontsize=18)
        
        targets_str = ', '.join(cluster_targets)
        # MODIFICA QUI: pad=20 sposta il titolo più in alto
        ax.set_title(f'Confusion Matrix - Cluster {cluster_id} ({targets_str}) - Round {round_num}', 
                     fontweight='bold', fontsize=19, pad=20)
        
        thresh = cm.max() / 2.0
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f'{cm[i, j]}', ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", 
                        fontsize=20, fontweight='bold')
        
        plt.tight_layout()
        fig.savefig(output_dir / f"cm_cluster_{cluster_id}_round_{round_num}.png", 
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
 
#=============================================================================#
#=============================================================================#
class ClusterPlotMetrics_C(Step):
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
        logger.info("\n📊 Generazione Grafici Cluster Comparativi + Metriche FL")
        
        data = self._extract_cluster_data()
        
        if not data['eval_rounds']:
            logger.warning("⚠️ Nessun round con valutazione cluster!")
            return {"plots_generated": False}
        
        # Grafici comparativi cluster
        self._plot_cluster_comparison(data)
        self._plot_cluster_evolution(data)
        
        # ✅ NUOVI GRAFICI: Metriche Federated Learning
        self._plot_std_loss_evolution()
        self._plot_num_clusters_evolution(state)
        
        # ⚠️ Similarity richiede dati dal server (implementare dopo)
        self._plot_similarity_metrics(state)
        
        logger.info(f"✅ Grafici salvati in: {self.output_dir}")
        return {"plots_generated": True}
    
    @staticmethod
    def _format_small_value(val: float) -> str:
        """
        Formatta valori piccoli in modo intelligente.
        - Se < 0.001 → notazione scientifica (1.2e-5)
        - Se >= 0.001 → formato decimale (0.0123)
        """
        if val == 0:
            return '0'
        elif abs(val) < 1e-3:
            # Notazione scientifica compatta
            formatted = f'{val:.1e}'
            # Rimuovi lo zero superfluo: e-05 → e-5
            formatted = formatted.replace('e-0', 'e-').replace('e+0', 'e+')
            return formatted
        else:
            return f'{val:.4f}'
    
    
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
        #===============================================================#
        ax1.set_ylabel('Accuracy', fontweight='bold', fontsize=18)
        ax1.set_title('Accuracy per Cluster (Ultimo Round)', fontweight='bold', fontsize=20, pad=15)
        ax1.tick_params(axis='both', labelsize=16)

        # Aggiungi valori sopra barre
        for i, v in enumerate(acc_values):
            ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', 
                    fontsize=16, fontweight='bold')
        #===============================================================#
        ax1.set_ylim([0, 1.05])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # F1 comparison
        f1_values = [last_round_metrics[cid]['f1'] for cid in cluster_ids]
        ax2.bar(range(len(cluster_ids)), f1_values, color='#A23B72', alpha=0.8)
        ax2.set_xticks(range(len(cluster_ids)))
        ax2.set_xticklabels([f"C{cid}" for cid in cluster_ids])
        #===============================================================#
        ax2.set_ylabel('F1 Score', fontweight='bold', fontsize=18)
        ax2.set_title('F1 Score per Cluster (Ultimo Round)', fontweight='bold', fontsize=20, pad=15)
        ax2.tick_params(axis='both', labelsize=16)

        # Aggiungi valori sopra barre
        for i, v in enumerate(f1_values):
            ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', 
                    fontsize=16, fontweight='bold')
        #===============================================================#
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
        
        #===============================================================#
        ax1.set_ylabel('Accuracy', fontweight='bold', fontsize=18)
        ax1.set_title('Evoluzione Accuracy per Cluster', fontweight='bold', fontsize=20, pad=15)
        ax1.legend(fontsize=14, frameon=True, shadow=True)
        ax1.tick_params(axis='both', labelsize=16)
        ax1.set_xticks(data['eval_rounds'])
        ax1.set_xticklabels([f"R{r}" for r in data['eval_rounds']], fontsize=16)
        #===============================================================#
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])
        
        #===============================================================#
        ax2.set_xlabel('Round', fontweight='bold', fontsize=18)
        ax2.set_ylabel('F1 Score', fontweight='bold', fontsize=18)
        ax2.set_title('Evoluzione F1 per Cluster', fontweight='bold', fontsize=20, pad=15)
        ax2.legend(fontsize=14, frameon=True, shadow=True)
        ax2.tick_params(axis='both', labelsize=16)
        ax2.set_xticks(data['eval_rounds'])
        ax2.set_xticklabels([f"R{r}" for r in data['eval_rounds']], fontsize=16)
        #===============================================================#
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.05])
        
        plt.tight_layout()
        fig.savefig(self.output_dir / "cluster_evolution.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_std_loss_evolution(self):
        """
        Mostra std_loss tra client per round.
        Più basso = più convergenza (clustering funziona).
        """
        fig, ax = plt.subplots(figsize=(16, 9))
        
        rounds = []
        std_losses = []
        
        for r in self.rounds_history:
            if 'round_metrics' in r and 'std_loss' in r['round_metrics']:
                rounds.append(r['round_num'])
                std_losses.append(r['round_metrics']['std_loss'])
        
        if not rounds:
            logger.warning("⚠️ Nessun dato std_loss disponibile")
            plt.close(fig)
            return
        
        # Plot linea con area riempita
        ax.plot(rounds, std_losses, 
                marker='D', color='#F4442E', linewidth=3.5, 
                markersize=12, label='STD Loss', 
                markerfacecolor='white', markeredgewidth=2.5)
        
        ax.fill_between(rounds, 0, std_losses, alpha=0.2, color='#F4442E')
        
        # ═══════════════════════════════════════════════════════════════
        # ✅ NUOVO: Posizionamento intelligente dei label
        # ═══════════════════════════════════════════════════════════════
        
        # Calcola limiti dell'asse Y (prima del rendering)
        max_std = max(std_losses) if std_losses else 1e-5
        
        # Imposta Y limit in modo da vedere bene i dati
        # Se max < 1e-4, usa scala adatta ai valori piccoli
        if max_std < 1e-4:
            y_upper = max_std * 1.3  # 30% margin sopra il max
        else:
            y_upper = max(max_std * 1.2, 1e-4)  # Almeno 1e-4 per visibilità
        
        ax.set_ylim([0, y_upper])
        
        # Offset dinamico (3% del range)
        offset = y_upper * 0.03
        
        # Threshold per posizionamento sotto (se supera 85% della griglia)
        y_threshold = y_upper * 0.85
        
        for r, std in zip(rounds, std_losses):
            # ✅ Formatta valore con notazione scientifica se necessario
            label_text = self._format_small_value(std)
            
            # ✅ Posiziona sotto se il marker è troppo alto
            if std > y_threshold:
                y_pos = std - offset
                v_align = 'top'
            else:
                y_pos = std + offset
                v_align = 'bottom'
            
            ax.text(r, y_pos, label_text, 
                    ha='center', va=v_align, 
                    fontsize=14, fontweight='bold')
        
        # ═══════════════════════════════════════════════════════════════
        # ✅ NUOVO: Formattazione asse Y con notazione scientifica
        # ═══════════════════════════════════════════════════════════════
        
        from matplotlib.ticker import FuncFormatter
        
        def y_formatter(val, pos):
            """Formatta asse Y con notazione scientifica per valori piccoli"""
            if val == 0:
                return '0'
            elif val < 1e-3:
                return f'{val:.1e}'.replace('e-0', 'e-')
            else:
                return f'{val:.4f}'
        
        ax.yaxis.set_major_formatter(FuncFormatter(y_formatter))
        
        # ═══════════════════════════════════════════════════════════════
        
        ax.set_xlabel('Rounds', fontweight='bold', fontsize=18)
        ax.set_ylabel('Standard Deviation of Loss', fontweight='bold', fontsize=18)
        ax.set_title('Loss Variance Between Clients (Lower = Better Convergence)', 
                    fontweight='bold', fontsize=22, pad=15)
        ax.set_xticks(rounds)
        ax.set_xticklabels([f"R{r}" for r in rounds], fontsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.2)
        ax.legend(fontsize=15, frameon=True, shadow=True)
        
        plt.tight_layout()
        fig.savefig(self.output_dir / "std_loss_evolution.png", 
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info("  ✅ Grafico STD Loss salvato")
    
    """  def _plot_std_loss_evolution(self):
        
        # Mostra std_loss tra client per round.
        # Più basso = più convergenza (clustering funziona).
        
        fig, ax = plt.subplots(figsize=(16, 9))
        
        rounds = []
        std_losses = []
        
        for r in self.rounds_history:
            if 'round_metrics' in r and 'std_loss' in r['round_metrics']:
                rounds.append(r['round_num'])
                std_losses.append(r['round_metrics']['std_loss'])
        
        if not rounds:
            logger.warning("⚠️ Nessun dato std_loss disponibile")
            plt.close(fig)
            return
        
        # Plot linea con area riempita
        ax.plot(rounds, std_losses, 
                marker='D', color='#F4442E', linewidth=3.5, 
                markersize=12, label='STD Loss', 
                markerfacecolor='white', markeredgewidth=2.5)
        
        ax.fill_between(rounds, 0, std_losses, alpha=0.2, color='#F4442E')
        
        # Aggiungi valori numerici
        for r, std in zip(rounds, std_losses):
            ax.text(r, std + max(std_losses)*0.03, f'{std:.4f}', 
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax.set_xlabel('Rounds', fontweight='bold', fontsize=18)
        ax.set_ylabel('Standard Deviation of Loss', fontweight='bold', fontsize=18)
        ax.set_title('Loss Variance Between Clients (Lower = Better Convergence)', 
                    fontweight='bold', fontsize=22, pad=15)
        ax.set_xticks(rounds)
        ax.set_xticklabels([f"R{r}" for r in rounds], fontsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.2)
        ax.legend(fontsize=15, frameon=True, shadow=True)
        
        plt.tight_layout()
        fig.savefig(self.output_dir / "std_loss_evolution.png", 
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info("  ✅ Grafico STD Loss salvato") """
    
    def _plot_num_clusters_evolution(self, state: State):
        """
        Mostra come il numero di cluster cambia nel tempo.
        Evidenzia adattività del sistema.
        """
        fig, ax = plt.subplots(figsize=(16, 9))
        
        rounds = []
        num_clusters_list = []
        
        # Recupera clustering_metrics salvate dal server
        for r in self.rounds_history:
            round_num = r['round_num']
            
            # Prova a recuperare dalle metriche di aggregazione
            if 'aggregation_metrics' in r:
                # Conta cluster unici da client_to_cluster
                # (Nota: questo richiede che il server salvi num_clusters)
                pass
            
            # Alternativa: conta cluster da cluster_test_metrics
            if 'cluster_test_metrics' in r:
                num_clusters = len(r['cluster_test_metrics'])
                rounds.append(round_num)
                num_clusters_list.append(num_clusters)
        
        if not rounds:
            logger.warning("⚠️ Nessun dato clustering disponibile")
            plt.close(fig)
            return
        
        # Plot barre
        colors = ['#06A77D' if nc > 1 else '#F4A261' for nc in num_clusters_list]
        bars = ax.bar([f"R{r}" for r in rounds], num_clusters_list, 
                    color=colors, alpha=0.85, edgecolor='black', linewidth=2)
        
        # Aggiungi valori sopra barre
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', 
                    fontsize=18, fontweight='bold')
        
        ax.set_xlabel('Rounds', fontweight='bold', fontsize=18)
        ax.set_ylabel('Number of Clusters', fontweight='bold', fontsize=18)
        ax.set_title('Cluster Adaptivity Over Time (Dynamic Clustering)', 
                    fontweight='bold', fontsize=22, pad=15)
        ax.tick_params(axis='both', labelsize=16)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_ylim([0, max(num_clusters_list) + 1])
        
        plt.tight_layout()
        fig.savefig(self.output_dir / "num_clusters_evolution.png", 
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info("  ✅ Grafico Numero Cluster salvato")
    
    def _plot_similarity_metrics(self, state: State):
        """
        Mostra cosine similarity intra-cluster vs inter-cluster.
        Ideale: Alta intra, Bassa inter (clustering efficace).
        """
        fig, ax = plt.subplots(figsize=(16, 9))
        
        rounds = []
        avg_intra_sim = []
        avg_inter_sim = []
        
        # NUOVO CODICE CON DATI REALI
        # Recupera similarity_matrix salvata dal server
        for r in self.rounds_history:
            clustering_metrics = r.get('clustering_metrics', {})
            
            # Salta round senza clustering (es. round 0)
            if not clustering_metrics or 'avg_intra_similarity' not in clustering_metrics:
                continue
            
            round_num = r['round_num']
            rounds.append(round_num)
            
            # ✅ Dati REALI dal server
            avg_intra_sim.append(clustering_metrics['avg_intra_similarity'])
            avg_inter_sim.append(clustering_metrics['avg_inter_similarity'])
        
        if not rounds:
            logger.warning("⚠️ Nessun dato similarity disponibile")
            plt.close(fig)
            return
        
        # Plot doppia linea
        ax.plot(rounds, avg_intra_sim, 
                marker='o', color='#06A77D', linewidth=3.5, 
                markersize=12, label='Intra-Cluster (↑ Better)', 
                markerfacecolor='white', markeredgewidth=2.5)
        
        ax.plot(rounds, avg_inter_sim, 
                marker='s', color='#E63946', linewidth=3.5, 
                markersize=12, label='Inter-Cluster (↓ Better)', 
                markerfacecolor='white', markeredgewidth=2.5)
        
        ax.axhline(y=0.85, color='green', linestyle='--', alpha=0.5, linewidth=2, 
                label='Target Intra (>0.85)')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=2, 
                label='Target Inter (<0.5)')
        
        ax.set_xlabel('Rounds', fontweight='bold', fontsize=18)
        ax.set_ylabel('Cosine Similarity', fontweight='bold', fontsize=18)
        ax.set_title('Cluster Cohesion vs Separation (Unsupervised Quality)', 
                    fontweight='bold', fontsize=22, pad=15)
        ax.set_xticks(rounds)
        ax.set_xticklabels([f"R{r}" for r in rounds], fontsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.2)
        ax.legend(fontsize=14, frameon=True, shadow=True, loc='best')
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        fig.savefig(self.output_dir / "similarity_metrics.png", 
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info("  ✅ Grafico Similarity salvato")