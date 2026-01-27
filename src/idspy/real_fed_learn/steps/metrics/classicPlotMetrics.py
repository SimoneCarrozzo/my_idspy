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

class Classic_RoundMetrics(Step):
    
    """Salva metriche per approccio CLASSICO (no clustering)"""
    
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
        
        # ✅ SEMPLIFICATO: solo 1 "cluster" (global model)
        self.global_history = {
            'rounds': [], 'accuracy': [], 'f1': [], 'precision': [], 'recall': []
        }
        
        super().__init__(
            name=name or "classic_round_metrics",
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
        
        # ✅ CORRETTO: Usa global_history (NO cluster_history)
        if state.has("federated.classic_metrics_history"):
            saved_history = state.get("federated.classic_metrics_history", dict)
            # Ricostruisci da dict salvato
            self.global_history = saved_history if saved_history else {
                'rounds': [], 'accuracy': [], 'f1': [], 'precision': [], 'recall': []
            }
        else:
            # Round 0: già inizializzato in __init__
            pass  # self.global_history già esiste


        # ✅ SEMPLIFICA F1 history
        if state.has("federated.classic_f1_history"):
            self.f1_history = state.get("federated.classic_f1_history", dict)
        else:
            self.f1_history = {}

        # ✅ AGGIUNGI QUESTO LOG:
        logger.debug(f"🔍 [Classic_RoundMetrics] History caricata: {len(self.global_history['rounds'])} rounds precedenti")
        logger.debug(f"🔍 [Classic_RoundMetrics] F1 history ha {len(self.f1_history)} targets")    

        # ═══════════════════════════════════════════════════════════════════
        # 2. ITERAZIONE SUI RISULTATI DI OGNI CLUSTER
        # ═══════════════════════════════════════════════════════════════════
        
        # ✅ ASSUMI cluster_id=0 FISSO (global model unico)
        if 0 not in cluster_results_detailed:
            logger.error("❌ Nessun risultato per modello globale!")
            return {"metrics_saved": False}

        result = cluster_results_detailed[0]  # ← Unico cluster

        # ✅ AGGIUNGI QUESTI LOG:
        logger.debug(f"🔍 [Classic_RoundMetrics] Cluster 0 result keys: {result.keys()}")
        logger.debug(f"🔍 [Classic_RoundMetrics] Predictions shape: {len(result.get('predictions', []))}")
        logger.debug(f"🔍 [Classic_RoundMetrics] Target column: {result.get('target_column', 'N/A')}")
        
        # Prepara cartelle output
        global_dir = self.log_dir / "global_model"
        dirs = {
            "cm": global_dir / "confusion_matrix",
            "f1": global_dir / "f1_per_classe",
            "evol": global_dir / "evolution"
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        # Estrai dati
        predictions = result.get('predictions')
        targets = result.get('targets')
        target_col_name = result.get('target_column', "Global_Model")
        clean_target_name = target_col_name.replace('is_', '')

        # ✅ AGGIORNA global_history
        if round_num not in self.global_history['rounds']:
            self.global_history['rounds'].append(round_num)
            self.global_history['accuracy'].append(result['accuracy'])
            self.global_history['f1'].append(result['f1'])
            self.global_history['precision'].append(result['precision'])
            self.global_history['recall'].append(result['recall'])

        # ✅ AGGIUNGI QUESTO LOG:
        logger.debug(f"🔍 [Classic_RoundMetrics] Global history aggiornata: rounds={self.global_history['rounds']}")
        logger.debug(f"🔍 [Classic_RoundMetrics] Ultimo F1: {self.global_history['f1'][-1]:.4f}")
        
        # ═══════════════════════════════════════════════════════════════════
        # 3. GENERAZIONE GRAFICI
        # ═══════════════════════════════════════════════════════════════════
        # USANO cluster_id=0 fisso!!
        
        # A. Confusion Matrix
        if self.save_confusion_matrix and predictions is not None:
            cm = confusion_matrix(targets, predictions)
            self._save_confusion_matrix(cm, round_num, 0, [clean_target_name], dirs["cm"])

        # B. F1 Evolution (Linea con quadratini)
        if self.save_f1_per_class:
            self._save_f1_evolution(round_num, 0, [clean_target_name], dirs["f1"])

        # C. Metrics Evolution (Barre consecutive 2x2)
        self._save_metrics_evolution(round_num, 0, [clean_target_name], dirs["evol"])
            
        # D. Classification Report
        if self.save_classification_report and predictions is not None:
            self._save_classification_report(targets, predictions, round_num, 0, [clean_target_name], global_dir)

        # ═══════════════════════════════════════════════════════════════════
        # 4. SALVATAGGIO FINALE NELLO STATE
        # ═══════════════════════════════════════════════════════════════════
        
        # ✅ AGGIUNGI:
        logger.debug(f"🔍 [Classic_RoundMetrics] Salvataggio nello state:")
        logger.debug(f"   - global_history rounds: {self.global_history['rounds']}")
        logger.debug(f"   - f1_history targets: {list(self.f1_history.keys())}")
                
        # ✅ Salva global_history (è già un dict)
        state.set("federated.classic_metrics_history", self.global_history, dict)
        state.set("federated.classic_f1_history", self.f1_history, dict)

        logger.info(f"✅ Grafici salvati con successo in: {self.log_dir}")
        return {"metrics_saved": True}
        

    def _save_classification_report(self, targets, predictions, round_num, cluster_id, cluster_targets, output_dir):
        from sklearn.metrics import classification_report
        
        targets_str = ', '.join(cluster_targets)
        report = classification_report(targets, predictions, 
                                    target_names=self.class_names if self.class_names else None,
                                    zero_division=0, digits=4)
        
        report_path = output_dir / f"report_global_model_round_{round_num}.txt"  # ✅
        with open(report_path, 'w') as f:
            f.write(f"Classification Report - Global Model ({targets_str}) - Round {round_num}\n")  # ✅
            f.write(f"{'='*80}\n\n")
            f.write(report)

    def _save_metrics_evolution(self, round_num, cluster_id, cluster_targets, output_dir):
        """Crea i 4 grafici a barre (Acc, F1, Prec, Rec) con barre consecutive."""
        
        # ✅ USA global_history (NON cluster_history)
        if len(self.global_history['rounds']) == 0:
            logger.warning(f"⚠️ Global Model: Nessun dato storico, skip plot")
            return
            
        targets_str = ', '.join(cluster_targets)
        
        fig, axes = plt.subplots(2, 2, figsize=(22, 18))
        metrics = ['accuracy', 'f1', 'precision', 'recall']
        titles = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
        
        # Recuperiamo la storia completa salvata nello state
        history = self.global_history  # ← SEMPRE global
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
        
        fig.suptitle(f'Metrics History - Global Model ({targets_str})', 
                     fontweight='bold', fontsize=25, y=0.98)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(output_dir / f"metrics_evolution_cluster_{cluster_id}.png", dpi=300)
        plt.close(fig)

    def _save_f1_evolution(self, round_num, cluster_id, cluster_targets, output_dir):
        """
        Line plot: evoluzione F1 per il modello globale.
        """
        # ✅ USA global_history (NON cluster_history)
        if len(self.global_history['rounds']) == 0:
            logger.warning(f"⚠️ Global Model: Nessun dato storico, skip plot")
            return

        history = self.global_history  # ← CORRETTO
        
        # ✅ Aggiorna F1 per-target (UN solo target per baseline)
        target_name = cluster_targets[0]
        
        # Inizializza se non esiste
        if target_name not in self.f1_history:
            self.f1_history[target_name] = []
        
        # Aggiungi F1 corrente SOLO se non già presente
        current_f1 = history['f1'][-1]
        if len(self.f1_history[target_name]) < len(history['f1']):
            self.f1_history[target_name].append(current_f1)
        
        # 📊 Plot linea spezzata
        fig, ax = plt.subplots(figsize=(16, 9))
        
        f1_values = self.f1_history[target_name]
        rounds = history['rounds'][:len(f1_values)]
        
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
        ax.set_title(f'F1 Evolution - Global Model ({target_name})',  # ✅ NON "Cluster"
                    fontweight='bold', fontsize=22, pad=15)
        ax.set_xticks(history['rounds'])
        ax.set_xticklabels([f"R{r}" for r in history['rounds']], fontsize=16)
        ax.set_yticks(np.arange(0, 1.05, 0.1))
        ax.tick_params(axis='y', labelsize=16)
        ax.legend(fontsize=15, frameon=True, shadow=True, loc='lower right')
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.2)
        ax.set_ylim([0, 1.08])
                
        plt.tight_layout()
        fig.savefig(output_dir / f"f1_evolution_global_model.png",  # ✅ NON "cluster"
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
        ax.set_title(f'Confusion Matrix - Global Model ({targets_str}) - Round {round_num}',  # ✅
             fontweight='bold', fontsize=19, pad=20)
        
        thresh = cm.max() / 2.0
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f'{cm[i, j]}', ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", 
                        fontsize=20, fontweight='bold')
        
        plt.tight_layout()
        fig.savefig(output_dir / f"cm_global_model_round_{round_num}.png",  # ✅
            dpi=300, bbox_inches='tight')
        plt.close(fig)
 
#=============================================================================#
#=============================================================================#
class Classic_PlotMetrics(Step):
    """
    Grafici finali per FedAvg CLASSICO (no clustering).
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
        
        # ✅ AGGIUNGI QUESTI LOG:
        logger.debug(f"🔍 [Classic_PlotMetrics] Rounds history: {len(self.rounds_history)} rounds")
        logger.debug(f"🔍 [Classic_PlotMetrics] Output dir: {self.output_dir}")
        
        data = self._extract_cluster_data()
        
        # ✅ AGGIUNGI QUESTI LOG:
        logger.debug(f"🔍 [Classic_PlotMetrics] Eval rounds: {data['eval_rounds']}")
        logger.debug(f"🔍 [Classic_PlotMetrics] Cluster IDs trovati: {data['cluster_ids']}")  # Deve essere [0]

        
        if not data['eval_rounds']:
            logger.warning("⚠️ Nessun round con valutazione cluster!")
            return {"plots_generated": False}
        
        
        # ✅ NUOVI GRAFICI: Metriche Federated Learning
        self._plot_std_loss_evolution()
        # self._plot_num_clusters_evolution(state)
        
        # ✅ AGGIUNGI: Grafici baseline-specific
        self._plot_global_f1_evolution()  # ← F1 del modello globale
        self._plot_global_accuracy_evolution()  # ← Accuracy
        
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
    
    """ def _plot_std_loss_evolution(self):
        
        # Mostra std_loss tra client per round.
        # Più basso = più convergenza (clustering funziona).
        
        fig, ax = plt.subplots(figsize=(16, 9))
        
        rounds = []
        std_losses = []
        
        for r in self.rounds_history:
            if 'round_metrics' in r and 'std_loss' in r['round_metrics']:
                rounds.append(r['round_num'])
                std_losses.append(r['round_metrics']['std_loss'])
        
        # ✅ AGGIUNGI QUESTO LOG:
        logger.debug(f"🔍 [STD Loss Plot] Dati raccolti: {len(rounds)} rounds")
        logger.debug(f"🔍 [STD Loss Plot] STD Loss values: {std_losses}")
        
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
        logger.info("  ✅ Grafico STD Loss salvato")
    
    # def _plot_num_clusters_evolution(self, state: State):
    #    
    #     Mostra come il numero di cluster cambia nel tempo.
    #     Evidenzia adattività del sistema.
    #    
    #     fig, ax = plt.subplots(figsize=(16, 9))
        
    #     rounds = []
    #     num_clusters_list = []
        
    #     # Recupera clustering_metrics salvate dal server
    #     for r in self.rounds_history:
    #         round_num = r['round_num']
            
    #         # Prova a recuperare dalle metriche di aggregazione
    #         if 'aggregation_metrics' in r:
    #             # Conta cluster unici da client_to_cluster
    #             # (Nota: questo richiede che il server salvi num_clusters)
    #             pass
            
    #         # Alternativa: conta cluster da cluster_test_metrics
    #         if 'cluster_test_metrics' in r:
    #             num_clusters = len(r['cluster_test_metrics'])
    #             rounds.append(round_num)
    #             num_clusters_list.append(num_clusters)
        
    #     if not rounds:
    #         logger.warning("⚠️ Nessun dato clustering disponibile")
    #         plt.close(fig)
    #         return
        
    #     # Plot barre
    #     colors = ['#06A77D' if nc > 1 else '#F4A261' for nc in num_clusters_list]
    #     bars = ax.bar([f"R{r}" for r in rounds], num_clusters_list, 
    #                 color=colors, alpha=0.85, edgecolor='black', linewidth=2)
        
    #     # Aggiungi valori sopra barre
    #     for bar in bars:
    #         height = bar.get_height()
    #         ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
    #                 f'{int(height)}', ha='center', va='bottom', 
    #                 fontsize=18, fontweight='bold')
        
    #     ax.set_xlabel('Rounds', fontweight='bold', fontsize=18)
    #     ax.set_ylabel('Number of Clusters', fontweight='bold', fontsize=18)
    #     ax.set_title('Cluster Adaptivity Over Time (Dynamic Clustering)', 
    #                 fontweight='bold', fontsize=22, pad=15)
    #     ax.tick_params(axis='both', labelsize=16)
    #     ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    #     ax.set_ylim([0, max(num_clusters_list) + 1])
        
    #     plt.tight_layout()
    #     fig.savefig(self.output_dir / "num_clusters_evolution.png", 
    #                 dpi=300, bbox_inches='tight')
    #     plt.close(fig)
    #     logger.info("  ✅ Grafico Numero Cluster salvato")
    
    def _plot_global_f1_evolution(self):
        # Plot dell'evoluzione del F1 del modello globale (FedAvg Classico).
        fig, ax = plt.subplots(figsize=(16, 9))
        
        rounds = []
        f1_values = []
                
        for r in self.rounds_history:
            # ✅ F1/Accuracy sono in cluster_test_metrics[0] (global model)
            if 'cluster_test_metrics' in r and 0 in r['cluster_test_metrics']:
                rounds.append(r['round_num'])
                f1_values.append(r['cluster_test_metrics'][0]['f1'])
        
        # ✅ AGGIUNGI QUESTI LOG:
        logger.debug(f"🔍 [F1 Plot] Rounds con valutazione: {rounds}")
        logger.debug(f"🔍 [F1 Plot] F1 values: {f1_values}")
            
        if not rounds:
            logger.warning("⚠️ Nessun dato F1 globale disponibile")
            plt.close(fig)
            return
        
        # Plot linea con marker
        ax.plot(rounds, f1_values, 
                marker='o', color='#1D3557', linewidth=3.5, 
                markersize=12, label='Global Model F1', 
                markerfacecolor='white', markeredgewidth=2.5)
        
        # Aggiungi valori numerici
        for r, f1 in zip(rounds, f1_values):
            ax.text(r, f1 + 0.02, f'{f1:.3f}', 
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax.set_xlabel('Rounds', fontweight='bold', fontsize=18)
        ax.set_ylabel('F1-Score', fontweight='bold', fontsize=18)
        ax.set_title('Global Model F1 Evolution (Classic FedAvg)', 
                    fontweight='bold', fontsize=22, pad=15)
        ax.set_xticks(rounds)
        ax.set_xticklabels([f"R{r}" for r in rounds], fontsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.2)
        ax.legend(fontsize=15, frameon=True, shadow=True)
        
        plt.tight_layout()
        fig.savefig(self.output_dir / "global_f1_evolution.png", 
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info("  ✅ Grafico F1 Globale salvato")
    
    def _plot_global_accuracy_evolution(self):
        # Plot dell'evoluzione dell'Accuracy del modello globale (FedAvg Classico).
        fig, ax = plt.subplots(figsize=(16, 9))
        
        rounds = []
        acc_values = []
        
        for r in self.rounds_history:
            # ✅ Accuracy sono in cluster_test_metrics[0] (global model)
            if 'cluster_test_metrics' in r and 0 in r['cluster_test_metrics']:
                rounds.append(r['round_num'])
                acc_values.append(r['cluster_test_metrics'][0]['accuracy'])
        
        # ✅ AGGIUNGI QUESTI LOG:
        logger.debug(f"🔍 [Accuracy Plot] Rounds con valutazione: {rounds}")
        logger.debug(f"🔍 [Accuracy Plot] Accuracy values: {acc_values}")
        
        if not rounds:
            logger.warning("⚠️ Nessun dato Accuracy globale disponibile")
            plt.close(fig)
            return
        
        # Plot linea con marker
        ax.plot(rounds, acc_values, 
                marker='s', color='#457B9D', linewidth=3.5, 
                markersize=12, label='Global Model Accuracy', 
                markerfacecolor='white', markeredgewidth=2.5)
        
        # Aggiungi valori numerici
        for r, acc in zip(rounds, acc_values):
            ax.text(r, acc + 0.02, f'{acc:.3f}', 
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax.set_xlabel('Rounds', fontweight='bold', fontsize=18)
        ax.set_ylabel('Accuracy', fontweight='bold', fontsize=18)
        ax.set_title('Global Model Accuracy Evolution (Classic FedAvg)', 
                    fontweight='bold', fontsize=22, pad=15)
        ax.set_xticks(rounds)
        ax.set_xticklabels([f"R{r}" for r in rounds], fontsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.2)
        ax.legend(fontsize=15, frameon=True, shadow=True)
        
        plt.tight_layout()
        fig.savefig(self.output_dir / "global_accuracy_evolution.png", 
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info("  ✅ Grafico Accuracy Globale salvato") """
    def _plot_global_accuracy_evolution(self):
        """Plot dell'evoluzione dell'Accuracy del modello globale (FedAvg Classico)."""
        fig, ax = plt.subplots(figsize=(16, 9))
        
        rounds = []
        acc_values = []
        
        for r in self.rounds_history:
            if 'cluster_test_metrics' in r and 0 in r['cluster_test_metrics']:
                rounds.append(r['round_num'])
                acc_values.append(r['cluster_test_metrics'][0]['accuracy'])
        
        logger.debug(f"🔍 [Accuracy Plot] Rounds: {rounds}, Values: {acc_values}")
        
        if not rounds:
            logger.warning("⚠️ Nessun dato Accuracy globale disponibile")
            plt.close(fig)
            return
        
        # Plot linea
        ax.plot(rounds, acc_values, 
                marker='s', color='#457B9D', linewidth=3.5, 
                markersize=12, label='Global Model Accuracy', 
                markerfacecolor='white', markeredgewidth=2.5)
        
        # ✅ POSIZIONAMENTO INTELLIGENTE DEI VALORI
        y_range = max(acc_values) - min(acc_values)
        threshold = max(acc_values) - (y_range * 0.15)  # 15% dal top
        
        for r, acc in zip(rounds, acc_values):
            if acc > threshold:  # Valore vicino al top
                # Metti SOTTO il punto
                ax.text(r, acc - 0.005, f'{acc:.3f}', 
                        ha='center', va='top', fontsize=14, fontweight='bold')
            else:
                # Metti SOPRA il punto
                ax.text(r, acc + 0.005, f'{acc:.3f}', 
                        ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax.set_xlabel('Rounds', fontweight='bold', fontsize=18)
        ax.set_ylabel('Accuracy', fontweight='bold', fontsize=18)
        ax.set_title('Global Model Accuracy Evolution (Classic FedAvg)', 
                    fontweight='bold', fontsize=22, pad=15)
        ax.set_xticks(rounds)
        ax.set_xticklabels([f"R{r}" for r in rounds], fontsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.2)
        ax.legend(fontsize=15, frameon=True, shadow=True)
        
        # ✅ MARGINE SUPERIORE AUTOMATICO
        ax.set_ylim([min(acc_values) - 0.01, 1.01])  # Lascia spazio per label
        
        plt.tight_layout()
        fig.savefig(self.output_dir / "global_accuracy_evolution.png", 
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info("  ✅ Grafico Accuracy Globale salvato")
        
    def _plot_global_f1_evolution(self):
        """Plot dell'evoluzione del F1 del modello globale (FedAvg Classico)."""
        fig, ax = plt.subplots(figsize=(16, 9))
        
        rounds = []
        f1_values = []
                
        for r in self.rounds_history:
            if 'cluster_test_metrics' in r and 0 in r['cluster_test_metrics']:
                rounds.append(r['round_num'])
                f1_values.append(r['cluster_test_metrics'][0]['f1'])
        
        logger.debug(f"🔍 [F1 Plot] Rounds: {rounds}, F1 values: {f1_values}")
            
        if not rounds:
            logger.warning("⚠️ Nessun dato F1 globale disponibile")
            plt.close(fig)
            return
        
        # Plot linea
        ax.plot(rounds, f1_values, 
                marker='o', color='#1D3557', linewidth=3.5, 
                markersize=12, label='Global Model F1', 
                markerfacecolor='white', markeredgewidth=2.5)
        
        # ✅ POSIZIONAMENTO INTELLIGENTE DEI VALORI
        y_range = max(f1_values) - min(f1_values)
        threshold = max(f1_values) - (y_range * 0.15)  # 15% dal top
        
        for r, f1 in zip(rounds, f1_values):
            if f1 > threshold:  # Valore vicino al top
                # Metti SOTTO il punto
                ax.text(r, f1 - 0.005, f'{f1:.3f}', 
                        ha='center', va='top', fontsize=14, fontweight='bold')
            else:
                # Metti SOPRA il punto
                ax.text(r, f1 + 0.005, f'{f1:.3f}', 
                        ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax.set_xlabel('Rounds', fontweight='bold', fontsize=18)
        ax.set_ylabel('F1-Score', fontweight='bold', fontsize=18)
        ax.set_title('Global Model F1 Evolution (Classic FedAvg)', 
                    fontweight='bold', fontsize=22, pad=15)
        ax.set_xticks(rounds)
        ax.set_xticklabels([f"R{r}" for r in rounds], fontsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.2)
        ax.legend(fontsize=15, frameon=True, shadow=True)
        
        # ✅ MARGINE SUPERIORE AUTOMATICO
        ax.set_ylim([min(f1_values) - 0.01, 1.01])  # Lascia spazio per label
        
        plt.tight_layout()
        fig.savefig(self.output_dir / "global_f1_evolution.png", 
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info("  ✅ Grafico F1 Globale salvato")
    
    def _plot_std_loss_evolution(self):
        """Mostra std_loss tra client per round."""
        fig, ax = plt.subplots(figsize=(16, 9))
        
        rounds = []
        std_losses = []
        
        for r in self.rounds_history:
            if 'round_metrics' in r and 'std_loss' in r['round_metrics']:
                rounds.append(r['round_num'])
                std_losses.append(r['round_metrics']['std_loss'])
        
        logger.debug(f"🔍 [STD Loss Plot] Rounds: {rounds}, STD: {std_losses}")
        
        if not rounds:
            logger.warning("⚠️ Nessun dato std_loss disponibile")
            plt.close(fig)
            return
        
        # Plot linea
        ax.plot(rounds, std_losses, 
                marker='D', color='#F4442E', linewidth=3.5, 
                markersize=12, label='STD Loss', 
                markerfacecolor='white', markeredgewidth=2.5)
        
        ax.fill_between(rounds, 0, std_losses, alpha=0.2, color='#F4442E')
        
        # ✅ POSIZIONAMENTO INTELLIGENTE (verso il basso vicino a 0)
        y_max = max(std_losses)
        threshold_low = y_max * 0.15  # 15% dal bottom
        
        for r, std in zip(rounds, std_losses):
            if std < threshold_low:  # Valore vicino a 0
                # Metti SOPRA il punto
                ax.text(r, std + y_max*0.02, f'{std:.4f}', 
                        ha='center', va='bottom', fontsize=14, fontweight='bold')
            else:
                # Metti SOPRA normalmente
                ax.text(r, std + y_max*0.02, f'{std:.4f}', 
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
        
        # ✅ MARGINE SUPERIORE AUTOMATICO
        ax.set_ylim([0, y_max * 1.15])  # +15% di spazio per label
        
        plt.tight_layout()
        fig.savefig(self.output_dir / "std_loss_evolution.png", 
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info("  ✅ Grafico STD Loss salvato")