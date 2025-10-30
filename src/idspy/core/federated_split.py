import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from src.idspy.events.handlers.logging import Logger
from src.idspy.core.pipeline import (
    FitAwareObservablePipeline,
)
from src.idspy.events.bus import EventBus

from collections import Counter
from src.idspy.data.tab_accessor import register_dataframe_accessor
from src.idspy.core.step import Step
from src.idspy.core.state import State
from sklearn.model_selection import train_test_split

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
Modulo di suddivisione del dataset di apprendimento federato
Fornisce classi Step per suddividere un dataset centralizzato in dataset federati
in base agli indirizzi IP host.
"""

# ============================================================================
# STEP 1: IDENTIFICO I TOP HOSTS
# ============================================================================

class IdentifyTopHosts(Step):
    """
    Identifica gli N indirizzi IP più frequenti nel set di dati.

    Conta sia le occorrenze di origine che di destinazione per determinare gli host
    con il volume di traffico più elevato.
    """
    
    def __init__(
        self,
        num_hosts: int = 10,
        src_ip_col: str = 'IPV4_SRC_ADDR',
        dst_ip_col: str = 'IPV4_DST_ADDR',
        in_scope: str = "data",
        out_scope: str = "federated",
        name: Optional[str] = None,
    ):
        self.num_hosts = num_hosts
        self.src_ip_col = src_ip_col
        self.dst_ip_col = dst_ip_col
        
        super().__init__(
            name=name or "identify_top_hosts",
            in_scope=in_scope,
            out_scope=out_scope,
        )
    
    @Step.requires(root=pd.DataFrame)
    @Step.provides(top_ips=list, ip_statistics=pd.DataFrame)
    def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:
        logger.info(f"Identifico i top {self.num_hosts} hosts dal traffico di volume più elevato...")
        
        # Count occurrences as source and destination
        src_counts = Counter(root[self.src_ip_col])
        dst_counts = Counter(root[self.dst_ip_col])
        
        # Combine counts (total traffic per IP)
        total_counts = Counter()
        for ip in set(list(src_counts.keys()) + list(dst_counts.keys())):
            total_counts[ip] = src_counts.get(ip, 0) + dst_counts.get(ip, 0)
        
        # Get top N IPs
        top_ips = [ip for ip, count in total_counts.most_common(self.num_hosts)]
        
        # Create statistics DataFrame
        stats_data = []
        #label_col = 'Label'  # Assuming this exists
        attack_col = 'Attack'
        for ip in top_ips:
            bidirectional_data = root[
                (root[self.src_ip_col] == ip) | 
                (root[self.dst_ip_col] == ip)
            ]
            
            #label_counts = bidirectional_data[label_col].value_counts()
            label_counts = bidirectional_data[attack_col].value_counts()
            benign_count = label_counts.get('Benign', 0)
            #attack_count = sum(v for k, v in label_counts.items() if k != 'Benign')
            attack_count = len(bidirectional_data) - benign_count

            stats_data.append({
                'ip_address': ip,
                'total_flows': len(bidirectional_data),
                'src_flows': src_counts.get(ip, 0),
                'dst_flows': dst_counts.get(ip, 0),
                'benign_flows': benign_count,
                'attack_flows': attack_count,
                'attack_percentage': (attack_count / len(bidirectional_data) * 100) if len(bidirectional_data) > 0 else 0
            })
        
        stats_df = pd.DataFrame(stats_data)
        logger.info(f"Top {self.num_hosts} hosts identificati")
        logger.info(f"\n{stats_df.to_string()}")
        
        return {"top_ips": top_ips, "ip_statistics": stats_df}


# ============================================================================
# STEP 2: SPLIT PER HOSTS (BIDIREZIONALE)
# ============================================================================

class SplitByHosts(Step):
    """
    Suddividere il dataset in dataset federati, uno per IP host.

    Ogni dataset contiene traffico BIDIREZIONALE:
    - Flussi in cui l'host è ORIGINE (in uscita)
    - Flussi in cui l'host è DESTINAZIONE (in entrata)
    """
    
    def __init__(
        self,
        min_samples_per_host: int = 100,
        src_ip_col: str = 'IPV4_SRC_ADDR',
        dst_ip_col: str = 'IPV4_DST_ADDR',
        in_scope: str = "data",
        out_scope: str = "federated",
        name: Optional[str] = None,
    ):
        self.min_samples_per_host = min_samples_per_host
        self.src_ip_col = src_ip_col
        self.dst_ip_col = dst_ip_col
        
        super().__init__(
            name=name or "split_by_hosts",
            in_scope=in_scope,
            out_scope=out_scope,
        )
    
    @Step.requires(root=pd.DataFrame)
    @Step.provides(federated_datasets=dict)
    def run(
        self, 
        state: State, 
        root: pd.DataFrame, 
    ) -> Optional[Dict[str, Any]]:
        top_ips = state.get("federated.top_ips", list)
        logger.info(f"Splittando il dataset in {len(top_ips)} dataset federati ...")

        federated_datasets = {}
        
        for ip in top_ips:
            # Extract BIDIRECTIONAL traffic
            host_data = root[
                (root[self.src_ip_col] == ip) |
                (root[self.dst_ip_col] == ip)
            ].copy()
            
            # Check minimum samples
            if len(host_data) < self.min_samples_per_host:
                logger.warning(
                    f"Host {ip} ha solo {len(host_data)} campioni "
                    f"(min: {self.min_samples_per_host}). Skipping..."
                )
                continue
            
            federated_datasets[ip] = host_data
            
            label_dist = host_data['Attack'].value_counts()
            logger.info(
                f"Host {ip}: {len(host_data)} campioni | "
                f"Labels: {dict(label_dist)}"
            )
        
        logger.info(f"Creati {len(federated_datasets)} dataset federati")
        
        return {"federated_datasets": federated_datasets}


# ============================================================================
# STEP 3: CREO TRAIN/VAL/TEST SPLITS PER HOST
# ============================================================================

class FederatedSplits(Step):
    """
    Crea suddivisioni train/val/test per ogni dataset federato. 
    Simile alla suddivisione stratificata, ma applicata in modo indipendente a ciascun host.
    """
    
    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        stratify_column: str = 'Label',
        in_scope: str = "federated",
        out_scope: str = "federated",
        name: Optional[str] = None,
    ):
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.stratify_column = stratify_column
        
        super().__init__(
            name=name or "federated_splits",
            in_scope=in_scope,
            out_scope=out_scope,
        )
    
    @Step.requires(federated_datasets=dict)
    @Step.provides(federated_splits=dict)
    def run(
        self, 
        state: State, 
        federated_datasets: Dict[str, pd.DataFrame],  
    ) -> Optional[Dict[str, Any]]:
        logger.info("Creando train/val/test splits per ogni host...")
        seed = state.get("seed", int) if state.has("seed") else state.get("data.seed", int)
        federated_splits = {}
        
        for ip, data in federated_datasets.items():
            # Stratified split per host
            
            # First split: train vs (val+test)
            train_data, temp_data = train_test_split(
                data,
                train_size=self.train_size,
                stratify=data[self.stratify_column],
                random_state=seed
            )
            
            # Second split: val vs test
            val_ratio = self.val_size / (self.val_size + self.test_size)
            val_data, test_data = train_test_split(
                temp_data,
                train_size=val_ratio,
                stratify=temp_data[self.stratify_column],
                random_state=seed
            )
            
            federated_splits[ip] = {
                'train': train_data.reset_index(drop=True),
                'val': val_data.reset_index(drop=True),
                'test': test_data.reset_index(drop=True)
            }
            
            logger.info(
                f"Host {ip}: train={len(train_data)}, "
                f"val={len(val_data)}, test={len(test_data)}"
            )
        
        return {"federated_splits": federated_splits}

# ============================================================================
# STEP 4: ANALIZZARE LA DISTRIBUZIONE NON-IID 
# ============================================================================
class AnalyzeNonIID(Step):
    """
    Analizza la natura non-IID dei set di dati federati.
    Quantifica le diverse distribuzioni dei dati tra gli host.
    """
    
    def __init__(
        self,
        label_col: str = 'Label',      # Colonna binaria: 0=Benign, 1=Attack
        attack_col: str = 'Attack',      # Colonna con tipo di attacco
        in_scope: str = "federated",
        out_scope: str = "federated",
        name: Optional[str] = None,
    ):
        self.label_col = label_col
        self.attack_col = attack_col
        
        super().__init__(
            name=name or "analyze_non_iid",
            in_scope=in_scope,
            out_scope=out_scope,
        )
    
    @Step.requires(federated_datasets=dict)
    @Step.provides(non_iid_analysis=pd.DataFrame)
    def run(
        self, 
        state: State, 
        federated_datasets: Dict[str, pd.DataFrame]
    ) -> Optional[Dict[str, Any]]:
        logger.info("Analizzando le caratteristiche non-IID...")
        
        analysis_data = []
        
        for ip, data in federated_datasets.items():
            logger.info(f"\n🔍 DEBUG - Host {ip}:")
            #logger.info(f"   Colonne disponibili: {data.columns.tolist()}")
            logger.info(f"   label_col (Attack) - valori unici: {data[self.label_col].unique()}")
            logger.info(f"   attack_col (Label) - valori unici: {data[self.attack_col].unique()[:10]}...")  # Prime 10
            
            # Conta benign/attack usando label_col = 'Attack' (0 = Benign, >0 = Attack)
            label_counts = data[self.label_col].value_counts(normalize=True)
            
            logger.info(f"   label_counts (normalizzati): {dict(label_counts)}")
            
            benign_pct = label_counts.get(0, 0) * 100  # 0 = Benign
            #attack_pct = (1 - label_counts.get(0, 0)) * 100  # tutto != 0 = Attack
            attack_pct = label_counts.get(1, 0) * 100
            
            logger.info(f"   benign_pct: {benign_pct:.2f}%, attack_pct: {attack_pct:.2f}%")
            
            # Attack type analysis usando attack_col = 'Label'
            if self.attack_col in data.columns:
                # Filtra solo le righe di attacco (Attack != 0)
                attack_data = data[data[self.label_col] != 0]
                
                logger.info(f"   Totale righe di attacco (Attack != 0): {len(attack_data)}")
                
                if len(attack_data) > 0:
                    # Conta i tipi di attacco dalla colonna 'Label'
                    attack_counts = attack_data[self.attack_col].value_counts()
                    
                    logger.info(f"   attack_counts (dalla colonna Label):")
                    logger.info(f"{attack_counts}")
                    
                    # Trova l'attacco più comune (escludendo 'Benign' se presente)
                    attack_counts_no_benign = attack_counts[attack_counts.index != 'Benign']
                    
                    logger.info(f"   attack_counts_no_benign (senza Benign):")
                    logger.info(f"{attack_counts_no_benign}")
                    
                    if len(attack_counts_no_benign) > 0:
                        top_attack = attack_counts_no_benign.index[0]
                        top_attack_count = attack_counts_no_benign.iloc[0]
                        
                        logger.info(f"   top_attack: {top_attack}")
                        logger.info(f"   top_attack_count: {top_attack_count}")
                        logger.info(f"   len(attack_data): {len(attack_data)}")
                        
                        # Percentuale SUL TOTALE DEGLI ATTACCHI
                        top_attack_pct = (top_attack_count / len(attack_data)) * 100
                        
                        # Percentuale sul dataset totale
                        top_attack_pct_total = (top_attack_count / len(data)) * 100
                        
                        logger.info(f"   top_attack_pct (tra attacchi): {top_attack_pct:.2f}%")
                        logger.info(f"   top_attack_pct_total (su tutto): {top_attack_pct_total:.2f}%")
                    else:
                        top_attack = None
                        top_attack_pct = 0
                        top_attack_pct_total = 0
                       # logger.info("   Nessun attacco trovato dopo rimozione Benign")
                else:
                    top_attack = None
                    top_attack_pct = 0
                    top_attack_pct_total = 0
                    #logger.info("   Nessuna riga di attacco trovata")
            else:
                #logger.warning(f"   Colonna {self.attack_col} non trovata!")
                top_attack = 'N/A'
                top_attack_pct = 0
                top_attack_pct_total = 0
            
            analysis_data.append({
                'ip_address': ip,
                'total_samples': len(data),
                'benign_percentage': benign_pct,
                'attack_percentage': attack_pct,
                'top_attack_type': top_attack,
                'top_attack_percentage': top_attack_pct,
                'top_attack_pct_total': top_attack_pct_total
            })
        
        analysis_df = pd.DataFrame(analysis_data)
        
        logger.info("\n" + "="*70)
        logger.info("Analisi dei Non-IID:")
        logger.info(f"\n{analysis_df.to_string()}")
        
        # Calcola varianza in attack percentages (misura di non-IID)
        variance = analysis_df['attack_percentage'].var()
        logger.info(f"\nVarianza % di attacco: {variance:.2f} (più alta = più non-IID)")
        
        return {"non_iid_analysis": analysis_df}


# ============================================================================
# STEP 5: APPLICAZIONE DELLA FIT_AWARE_PIPELINE PER OGNI HOST
# ============================================================================
class ApplyFitAwareToFederatedSplits(Step):
    """
    Applica fit-aware transformations a ogni SPLIT di ogni host.
    Fitta su TRAIN, applica su train/val/test.
    """
    
    def __init__(
        self,
        fit_aware_steps: List[Step],
        in_scope: str = "federated",
        out_scope: str = "federated",
        name: Optional[str] = None,
    ):
        self.fit_aware_steps = fit_aware_steps
        
        super().__init__(
            name=name or "apply_fit_aware_to_federated_splits",
            in_scope=in_scope,
            out_scope=out_scope,
        )
    
    @Step.requires(federated_splits=dict, seed=int)
    @Step.provides(federated_splits=dict)
    def run(
        self, 
        state: State, 
        federated_splits: Dict[str, Dict[str, pd.DataFrame]],
        seed: int
    ) -> Optional[Dict[str, Any]]:
        logger.info("Applicando trasformazioni fit-aware su ogni split federato...")
        
        transformed_splits = {}
        
        for ip, splits in federated_splits.items():
            logger.info(f"   🔧 Processing host {ip}...")
            
            # Combina train/val/test in un unico DataFrame mantenendo gli indici originali
            train_df = splits['train'].copy()
            val_df = splits['val'].copy()
            test_df = splits['test'].copy()
            
            # Crea indici univoci per ogni partizione
            train_df.index = pd.RangeIndex(0, len(train_df), name='index')
            val_df.index = pd.RangeIndex(len(train_df), len(train_df) + len(val_df), name='index')
            test_df.index = pd.RangeIndex(len(train_df) + len(val_df), 
                                         len(train_df) + len(val_df) + len(test_df), 
                                         name='index')
            
            # Combina tutti i dati
            combined_df = pd.concat([train_df, val_df, test_df], ignore_index=False)
            
            # Suddivisione in partizioni 
            partition_mapping = {
                'train': train_df.index,
                'val': val_df.index,
                'test': test_df.index
            }
            combined_df.tab.set_partitions_from_labels(partition_mapping)
            
            # Crea state temporaneo
            temp_state = State({
                "data.root": combined_df,
                "data.seed": seed
            })
            
            # Crea pipeline fit-aware INDIPENDENTE per questo host
            host_pipeline = FitAwareObservablePipeline(
                steps=self.fit_aware_steps,
                bus=EventBus(),
                name=f"fit_aware_host_{ip.replace('.', '_')}"
            )
            
            # Fit su TRAIN, applica su train/val/test
            try:
                host_pipeline.run(temp_state)
            except Exception as e:
                logger.error(f"Errore durante trasformazione host {ip}: {e}")
                raise
            
            # Recupera i dati trasformati
            transformed_df = temp_state.get("data.root", pd.DataFrame)
            
            # Ri-separa usando le partizioni
            transformed_splits[ip] = {
                'train': transformed_df.tab.train.reset_index(drop=True),
                'val': transformed_df.tab.val.reset_index(drop=True),
                'test': transformed_df.tab.test.reset_index(drop=True)
            }
            
            logger.info(f"      ✅ Host {ip}: train={len(transformed_splits[ip]['train'])}, "
                       f"val={len(transformed_splits[ip]['val'])}, test={len(transformed_splits[ip]['test'])}")
        
        logger.info("✅ Tutti i dataset federati trasformati con successo!")
        
        return {"federated_splits": transformed_splits}

# ============================================================================
# STEP 6: CREO UN TEST SET AGGREGATO GLOBALE
# ============================================================================

class AggregatedTestSet(Step):
    """
    Crea un singolo set di test aggregato dai set di test di tutti gli host.
    Utilizzato per la valutazione finale per confrontare i modelli locali con quelli globali.
    """
    
    def __init__(
        self,
        in_scope: str = "federated",
        out_scope: str = "federated",
        name: Optional[str] = None,
    ):
        super().__init__(
            name=name or "aggregated_test_set",
            in_scope=in_scope,
            out_scope=out_scope,
        )
    
    @Step.requires(federated_splits=dict)
    @Step.provides(aggregated_test=pd.DataFrame)
    def run(
        self, 
        state: State, 
        federated_splits: Dict[str, Dict[str, pd.DataFrame]]
    ) -> Optional[Dict[str, Any]]:
        seed = state.get("seed", int) if state.has("seed") else state.get("data.seed", int)
        
        logger.info("Creando un test set aggregato ricavato da tutti gli hosts...")
        
        # ✅ Estrai i test set e pulisci gli attributi
        test_sets = []
        for splits in federated_splits.values():
            test_df = splits['test'].copy()
            
            # 🔧 Rimuovi attributi custom che causano problemi nel concat
            test_df.attrs = {}
            
            test_sets.append(test_df)
        
        aggregated_test = pd.concat(test_sets, ignore_index=True)
        
        # Shuffle
        aggregated_test = aggregated_test.sample(
            frac=1, 
            random_state=seed
        ).reset_index(drop=True)
        
        logger.info(f"Test set aggregato: {len(aggregated_test)} campioni")
        label_dist = aggregated_test['Attack'].value_counts()
        logger.info(f"Distribuzione delle etichette: {dict(label_dist)}")
        
        return {"aggregated_test": aggregated_test}

