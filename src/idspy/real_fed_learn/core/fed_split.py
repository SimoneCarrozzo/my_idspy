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

logger = logging.getLogger(__name__)

"""
Modulo di suddivisione del dataset di apprendimento federato
Fornisce classi Step per suddividere un dataset centralizzato in dataset federati
in base agli indirizzi IP host.
"""

# ============================================================================
# STEP 1: IDENTIFICO I TOP HOSTS
# ============================================================================

class IdentifyTopHosts_Orig(Step):
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
        attack_col: str = 'Attack',  # 🆕 Parametro che andava aggiunto
        in_scope: str = "data",
        out_scope: str = "federated",
        name: Optional[str] = None,
    ):
        self.num_hosts = num_hosts
        self.src_ip_col = src_ip_col
        self.dst_ip_col = dst_ip_col
        
        self.attack_col = attack_col # 🆕 di conseguenza anche questo va aggiunto per il conteggio delle etichette
        
        super().__init__(
            name=name or "identify_top_hosts",
            in_scope=in_scope,
            out_scope=out_scope,
        )
    
    @Step.requires(root=pd.DataFrame)
    @Step.provides(top_ips=list, ip_statistics=pd.DataFrame)
    def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:
        
        logger.info(f"Identifico i top {self.num_hosts} hosts dal traffico di volume più elevato...")
        
        # 🆕 VALIDAZIONE robusta che verifica che le colonne esistano
        required_cols = [self.src_ip_col, self.dst_ip_col, self.attack_col]
        missing_cols = [col for col in required_cols if col not in root.columns]
        if missing_cols:
            raise ValueError(f"Colonne mancanti nel dataset: {missing_cols}")
        
        # Count occurrences as source and destination
        src_counts = Counter(root[self.src_ip_col])
        dst_counts = Counter(root[self.dst_ip_col])
        
        # Combine counts (total traffic per IP)
        total_counts = Counter()
        for ip in set(list(src_counts.keys()) + list(dst_counts.keys())):
            total_counts[ip] = src_counts.get(ip, 0) + dst_counts.get(ip, 0)
            
        # 🆕 VALIDAZIONE robusta che verifica che ci siano abbastanza IP
        num_unique_ips = len(total_counts)
        if num_unique_ips < self.num_hosts:
            logger.warning(
                f"⚠️ Richiesti {self.num_hosts} hosts ma trovati solo {num_unique_ips} IP unici. "
                f"Usando tutti gli {num_unique_ips} disponibili."
            )
            self.num_hosts = num_unique_ips
        
        # Get top N IPs
        top_ips = [ip for ip, count in total_counts.most_common(self.num_hosts)]
        
        # Create statistics DataFrame
        stats_data = []
        
        for ip in top_ips:
            bidirectional_data = root[
                (root[self.src_ip_col] == ip) | 
                (root[self.dst_ip_col] == ip)
            ]
            
            # 🆕 MIGLIORATA la gestione dei conteggi che ora è più robusta
            attack_counts = bidirectional_data[self.attack_col].value_counts()
            benign_count = attack_counts.get('Benign', 0)
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
    
class IdentifyTopHosts_0(Step):
    """
    LA FUNZIONE ORIGINALE SVOLGEVA QUESTO: Identifica gli N indirizzi IP più frequenti nel set di dati.
    Conta sia le occorrenze di origine che di destinazione per determinare gli host
    con il volume di traffico più elevato.

    LA FUNZIONE CORRENTE INVECE: SELEZIONA IP CON PIù ATTACCHI, QUINDI PER VOLUME DI PRENDENDO GLI ATTACCANTI VERI.
    """
    
    def __init__(
        self,
        num_hosts: int = 10,
        src_ip_col: str = 'IPV4_SRC_ADDR',
        dst_ip_col: str = 'IPV4_DST_ADDR',
        attack_col: str = 'Attack',  # 🆕 Parametro che andava aggiunto
        in_scope: str = "data",
        out_scope: str = "federated",
        name: Optional[str] = None,
    ):
        self.num_hosts = num_hosts
        self.src_ip_col = src_ip_col
        self.dst_ip_col = dst_ip_col
        
        self.attack_col = attack_col # 🆕 di conseguenza anche questo va aggiunto per il conteggio delle etichette
        
        super().__init__(
            name=name or "identify_top_hosts",
            in_scope=in_scope,
            out_scope=out_scope,
        )
    
    @Step.requires(root=pd.DataFrame)
    @Step.provides(top_ips=list, ip_statistics=pd.DataFrame)
    def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:
        
        logger.info(f"Identifico i top {self.num_hosts} hosts dal traffico di volume più elevato...")
        
        # 🆕 VALIDAZIONE robusta che verifica che le colonne esistano
        required_cols = [self.src_ip_col, self.dst_ip_col, self.attack_col]
        missing_cols = [col for col in required_cols if col not in root.columns]
        if missing_cols:
            raise ValueError(f"Colonne mancanti nel dataset: {missing_cols}")
        
        # Count occurrences as source and destination
        src_counts = Counter(root[self.src_ip_col])
        dst_counts = Counter(root[self.dst_ip_col])
        
        # Combine counts (total traffic per IP)
        total_counts = Counter()
        for ip in set(list(src_counts.keys()) + list(dst_counts.keys())):
            total_counts[ip] = src_counts.get(ip, 0) + dst_counts.get(ip, 0)
            
        # 🆕 VALIDAZIONE robusta che verifica che ci siano abbastanza IP
        num_unique_ips = len(total_counts)
        if num_unique_ips < self.num_hosts:
            logger.warning(
                f"⚠️ Richiesti {self.num_hosts} hosts ma trovati solo {num_unique_ips} IP unici. "
                f"Usando tutti gli {num_unique_ips} disponibili."
            )
            self.num_hosts = num_unique_ips
        
        # Get top N IPs
        top_ips = [ip for ip, count in total_counts.most_common(self.num_hosts)]
        
        # Create statistics DataFrame
        stats_data = []
        
        for ip in top_ips:
            bidirectional_data = root[
                (root[self.src_ip_col] == ip) | 
                (root[self.dst_ip_col] == ip)
            ]
            
            # 🆕 MIGLIORATA la gestione dei conteggi che ora è più robusta
            attack_counts = bidirectional_data[self.attack_col].value_counts()
            # benign_count = attack_counts.get('Benign', 0)
            benign_count = attack_counts.get(0, 0)
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

class IdentifyTopHosts(Step):
    """
    LA FUNZIONE ORIGINALE SVOLGEVA QUESTO: Identifica gli N indirizzi IP più frequenti nel set di dati.
    Conta sia le occorrenze di origine che di destinazione per determinare gli host
    con il volume di traffico più elevato.

    LA FUNZIONE CORRENTE INVECE: SELEZIONA IP CON PIù ATTACCHI, QUINDI PER VOLUME DI ATTACCHI, PRENDENDO GLI ATTACCANTI VERI.
    
    Dunque, se prima la funzione contava tutto il traffico, sia benigno che attacchi per poi ordinare per volume totale, 
    ciò che accadeva era che molto probabilmente andava a selezionare i server più attivi (DNS, web, etc.) che spesso 
    hanno ZERO attacchi ma tanto traffico normale. Esempio dal tuo dataset:
    
        IP 59.166.0.1 aveva 230k flussi → selezionato ✅ --> Ma TUTTI erano benigni → inutile per FL! ❌
    """
    
    def __init__(
        self,
        num_hosts: int = 10,
        src_ip_col: str = 'IPV4_SRC_ADDR',
        dst_ip_col: str = 'IPV4_DST_ADDR',
        attack_col: str = 'Attack',  # 🆕 Parametro che andava aggiunto
        in_scope: str = "data",
        out_scope: str = "federated",
        name: Optional[str] = None,
    ):
        self.num_hosts = num_hosts
        self.src_ip_col = src_ip_col
        self.dst_ip_col = dst_ip_col
        
        self.attack_col = attack_col # 🆕 di conseguenza anche questo va aggiunto per il conteggio delle etichette
        
        super().__init__(
            name=name or "identify_top_hosts",
            in_scope=in_scope,
            out_scope=out_scope,
        )
    
    @Step.requires(root=pd.DataFrame)
    @Step.provides(top_ips=list, ip_statistics=pd.DataFrame)
    def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:
        
        logger.info(f"Identifico i top {self.num_hosts} hosts dal traffico di volume più elevato...")
        
        # Validazione
        required_cols = [self.src_ip_col, self.dst_ip_col, self.attack_col]
        missing_cols = [col for col in required_cols if col not in root.columns]
        if missing_cols:
            raise ValueError(f"Colonne mancanti nel dataset: {missing_cols}")
        
        # Count occurrences
        src_counts = Counter(root[self.src_ip_col])
        dst_counts = Counter(root[self.dst_ip_col])
        
        # ✅ NUOVO: Conta ATTACCHI per IP invece di traffico totale
        attack_counts_per_ip = {}
        benign_counts_per_ip = {}
        total_counts_per_ip = {}
        
        unique_ips = set(list(src_counts.keys()) + list(dst_counts.keys()))
        
        for ip in unique_ips:
            bidirectional_data = root[
                (root[self.src_ip_col] == ip) | 
                (root[self.dst_ip_col] == ip)
            ]
            
            # Conta benign (0) e attack (1)
            label_counts = bidirectional_data[self.attack_col].value_counts()
            # Per ogni IP conta separatamente:
            benign_count = label_counts.get(0, 0)  # Traffico normale
            attack_count = label_counts.get(1, 0)  # Attacchi veri
            
            attack_counts_per_ip[ip] = attack_count
            benign_counts_per_ip[ip] = benign_count
            total_counts_per_ip[ip] = len(bidirectional_data)
        
        # ✅ STRATEGIA: Seleziona IP con PIÙ ATTACCHI, CIOè ORDINA PER NUMERO DI ATTACCHI 
        # (non più traffico totale, che favorisce server normali)
        sorted_by_attacks = sorted(
            attack_counts_per_ip.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Filtra IP con ALMENO qualche attacco
        ips_with_attacks = [ip for ip, count in sorted_by_attacks if count > 0]
        
        # seleziona gli IP con il MAGGIOR NUMERO DI ATTACCHI, quindi prende gli IP 
        # degli attaccanti veri. Se non ci sono abbastanza IP con attacchi, 
        # li completa con quelli benigni.
        if len(ips_with_attacks) < self.num_hosts:
            logger.warning(
                f"⚠️ Trovati solo {len(ips_with_attacks)} IP con attacchi "
                f"(richiesti {self.num_hosts}). Completamento con IP normali..."
            )
            # Completa con IP normali se necessario
            ips_only_benign = [ip for ip, count in sorted_by_attacks if count == 0]
            sorted_by_total = sorted(
                [(ip, total_counts_per_ip[ip]) for ip in ips_only_benign],
                key=lambda x: x[1],
                reverse=True
            )
            top_ips = ips_with_attacks + [ip for ip, _ in sorted_by_total[:self.num_hosts - len(ips_with_attacks)]]
        else:
            top_ips = ips_with_attacks[:self.num_hosts]
        
        # Create statistics DataFrame
        stats_data = []
        for ip in top_ips:
            stats_data.append({
                'ip_address': ip,
                'total_flows': total_counts_per_ip[ip],
                'src_flows': src_counts.get(ip, 0),
                'dst_flows': dst_counts.get(ip, 0),
                'benign_flows': benign_counts_per_ip[ip],
                'attack_flows': attack_counts_per_ip[ip],
                'attack_percentage': (attack_counts_per_ip[ip] / total_counts_per_ip[ip] * 100) if total_counts_per_ip[ip] > 0 else 0
            })
        
        stats_df = pd.DataFrame(stats_data)
        logger.info(f"Top {self.num_hosts} hosts identificati (per volume attacchi)")
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
        attack_col: str = 'Attack',  # 🆕 Parametro che andava aggiunto
        in_scope: str = "data",
        out_scope: str = "federated",
        name: Optional[str] = None,
    ):
        self.min_samples_per_host = min_samples_per_host
        self.src_ip_col = src_ip_col
        self.dst_ip_col = dst_ip_col
        
        self.attack_col = attack_col # 🆕 di conseguenza anche questo va aggiunto per il conteggio delle etichette
        
        super().__init__(
            name=name or "split_by_hosts",
            in_scope=in_scope,
            out_scope=out_scope,
        )
    
    @Step.requires(root=pd.DataFrame)
    @Step.provides(federated_datasets=dict, skipped_hosts=list) #Aggiungo gli host skippati in modo da avere un debugging log migliore 
    def run(
        self, 
        state: State, 
        root: pd.DataFrame, 
    ) -> Optional[Dict[str, Any]]:
        
        # 🆕 VALIDAZIONE robusta che verifica che top_ips esista
        if not state.has("federated.top_ips"):
            raise RuntimeError(
                "❌ 'federated.top_ips' non trovato nello state! "
                "Assicurati di eseguire IdentifyTopHosts prima di questo step."
            )
        
        top_ips = state.get("federated.top_ips", list)
        logger.info(f"Splittando il dataset in {len(top_ips)} dataset federati ...")

        federated_datasets = {}
        
        skipped_hosts = []  # 🆕 Tracciamento host skippati

        
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
                skipped_hosts.append(ip)  # 🆕 Aggiungo l'host alla lista degli skippati
                continue
            
            federated_datasets[ip] = host_data
            
            ## label_dist = host_data['Attack'].value_counts() --> correggo con la versione parametrizzata
            label_dist = host_data[self.attack_col].value_counts()
            logger.info(
                f"Host {ip}: {len(host_data)} campioni | "
                f"Labels: {dict(label_dist)}"
            )
        
        logger.info(f"Creati {len(federated_datasets)} dataset federati")
        
        # Log degli host skippati
        if skipped_hosts:
            logger.warning(f"⚠️ Host skippati ({len(skipped_hosts)}): {skipped_hosts}")
        
        
        return {"federated_datasets": federated_datasets, "skipped_hosts": skipped_hosts} # Ritorno anche gli host skippati


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
        train_size: float = 0.8,
        val_size: float = 0.10,
        test_size: float = 0.10,
        # stratify_column: str = 'Label', cambiato in Attack per coerenza con il resto
        stratify_column: str = 'Attack',
        min_samples_per_class: int = 2,  # 🆕 Parametro aggiunto che mettere un limite inferiore al numero di campioni per classe
        in_scope: str = "federated",
        out_scope: str = "federated",
        name: Optional[str] = None,
    ):
        
        # 🆕 VALIDAZIONE robusta che verifica che le percentuali sommino a 1
        total = train_size + val_size + test_size
        if not np.isclose(total, 1.0):
            raise ValueError(
                f"❌ Le percentuali devono sommare a 1.0, ma sommano a {total}"
            )
        
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.stratify_column = stratify_column
        
        self.min_samples_per_class = min_samples_per_class  # di conseguenza anche questo va aggiunto per il numero minimo di campioni per classe
        
        super().__init__(
            name=name or "federated_splits",
            in_scope=in_scope,
            out_scope=out_scope,
        )
    
    @Step.requires(federated_datasets=dict)
    @Step.provides(federated_splits=dict, split_warnings=dict)  # 🆕 Aggiungo warnings per eventuali avvisi di split così da tracciare i problemi per i vari host
    def run(
        self, 
        state: State, 
        federated_datasets: Dict[str, pd.DataFrame],  
    ) -> Optional[Dict[str, Any]]:
        
        logger.info("Creando train/val/test splits per ogni host...")
        
        # Recupero seed in modo più robusto
        if state.has("seed"):
            seed = state.get("seed", int)
        elif state.has("data.seed"):
            seed = state.get("data.seed", int)
        else:
            seed = 42  # Default fallback
            logger.warning(f"⚠️ Seed non trovato nello state, uso default: {seed}")
        
        federated_splits = {}
        
        split_warnings = {}  # Traccia problemi per host specifici
            
        #Eseguo un controllo più robusto per ogni host
        for ip, data in federated_datasets.items():
            try:
                # 🆕 VALIDAZIONE robusta che verifica che ci siano abbastanza campioni per classe
                class_counts = data[self.stratify_column].value_counts()
                min_count = class_counts.min()
                
                if min_count < self.min_samples_per_class:
                    warning_msg = (
                        f"⚠️ Host {ip}: classe con solo {min_count} campioni "
                        f"(min richiesto: {self.min_samples_per_class}). "
                        f"Split NON stratificato per questo host."
                    )
                    logger.warning(warning_msg)
                    split_warnings[ip] = warning_msg
                    
                    # Split NON stratificato
                    train_data, temp_data = train_test_split(
                        data,
                        train_size=self.train_size,
                        random_state=seed
                    )
                    val_ratio = self.val_size / (self.val_size + self.test_size)
                    val_data, test_data = train_test_split(
                        temp_data,
                        train_size=val_ratio,
                        random_state=seed
                    )
                else:
                    # Split stratificato normale
                    train_data, temp_data = train_test_split(
                        data,
                        train_size=self.train_size,
                        stratify=data[self.stratify_column],
                        random_state=seed
                    )
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
                    f"✅ Host {ip}: train={len(train_data)}, "
                    f"val={len(val_data)}, test={len(test_data)}"
                )
                
            except Exception as e:
                error_msg = f"❌ Errore nello split dell'host {ip}: {str(e)}"
                logger.error(error_msg)
                split_warnings[ip] = error_msg
                # Non aggiungo questo host agli splits
                continue
        
        if split_warnings:
            logger.warning(f"⚠️ Problemi riscontrati in {len(split_warnings)} host")
        
        return {"federated_splits": federated_splits, "split_warnings": split_warnings}  # Ritorno anche i warnings

# ============================================================================
# STEP 4: ANALIZZARE LA DISTRIBUZIONE NON-IID 
# ============================================================================
class AnalyzeNonIID(Step):
    def __init__(
        self,
        label_col: str = 'Attack',  # La colonna binaria (0/1) usata per contare quanti attacchi ci sono
        benign_label: str = 'Benign', 
        attack_type_col: str = 'Label',  # La colonna originale target
        debug: bool = True,
        in_scope: str = "federated",
        out_scope: str = "federated",
        name: Optional[str] = None,
    ):
        self.label_col = label_col
        self.benign_label = benign_label
        self.attack_type_col = attack_type_col
        self.debug = debug
        
        super().__init__(
            name=name or "analyze_non_iid",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    @Step.requires(federated_datasets=dict)
    @Step.provides(non_iid_analysis=pd.DataFrame, non_iid_metrics=dict)
    def run(
        self, 
        state: State, 
        federated_datasets: Dict[str, pd.DataFrame]
    ) -> Optional[Dict[str, Any]]:
        logger.info("Analizzando le caratteristiche non-IID (con recupero nomi originali)...")
        
        # Tentativo di recuperare il mapping (serve solo se non troviamo la colonna original)
        cat_mapping = state.get("data.cat_mapping", dict) if state.has("data.cat_mapping") else {}
        
        analysis_data = []
        
        for ip, data in federated_datasets.items():
            
            # 1. Calcolo % Benign vs Attack (Basato sulla colonna binaria)
            # label_col qui dovrebbe essere quella binaria (0/1) o mappata
            label_counts = data[self.label_col].value_counts(normalize=True)
            
            # Gestione sicura 0/1
            benign_pct = label_counts.get(0, 0.0) * 100
            attack_pct = label_counts.get(1, 0.0) * 100
            
            # 2. Analisi Tipo di Attacco
            """ top_attack_name = "NO ATTACK"
            top_attack_pct = 0.0
            dominance = 0.0 """
            # Inizializza SEMPRE per ogni host
            top_attack_name = "NO ATTACK"
            top_attack_pct = 0.0
            snd_attack_name = "NO ATTACK"
            snd_attack_pct = 0.0
            trd_attack_name = "NO ATTACK"
            trd_attack_pct = 0.0
            frt_attack_name = "NO ATTACK"
            frt_attack_pct = 0.0
            fifth_attack_name = "NO ATTACK"
            fifth_attack_pct = 0.0
            dominance = 0.0
            # Filtriamo solo le righe che sono ATTACCHI
            # Assumiamo che 1 = Attack nella colonna label_col
            attack_mask = data[self.label_col] == 1
            attack_data = data[attack_mask]
            
            #logger.debug(f"Colonne disponibili in attack_data: {attack_data.columns.tolist()}")
            
            if len(attack_data) > 0:
                # 🛠️ STRATEGIA DI SELEZIONE FORZATA
                # Cerchiamo prioritariamente la colonna "original_..." perché contiene le STRINGHE
                
                target_col_name = None
                
                # 1. Forza la ricerca del backup testuale (quello creato da LabelMapGlobal)
                if "original_Attack" in attack_data.columns:
                    target_col_name = "original_Attack"
                elif "original_Label" in attack_data.columns:
                    target_col_name = "original_Label"
                elif f"original_{self.attack_type_col}" in attack_data.columns:
                    target_col_name = f"original_{self.attack_type_col}"
                else:
                    # 2. Se non c'è il backup, usiamo quella definita, ma sappiamo che sarà un codice
                    target_col_name = self.attack_type_col

                if self.debug and ip == list(federated_datasets.keys())[0]:
                    logger.info(f"🎯 DECISIONE: Uso '{target_col_name}' (Trovate: {[c for c in attack_data.columns if 'original' in c]})")

                # Conteggio valori
                type_counts = attack_data[target_col_name].value_counts()
                
                if len(type_counts) > 0:
                    # 🆕 HELPER FUNCTION per convertire codice → nome
                    def get_attack_name(value):
                        """Converte un valore (int/string) nel nome dell'attacco."""
                        if isinstance(value, (int, float, np.integer)):
                            # È un codice numerico, cerchiamo il mapping
                            mapping = None
                            if self.attack_type_col in cat_mapping:
                                mapping = cat_mapping[self.attack_type_col]
                            elif 'Label' in cat_mapping:
                                mapping = cat_mapping['Label']
                            elif 'Attack' in cat_mapping:
                                mapping = cat_mapping['Attack']
                            
                            if mapping is not None and hasattr(mapping, 'categories'):
                                try:
                                    cats = mapping.categories
                                    idx = int(value)
                                    # Spesso i codici sono 1-based se 0 è riservato
                                    if 1 <= idx <= len(cats):
                                        return str(cats[idx-1])
                                    elif 0 <= idx < len(cats):
                                        return str(cats[idx])
                                    else:
                                        return f"Code_{value}"
                                except:
                                    return f"Code_{value}"
                            else:
                                return f"Code_{value}"
                        else:
                            # È già una stringa
                            return str(value)
                    
                    # ─────────────────────────────────────────────────────────────
                    # 🥇 TOP 1 (quello che già avevi)
                    # ─────────────────────────────────────────────────────────────
                    top_val = type_counts.index[0]
                    top_count = type_counts.iloc[0]
                    top_attack_pct = (top_count / len(attack_data)) * 100
                    dominance = top_attack_pct / 100
                    top_attack_name = get_attack_name(top_val)
                    
                    # ─────────────────────────────────────────────────────────────
                    # 🥈 TOP 2 (NUOVO)
                    # ─────────────────────────────────────────────────────────────
                    snd_attack_name = "NO ATTACK"
                    snd_attack_pct = 0.0
                    
                    if len(type_counts) >= 2:
                        snd_val = type_counts.index[1]
                        snd_count = type_counts.iloc[1]
                        snd_attack_pct = (snd_count / len(attack_data)) * 100
                        snd_attack_name = get_attack_name(snd_val)
                    
                    # ─────────────────────────────────────────────────────────────
                    # 🥉 TOP 3 (NUOVO)
                    # ─────────────────────────────────────────────────────────────
                    trd_attack_name = "NO ATTACK"
                    trd_attack_pct = 0.0
                    
                    if len(type_counts) >= 3:
                        trd_val = type_counts.index[2]
                        trd_count = type_counts.iloc[2]
                        trd_attack_pct = (trd_count / len(attack_data)) * 100
                        trd_attack_name = get_attack_name(trd_val)
                    # ─────────────────────────────────────────────────────────────
                    # 🥉 TOP 4 (NUOVO)
                    # ─────────────────────────────────────────────────────────────
                    frt_attack_name = "NO ATTACK"
                    frt_attack_pct = 0.0
                    
                    if len(type_counts) >= 4:
                        frt_val = type_counts.index[3]
                        frt_count = type_counts.iloc[3]
                        frt_attack_pct = (frt_count / len(attack_data)) * 100
                        frt_attack_name = get_attack_name(frt_val)
                    # ─────────────────────────────────────────────────────────────
                    # 🥉 TOP 5 (NUOVO)
                    # ─────────────────────────────────────────────────────────────
                    fifth_attack_name = "NO ATTACK"
                    fifth_attack_pct = 0.0
                    
                    if len(type_counts) >= 5:
                        fifth_val = type_counts.index[4]
                        fifth_count = type_counts.iloc[4]
                        fifth_attack_pct = (fifth_count / len(attack_data)) * 100
                        fifth_attack_name = get_attack_name(fifth_val)
                        
            
            analysis_data.append({
                'ip_address': ip,
                'total_samples': len(data),
                'benign_percentage': benign_pct,
                'attack_percentage': attack_pct,
                'top_attack_type': top_attack_name,
                'top_attack_percentage': top_attack_pct,
                'snd_attack_type': snd_attack_name,        # 🆕
                'snd_attack_percentage': snd_attack_pct,   # 🆕
                'trd_attack_type': trd_attack_name,        # 🆕
                'trd_attack_percentage': trd_attack_pct,   # 🆕
                'frt_attack_type': frt_attack_name,        # 🆕
                'frt_attack_percentage': frt_attack_pct,   # 🆕
                'fifth_attack_type': fifth_attack_name,        # 🆕
                'fifth_attack_percentage': fifth_attack_pct,   # 🆕
                'attack_dominance': dominance,
            })
        
        analysis_df = pd.DataFrame(analysis_data)
        
        logger.info("\n" + "="*70)
        logger.info("📊 Analisi Non-IID:")
        logger.info(f"\n{analysis_df.to_string()}")
        
        #======MODIFICA============#
        logger.info("\n" + "="*70)
        logger.info("🎯 SPECIALIZZAZIONE AUTOMATICA DEI CLIENT:")
        logger.info("="*70)

        # ovr_cols = ['is_ddos_attack_hoic', 'is_dos_attacks_hulk', 'is_bot', 'is_infilteration', 'is_ddos_attacks_loic_http']
        ovr_cols = ['is_analysis', 'is_backdoor', 'is_dos', 'is_exploits', 'is_fuzzers', 'is_shellcode', 'is_worms']  # Lista di colonne OVR
        for ip, data in federated_datasets.items():
            # attack_counts = {col: data[col].sum() for col in ovr_cols if col in data.columns}
            attack_counts = {col: int(data[col].sum()) for col in ovr_cols if col in data.columns}

            if sum(attack_counts.values()) > 0:
                dominant = max(attack_counts, key=attack_counts.get)
                logger.info(f"   • {ip} → {dominant} ({attack_counts[dominant]} samples)")
            else:
                logger.info(f"   • {ip} → SKIP (solo benigno)")
        #======FINE MODIFICA============#
        
        # Metriche
        top_attacks = analysis_df['top_attack_type'].value_counts().to_dict()
        non_iid_metrics = {
            'attack_percentage_variance': analysis_df['attack_percentage'].var(),
            'attack_percentage_std': analysis_df['attack_percentage'].std(),
            'average_attack_dominance': analysis_df['attack_dominance'].mean(),
            'num_hosts': len(analysis_df),
            'top_attacks_distribution': top_attacks
        }
        
        return {
            "non_iid_analysis": analysis_df,
            "non_iid_metrics": non_iid_metrics
        }
# ============================================================================
# STEP 5: APPLICAZIONE DELLA FIT_AWARE_PIPELINE PER OGNI HOST
# ============================================================================

class ApplyFitAwareToFederatedSplits(Step):
    def __init__(
        self,
        fit_aware_steps: List[Step],
        use_shared_bus: bool = False,  # 🆕 Opzione per riutilizzare bus
        in_scope: str = "federated",
        out_scope: str = "federated",
        name: Optional[str] = None,
    ):
        self.fit_aware_steps = fit_aware_steps
        self.use_shared_bus = use_shared_bus  # 🆕
        
        super().__init__(
            name=name or "apply_fit_aware_to_federated_splits",
            in_scope=in_scope,
            out_scope=out_scope,
        )
    
    @Step.requires(federated_splits=dict)
    @Step.provides(federated_splits=dict, transformation_errors=dict)  # traccia eventuali errori di trasformazione
    def run(
        self, 
        state: State, 
        federated_splits: Dict[str, Dict[str, pd.DataFrame]],
    ) -> Optional[Dict[str, Any]]:
        logger.info("Applicando trasformazioni fit-aware su ogni split federato...")
        
        # Recupero seed robusto
        if state.has("seed"):
            seed = state.get("seed", int)
        elif state.has("data.seed"):
            seed = state.get("data.seed", int)
        else:
            seed = 42
            logger.warning(f"⚠️ Seed non trovato, uso default: {seed}")
        
        transformed_splits = {}
        transformation_errors = {}  # Traccia errori
        
        # EventBus condiviso opzionale
        if self.use_shared_bus and state.has("bus"):
            shared_bus = state.get("bus", EventBus)
        else:
            shared_bus = None
        
        for ip, splits in federated_splits.items():
            try:
                logger.info(f"   🔧 Processing host {ip}...")
                
                # Combina train/val/test
                train_df = splits['train'].copy()
                val_df = splits['val'].copy()
                test_df = splits['test'].copy()
                
                # Indici univoci
                train_df.index = pd.RangeIndex(0, len(train_df), name='index')
                val_df.index = pd.RangeIndex(len(train_df), len(train_df) + len(val_df), name='index')
                test_df.index = pd.RangeIndex(
                    len(train_df) + len(val_df), 
                    len(train_df) + len(val_df) + len(test_df), 
                    name='index'
                )
                
                # Combina
                combined_df = pd.concat([train_df, val_df, test_df], ignore_index=False)
                
                # Partizioni
                partition_mapping = {
                    'train': train_df.index,
                    'val': val_df.index,
                    'test': test_df.index
                }
                combined_df.tab.set_partitions_from_labels(partition_mapping)
                
                # State temporaneo
                temp_state = State({
                    "data.root": combined_df,
                    "data.seed": seed
                })
                
                # Pipeline fit-aware
                host_bus = shared_bus if shared_bus else EventBus()
                host_pipeline = FitAwareObservablePipeline(
                    steps=self.fit_aware_steps,
                    bus=host_bus,
                    name=f"fit_aware_host_{ip.replace('.', '_')}"
                )
                
                # Esegui
                host_pipeline.run(temp_state)
                
                # Recupera dati trasformati
                transformed_df = temp_state.get("data.root", pd.DataFrame)
                
                # Ri-separa
                transformed_splits[ip] = {
                    'train': transformed_df.tab.train.reset_index(drop=True),
                    'val': transformed_df.tab.val.reset_index(drop=True),
                    'test': transformed_df.tab.test.reset_index(drop=True)
                }
                
                logger.info(
                    f"      ✅ Host {ip}: train={len(transformed_splits[ip]['train'])}, "
                    f"val={len(transformed_splits[ip]['val'])}, "
                    f"test={len(transformed_splits[ip]['test'])}"
                )
                
            except Exception as e:
                error_msg = f"❌ Errore durante trasformazione host {ip}: {str(e)}"
                logger.error(error_msg)
                transformation_errors[ip] = str(e)
                # Non aggiungo questo host ai transformed_splits
                continue
        
        if transformation_errors:
            logger.warning(
                f"⚠️ Trasformazione fallita per {len(transformation_errors)} host: "
                f"{list(transformation_errors.keys())}"
            )
        
        logger.info("✅ Tutti i dataset federati trasformati con successo!")
        
        return {
            "federated_splits": transformed_splits,
            "transformation_errors": transformation_errors  # 🆕
        }

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
        attack_col: str = 'Attack',  # 🆕 Colonna di attacchi parametrizzata correttamente
        save_host_labels: bool = True,  # 🆕 Opzione per tracciare origine dei campioni
        in_scope: str = "federated",
        out_scope: str = "federated",
        name: Optional[str] = None,
    ):
        
        self.attack_col = attack_col  # 🆕
        self.save_host_labels = save_host_labels  # 🆕
        
        super().__init__(
            name=name or "aggregated_test_set",
            in_scope=in_scope,
            out_scope=out_scope,
        )
    
    @Step.requires(federated_splits=dict)
    @Step.provides(aggregated_test=pd.DataFrame, test_host_mapping=pd.DataFrame)  # 🆕 Aggiungo mapping host-test per tracciare origine campioni
    def run(
        self, 
        state: State, 
        federated_splits: Dict[str, Dict[str, pd.DataFrame]]
    ) -> Optional[Dict[str, Any]]:

        logger.info("Creando un test set aggregato ricavato da tutti gli hosts...")
        
        # 🆕 Seed robusto
        if state.has("seed"):
            seed = state.get("seed", int)
        elif state.has("data.seed"):
            seed = state.get("data.seed", int)
        else:
            seed = 42
            logger.warning(f"⚠️ Seed non trovato, uso default: {seed}")
        
        # ✅ Estrai i test set e pulisci gli attributi
        test_sets = []
        host_mapping = []  # 🆕 Traccia origine dei sample
        
        for ip, splits in federated_splits.items():
            test_df = splits['test'].copy()
            test_df.attrs = {}  # Pulisci attributi
        
            # 🆕 Aggiungo colonna per tracciare l'host di origine
            if self.save_host_labels:
                test_df['_host_origin'] = ip
                host_mapping.extend([ip] * len(test_df))
            
            test_sets.append(test_df)
        
        # Concatena tutti i test set
        aggregated_test = pd.concat(test_sets, ignore_index=True)
        
        # Shuffle
        aggregated_test = aggregated_test.sample(
            frac=1, 
            random_state=seed
        ).reset_index(drop=True)
        
        logger.info(f"Test set aggregato: {len(aggregated_test)} campioni")
        
        
        # 🆕 Verifica robusta che controlla se attack_col esista
        if self.attack_col in aggregated_test.columns:
            label_dist = aggregated_test[self.attack_col].value_counts()
            logger.info(f"📊 Distribuzione etichette: {dict(label_dist)}")
        else:
            logger.warning(f"⚠️ Colonna '{self.attack_col}' non trovata nel test set")
            
            
        # 🆕 Crea DataFrame di mapping host
        if self.save_host_labels:
            host_mapping_df = pd.DataFrame({
                'sample_index': range(len(aggregated_test)),
                'host_origin': aggregated_test['_host_origin']
            })
            # Rimuovi colonna temporanea
            aggregated_test = aggregated_test.drop(columns=['_host_origin'])
        else:
            host_mapping_df = None
        
        return {
            "aggregated_test": aggregated_test,
            "test_host_mapping": host_mapping_df  # aggiunto mapping host
        }

