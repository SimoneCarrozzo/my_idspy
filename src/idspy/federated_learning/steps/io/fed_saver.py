from pathlib import Path
from typing import Optional, Any, Dict, Union

import pandas as pd

import logging
from src.idspy.events.handlers.logging import Logger

from src.idspy.core.step import Step
from src.idspy.core.state import State
from src.idspy.data.repository import DataFrameRepository
from src.idspy.nn.models.base import BaseModel
from src.idspy.nn.checkpoints import save_weights, save_checkpoint


class SaveFederatedData(Step):
    """Save federated datasets from state following the framework's pattern."""

    def __init__(
        self,
        base_path: Union[str, Path],
        fmt: Optional[str] = "parquet",
        save_meta: bool = True,
        save_statistics: bool = True,  # 🆕 Opzione per salvare stats
        save_other_stats: bool = True,  # 🆕 Salva anche JSON
        in_scope: str = "federated",
        out_scope: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.base_path: Path = Path(base_path)
        self.fmt = fmt
        self.save_meta = save_meta
        self.kwargs = kwargs

        self.save_statistics = save_statistics  # 🆕
        self.save_other_stats = save_other_stats  # 🆕

        super().__init__(
            name=name or "save_federated_data",
            in_scope=in_scope,
            out_scope=out_scope or in_scope,
        )

    @Step.requires(federated_splits=dict,
                   aggregated_test=pd.DataFrame,
                   test_host_mapping=pd.DataFrame,
                   non_iid_analysis=pd.DataFrame,  # 🆕 Aggiungiamo anche questa!
                   non_iid_metrics=dict,  # 🆕
                   ip_statistics=pd.DataFrame    # 🆕 Top hosts stats
    ) #AGGIUNTO test_host_mapping
    def run(
        self, 
        state: State, 
        federated_splits: Dict[str, Dict[str, pd.DataFrame]],
        aggregated_test: pd.DataFrame,
        test_host_mapping: pd.DataFrame,     #AGGIUNTO
        non_iid_analysis: pd.DataFrame,  # 🆕
        non_iid_metrics: dict,  # 🆕
        ip_statistics: pd.DataFrame    # 🆕 Top hosts stats
    ) -> Optional[Dict[str, Any]]:
        """
        Salva i dataset federati seguendo la struttura:
        base_path/
        ├── host_0/
        │   ├── train.parquet
        │   ├── val.parquet
        │   └── test.parquet
        ├── host_1/
        │   └── ...
        └── aggregated_test.parquet
        ├── test_host_mapping.parquet
        ├── non_iid_analysis.parquet  # 🆕
        ├── non_iid_analysis.json  # 🆕 
        ├── top_hosts.json    
        └── statistics.json  # 🆕
        """
        logger = logging.getLogger(__name__)
        
        # Crea directory base
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"💾 Inizio salvataggio dataset federati in: {self.base_path}")
        except Exception as e:
            logger.error(f"❌ Errore creazione directory base {self.base_path}: {e}")
            raise e
        
        logger.info(f"💾 Salvando {len(federated_splits)} dataset federati in {self.base_path}")
        
        # 🆕 Statistiche di salvataggio
        save_stats = {
            'total_hosts': len(federated_splits),
            'saved_hosts': 0,
            'failed_hosts': [],
            'splits_sizes': {}  # Per tracciare dimensioni di ogni split
        }
        
        for ip, splits in federated_splits.items():
            #AGGIUNTO BLOCCO TRY-EXCEPT PER CONTINUARE ANCHE IN CASO DI ERRORE SINGOLO
            try:
                # Sanitizza IP per nome cartella (es. "192.168.1.1" -> "192_168_1_1")
                safe_ip = ip.replace('.', '_').replace(':', '_')
                host_dir = self.base_path / f"host_{safe_ip}"
                host_dir.mkdir(exist_ok=True)
                
                logger.info(f"   💾 Host {ip}:")
                
                # 🆕 Traccia dimensioni
                host_sizes = {}
                
                # Salva train/val/test per questo host
                for split_name in ['train', 'val', 'test']:
                    df = splits[split_name]
                    
                    # 🆕 Traccia dimensione
                    host_sizes[split_name] = len(df)
                    
                    # Usa DataFrameRepository per mantenere coerenza con il framework
                    DataFrameRepository.save(
                        df,
                        host_dir,
                        name=split_name,
                        fmt=self.fmt,
                        save_meta=self.save_meta,
                        **self.kwargs,
                    )
                    logger.info(f"      ✅ {split_name}: {len(df)} samples")
                
                save_stats['saved_hosts'] += 1
                save_stats['splits_sizes'][ip] = host_sizes  # 🆕
                
            except Exception as e:
                logger.error(f"❌ Errore salvataggio dataset per host {ip}: {e}")
                save_stats['failed_hosts'].append(ip)  # 🆕
                # Continua con il prossimo host invece di crashare
                continue    
        logger.info(
            f"✅ Salvati correttamente dati per {save_stats['saved_hosts']}/"
            f"{save_stats['total_hosts']} hosts."
        )
        
        #AGGIUNTO IF DI CONTROLLO + BLOCCO TRY-EXCEPT VISTO CHE PRIMA C'ERA SOLO DATAFRAMEREPOSITORY.SAVE
        if aggregated_test is not None:
            try:
                DataFrameRepository.save(
                    aggregated_test,
                    self.base_path,
                    name="aggregated_test",
                    fmt=self.fmt,
                    save_meta=self.save_meta,
                    **self.kwargs,
                )
                logger.info(f"✅ Test Set Aggregato salvato ({len(aggregated_test)} righe).")
                save_stats['aggregated_test_size'] = len(aggregated_test)  # 🆕
            except Exception as e:
                logger.error(f"❌ Errore salvataggio aggregated_test: {e}")
        
        #AGGIUNTO NUOVO BLOCCO CHE PRIMA NON C'ERA
        if test_host_mapping is not None:
            try:
                DataFrameRepository.save(
                    test_host_mapping,
                    self.base_path,
                    name="test_host_mapping",
                    fmt=self.fmt,
                    save_meta=self.save_meta,
                    **self.kwargs,
                )
                logger.info(f"✅ Mapping Host-Test salvato.")
            except Exception as e:
                 logger.warning(f"⚠️ Impossibile salvare test_host_mapping: {e}")
        
        # 🆕 Salva analisi Non-IID
        if non_iid_analysis is not None:
            try:
                DataFrameRepository.save(
                    non_iid_analysis,
                    self.base_path,
                    name="non_iid_analysis",
                    fmt=self.fmt,
                    save_meta=self.save_meta,
                    **self.kwargs,
                )
                logger.info(f"✅ Analisi Non-IID salvata (parquet-version).")
            
                if self.save_other_stats:
                        json_path = self.base_path / "non_iid_analysis.json"
                        non_iid_analysis.to_json(json_path, orient='records', indent=2)
                        logger.info(f"✅ Analisi Non-IID salvata (JSON): {json_path}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Impossibile salvare non_iid_analysis: {e}")
        
        # 🆕 Salva statistiche Top Hosts
        if ip_statistics is not None:
            try:                
                # 🆕 JSON
                if self.save_other_stats:
                    json_path = self.base_path / "top_hosts.json"
                    ip_statistics.to_json(json_path, orient='records', indent=2)
                    logger.info(f"✅ Top Hosts salvati (JSON): {json_path}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Impossibile salvare top_hosts: {e}")
                
                    
        # 🆕 Salva statistiche come JSON
        if self.save_statistics:
            try:
                import json
                
                # Aggiungi metriche non-IID alle stats
                save_stats['non_iid_metrics'] = non_iid_metrics
                
                stats_file = self.base_path / "statistics.json"
                with open(stats_file, 'w') as f:
                    json.dump(save_stats, f, indent=2)
                
                logger.info(f"✅ Statistiche salvate in {stats_file}")
            except Exception as e:
                logger.warning(f"⚠️ Impossibile salvare statistiche: {e}")
        
        logger.info(f"\n✅ Tutti i dataset federati salvati con successo!")
        
        # 🆕 Ritorna le statistiche nello state
        return {"save_statistics": save_stats}
