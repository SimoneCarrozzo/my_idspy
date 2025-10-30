from pathlib import Path
from typing import Optional, Any, Dict, Union

import pandas as pd

import logging
from src.idspy.events.handlers.logging import Logger

from ...core.step import Step
from ...core.state import State
from ...data.repository import DataFrameRepository
from ...nn.models.base import BaseModel
from ...nn.checkpoints import save_weights, save_checkpoint


class SaveFederatedData(Step):
    """Save federated datasets from state following the framework's pattern."""

    def __init__(
        self,
        base_path: Union[str, Path],
        fmt: Optional[str] = "parquet",
        save_meta: bool = True,
        in_scope: str = "federated",
        out_scope: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.base_path: Path = Path(base_path)
        self.fmt = fmt
        self.save_meta = save_meta
        self.kwargs = kwargs

        super().__init__(
            name=name or "save_federated_data",
            in_scope=in_scope,
            out_scope=out_scope or in_scope,
        )

    @Step.requires(federated_splits=dict, aggregated_test=pd.DataFrame)
    def run(
        self, 
        state: State, 
        federated_splits: Dict[str, Dict[str, pd.DataFrame]],
        aggregated_test: pd.DataFrame
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
        """
        logger = logging.getLogger(__name__)
        
        # Crea directory base
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Salvando {len(federated_splits)} dataset federati in {self.base_path}")

         # Salva dataset per ogni host (IP)
        for ip, splits in federated_splits.items():
            # Sanitizza IP per nome cartella (es. "192.168.1.1" -> "192_168_1_1")
            safe_ip = ip.replace('.', '_').replace(':', '_')
            host_dir = self.base_path / f"host_{safe_ip}"
            host_dir.mkdir(exist_ok=True)
            
            logger.info(f"   💾 Host {ip}:")
            
            # Salva train/val/test per questo host
            for split_name in ['train', 'val', 'test']:
                df = splits[split_name]
                
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
        
        # Salva test set aggregato
        logger.info(f"   💾 Test set aggregato:")
        DataFrameRepository.save(
            aggregated_test,
            self.base_path,
            name="aggregated_test",
            fmt=self.fmt,
            save_meta=self.save_meta,
            **self.kwargs,
        )
        logger.info(f"      ✅ {len(aggregated_test)} samples")
        
        logger.info(f"\n✅ Tutti i dataset federati salvati con successo!")







# class SaveModel(Step):
#     """Save model from state."""

#     def __init__(
#         self,
#         path_out: Union[str, Path],
#         in_scope: Optional[str] = None,
#         name: Optional[str] = None,
#         **kwargs: Any,
#     ) -> None:
#         self.path_out: Path = Path(path_out)
#         self.kwargs = kwargs

#         super().__init__(
#             name=name or "save_model",
#             in_scope=in_scope,
#         )

#     @Step.requires(model=BaseModel)
#     def run(self, state: State, model: BaseModel) -> None:
#         save_weights(model, self.path_out, **self.kwargs)
