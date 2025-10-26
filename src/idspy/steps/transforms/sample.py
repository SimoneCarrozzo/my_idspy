from typing import Any, Dict, Optional

import pandas as pd

from ...core.step import Step
from ...core.state import State
from ...data.tab_accessor import reattach_meta


class DownsampleToMinority(Step):
    """Downsample each class to the size of the minority class."""

    def __init__(
        self,
        class_column: str,
        in_scope: str = "data",
        out_scope: str = "data",
        name: Optional[str] = None,
    ) -> None:
        self.class_column = class_column

        super().__init__(
            name=name or "downsample_to_minority",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    @Step.requires(data_root=pd.DataFrame, seed=int)
    @Step.provides(data_root=pd.DataFrame)
    def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:

        # Early exits for edge cases
        if root.empty or self.class_column not in root.columns:
            return {"root": root}

        counts = root[self.class_column].value_counts(dropna=False)
        if counts.empty:
            return {"root": root}

        minority = int(counts.min())
        if minority <= 0:
            sampled = root.iloc[0:0]  # empty but keep schema
            return {"root": reattach_meta(root, sampled)}

        sampled = root.groupby(
            self.class_column, dropna=False, group_keys=False, sort=False
        ).sample(n=minority, replace=False, random_state=self.random_state)

        return {"root": reattach_meta(root, sampled)}


class Downsample(Step):
    """Downsample rows globally or per class."""

    def __init__(
        self,
        frac: float,
        class_column: Optional[str] = None,
        in_scope: str = "data",
        out_scope: str = "data",
        name: Optional[str] = None,
    ) -> None:
        if not (0.0 < frac <= 1.0):
            raise ValueError(f"downsample: frac must be in (0, 1], got {frac}.")

        self.frac = frac
        self.class_column = class_column

        super().__init__(
            name=name or "downsample",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    @Step.requires(root=pd.DataFrame)
    @Step.provides(root=pd.DataFrame)
    def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:

        if root.empty:
            return {"root": root}

        if self.class_column is not None and self.class_column in root.columns:
            sampled = root.groupby(
                self.class_column, dropna=False, group_keys=False, sort=False
            ).sample(frac=self.frac, replace=False, random_state=self.random_state)
        else:
            # Global sampling (handles both None class_column and missing column cases)
            sampled = root.sample(
                frac=self.frac, replace=False, random_state=self.random_state
            )

        return {"root": reattach_meta(root, sampled)}




#MIo TRY

class DownSampler2Min(Step):
    
    def __init__(
        self,
        class_column: str,
        name: Optional[str] = None,
        in_scope: str = "data",
        out_scope: str = "data",
        random_state: Optional[int] = None,
      ) -> None:
        self.class_column = class_column
        self.random_state = random_state
        super().__init__(
            name=name or "down_sampler_2_min",
            in_scope=in_scope,
            out_scope=out_scope,
        )
    
    @Step.requires(root=pd.DataFrame)
    @Step.provides(root=pd.DataFrame)
    def run(self, state:State, root:pd.DataFrame) -> Optional[Dict[str,Any]]:
        
        if self.class_column is None:
            print("[DownSampler2Min] ⚠️ Nessuna colonna target definita, ritorno dataset originale.")
            return {"root": root}
        
        print(f"[DownSampler2Min] Target column: {self.class_column}")
        counts = root[self.class_column].value_counts(dropna=False)
        print("[DownSampler2Min] Distribuzione iniziale classi:")
        print(counts)
        min_size = counts.min()     
        print(f"[DownSampler2Min] Dimensione classe minoritaria: {min_size}")

        df_list = []
        for class_value in root[self.class_column].unique():
            class_count = counts[class_value]
            if class_count > min_size:
                subset = root[root[self.class_column] == class_value].sample(
                n=min_size, replace=False, random_state=self.random_state)
                print(f"[DownSampler2Min] Classe '{class_value}': {class_count} → {len(subset)} (downsampled)")
            else:
                subset = root[root[self.class_column] == class_value]
                print(f"[DownSampler2Min] Classe '{class_value}': {class_count} (minority, invariata)")
            df_list.append(subset)
        
        sampled = pd.concat(df_list, axis=0)
        print("[DownSampler2Min] Distribuzione finale classi:")
        print(sampled[self.class_column].value_counts())
        
        return {"root": reattach_meta(root,sampled)}