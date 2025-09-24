from typing import Any, Dict, Optional

import pandas as pd
import numpy as np

from ...core.step import Step
<<<<<<< HEAD
from ...data.partition import random_split, stratified_split    #funzioni per suddividere DataFrame in train/val/test calcolando gli indici delle 3 partizioni
from ...steps.helpers import validate_instance  #funzione helper per controllare che l’oggetto sia un DataFrame


def _validate_sizes(step: str, train: float, val: float, test: float) -> None:
    """Ensure sizes in [0,1] and sum to 1.0."""
    for label, v in (("train_size", train), ("val_size", val), ("test_size", test)):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{step}: {label} must be in [0, 1], got {v}.")
    total = train + val + test
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"{step}: sizes must sum to 1.0, got {total}.")
#tutta questa prima funzione serve per validare i parametri di split (train,val,test) che devono essere tra 0 e 1 e 
# controlla che la somma sia esattamente 1 (o molto vicino, tolleranza 1e-9).
#Serve a evitare errori logici nella definizione delle partizioni.
=======
from ...core.state import State
from ...data.partition import random_split, stratified_split

>>>>>>> upstream/main

class RandomSplit(Step):
    """Random split into train/val/test."""

    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        in_scope: str = "data",
        out_scope: str = "data",
        name: Optional[str] = None,
    ) -> None:
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
<<<<<<< HEAD
        self.random_state = random_state

        _validate_sizes(name or "random_split", train_size, val_size, test_size)    #la funzione di prima che controlla le proporzioni dei parametri

        super().__init__(
            name=name or "random_split",
            requires=[self.dataframe_in],
            provides=[self.dataframe_out, "mapping.split"], #fornisce il DataFrame di output e una mappatura delle partizioni
        )                                                   #tramite "mapping.split" si può accedere alla mappatura delle partizioni create (train/val/test)
=======

        super().__init__(
            name=name or "random_split",
            in_scope=in_scope,
            out_scope=out_scope,
        )
>>>>>>> upstream/main

    @Step.requires(root=pd.DataFrame, seed=int)
    @Step.provides(root=pd.DataFrame, split_mapping=dict)
    def run(
        self, state: State, root: pd.DataFrame, seed: int
    ) -> Optional[Dict[str, Any]]:

<<<<<<< HEAD
        if dataframe.empty: #se il DataFrame è vuoto, non fa nulla e assegna una mappatura vuota
            state["mapping.split"] = {}
            state[self.dataframe_out] = dataframe
            return

        split_mapping = random_split(       #divide casualmente il DataFrame in partizioni train/val/test
            dataframe,
=======
        if root.empty:
            return {"split_mapping": {}, "root": root}

        split_mapping = random_split(
            root,
>>>>>>> upstream/main
            train_size=self.train_size,
            val_size=self.val_size,
            test_size=self.test_size,
            random_state=seed,
        )

<<<<<<< HEAD
        dataframe.tab.set_partitions_from_labels(split_mapping)  #imposta le partizioni del DataFrame usando le etichette calcolate sullo split_mapping
        state["mapping.split"] = split_mapping              #salva la mappatura delle partizioni nello State sotto la chiave "mapping.split"
        state[self.dataframe_out] = dataframe               #salva il DataFrame con le partizioni nello State usando la chiave specificata in dataframe_out
=======
        root.tab.set_partitions_from_labels(split_mapping)
        return {"split_mapping": split_mapping, "root": root}
>>>>>>> upstream/main


class StratifiedSplit(Step):
    """Stratified split into train/val/test."""

    def __init__(
        self,
        class_column: str,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        in_scope: str = "data",
        out_scope: str = "data",
        name: Optional[str] = None,
    ) -> None:
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.class_column = class_column

        super().__init__(
            name=name or "stratified_split",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    @Step.requires(root=pd.DataFrame, seed=int)
    @Step.provides(root=pd.DataFrame, split_mapping=dict)
    def run(
        self, state: State, root: pd.DataFrame, seed: int
    ) -> Optional[Dict[str, Any]]:

        if not isinstance(self.class_column, str):
            raise ValueError("stratified_split: 'class_column' must be a string.")

<<<<<<< HEAD
        if not self.class_column:   #se non è specificata la colonna di classe, solleva un errore
            raise ValueError("stratified_split: 'class_column' must be provided.")

        split_mapping = stratified_split(   #divide il DataFrame in partizioni train/val/test mantenendo la distribuzione delle classi
            dataframe,
=======
        if root.empty:
            return {"split_mapping": {}, "root": root}

        split_mapping = stratified_split(
            root,
>>>>>>> upstream/main
            self.class_column,
            train_size=self.train_size,
            val_size=self.val_size,
            test_size=self.test_size,
            random_state=seed,
        )

        root.tab.set_partitions_from_labels(split_mapping)
        return {"split_mapping": split_mapping, "root": root}


class AssignSplitPartitions(Step):  #estrae le partizioni train/val/test dal DataFrame già splittato e le salva nello State con chiavi separate
    """Assign split partitions to separate keys in the State."""
    def __init__(
        self,
        in_scope: str = "data",
        out_scope: str = "data",
        name: Optional[str] = None,
    ) -> None:
<<<<<<< HEAD
        self.dataframe_in = dataframe_in
        self.dataframe_out = dataframe_out

        super().__init__(           #chiamata al costruttore della superclasse Step dichiarando:
            name=name or "assign_split_partitions",     #nome dello step (default "assign_split_partitions")
            requires=[self.dataframe_in],           #richiede il DataFrame di input / dati di input richiesti dallo step
            provides=[                              #fornisce i DataFrame di output / dati di output prodotti dallo step
                self.dataframe_out + ".train",      #chiave per il DataFrame di training
                self.dataframe_out + ".val",        #chiave per il DataFrame di validation
                self.dataframe_out + ".test",       #chiave per il DataFrame di test
            ],
        )

    def run(self, state: State) -> None:        #implementazione concreta del metodo astratto run di Step
        dataframe = state[self.dataframe_in]    #prende il DataFrame dallo State usando la chiave specificata in dataframe_in
        validate_instance(dataframe, pd.DataFrame, self.name)

        state[self.dataframe_out + ".train"] = dataframe.tab.train  #salva il DataFrame di training nello State usando la chiave specificata in dataframe_out + ".train"
        state[self.dataframe_out + ".val"] = dataframe.tab.val      #salva il DataFrame di validation nello State usando la chiave specificata in dataframe_out + ".val"
        state[self.dataframe_out + ".test"] = dataframe.tab.test    #salva il DataFrame di test nello State usando la chiave specificata in dataframe_out + ".test"
=======
        super().__init__(
            name=name or "assign_split_partitions",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    @Step.requires(root=pd.DataFrame)
    @Step.provides(train=pd.DataFrame, val=pd.DataFrame, test=pd.DataFrame)
    def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:
        return {"train": root.tab.train, "val": root.tab.val, "test": root.tab.test}


class AssignSplitTarget(Step):
    def __init__(
        self,
        in_scope: str = "data",
        out_scope: str = "test",
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            name=name or "assign_split_target",
            in_scope=in_scope,
            out_scope=out_scope,
        )

    @Step.requires(root=pd.DataFrame)
    @Step.provides(targets=np.ndarray)
    def run(self, state: State, root: pd.DataFrame) -> Optional[Dict[str, Any]]:
        return {"targets": root.tab.target.to_numpy()}
>>>>>>> upstream/main
