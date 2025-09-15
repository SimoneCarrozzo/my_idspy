import logging

import torch

from src.idspy.common.logging import setup_logging
from src.idspy.common.seeds import set_seeds

from src.idspy.core.state import State
from src.idspy.core.pipeline import (
    FitAwareObservablePipeline,
    ObservablePipeline,
    PipelineEvent,
)

from src.idspy.data.schema import Schema, ColumnRole
from src.idspy.data.tab_accessor import TabAccessor

from src.idspy.events.bus import EventBus
from src.idspy.events.handlers.logging import Logger, DataFrameProfiler

from src.idspy.steps.io.saver import SaveData
from src.idspy.steps.io.loader import LoadData
from src.idspy.steps.builders.dataloader import BuildDataLoader
from src.idspy.steps.builders.dataset import BuildDataset
from src.idspy.steps.transforms.adjust import DropNulls
from src.idspy.steps.transforms.map import FrequencyMap, LabelMap
from src.idspy.steps.transforms.scale import StandardScale
from src.idspy.steps.transforms.split import AssignSplitPartitions, StratifiedSplit
from src.idspy.steps.model.training import TrainOneEpoch

from src.idspy.nn.batch import default_collate, Batch
from src.idspy.nn.helpers import get_device
from src.idspy.nn.models.classifier import TabularClassifier
from src.idspy.nn.losses.classification import ClassificationLoss


setup_logging()
logger = logging.getLogger(__name__)
set_seeds(42)


def main():
    schema = Schema()
    schema.add(["Attack"], ColumnRole.TARGET)
    schema.add(
        [
            "IN_BYTES",
            "IN_PKTS",
            "OUT_BYTES",
            "OUT_PKTS",
            "FLOW_DURATION_MILLISECONDS",
            "DURATION_IN",
            "DURATION_OUT",
            "MIN_TTL",
            "MAX_TTL",
            "LONGEST_FLOW_PKT",
            "SHORTEST_FLOW_PKT",
            "MIN_IP_PKT_LEN",
            "MAX_IP_PKT_LEN",
            "SRC_TO_DST_SECOND_BYTES",
            "DST_TO_SRC_SECOND_BYTES",
            "RETRANSMITTED_IN_BYTES",
            "RETRANSMITTED_IN_PKTS",
            "RETRANSMITTED_OUT_BYTES",
            "RETRANSMITTED_OUT_PKTS",
            "SRC_TO_DST_AVG_THROUGHPUT",
            "DST_TO_SRC_AVG_THROUGHPUT",
            "NUM_PKTS_UP_TO_128_BYTES",
            "NUM_PKTS_128_TO_256_BYTES",
            "NUM_PKTS_256_TO_512_BYTES",
            "NUM_PKTS_512_TO_1024_BYTES",
            "NUM_PKTS_1024_TO_1514_BYTES",
            "TCP_WIN_MAX_IN",
            "TCP_WIN_MAX_OUT",
            "DNS_TTL_ANSWER",
        ],
        ColumnRole.NUMERICAL,
    )
    schema.add(
        [
            "L4_SRC_PORT",
            "L4_DST_PORT",
            "PROTOCOL",
            "L7_PROTO",
            "TCP_FLAGS",
            "CLIENT_TCP_FLAGS",
            "SERVER_TCP_FLAGS",
            "ICMP_TYPE",
            "ICMP_IPV4_TYPE",
            "DNS_QUERY_ID",
            "DNS_QUERY_TYPE",
        ],
        ColumnRole.CATEGORICAL,
    )

    bus = EventBus()
    bus.subscribe(callback=Logger(), event_type=PipelineEvent.BEFORE_STEP)
    # bus.subscribe(callback=Tracer())
    # bus.subscribe(callback=DataFrameProfiler(), event_type=PipelineEvent.AFTER_STEP)

    fit_aware_pipeline = FitAwareObservablePipeline(
        steps=[
            StandardScale(),
            FrequencyMap(max_levels=20),
            LabelMap(),
        ],
        bus=bus,
        name="fit_aware_pipeline",
    )

    preprocessing_pipeline = ObservablePipeline(
        steps=[
            LoadData(
                path_in="resources/data/dataset_v2/cic_2018_v2.csv",
                schema=schema,
            ),
            DropNulls(),
            # DownsampleToMinority(class_column=schema.columns(ColumnRole.TARGET)[0]),
            StratifiedSplit(class_column=schema.columns(ColumnRole.TARGET)[0]),
            fit_aware_pipeline,
            SaveData(path_out="resources/data/processed/cic_2018_v2", fmt="parquet"),
        ],
        bus=bus,
        name="preprocessing_pipeline",
    )

    training_pipeline = ObservablePipeline(
        steps=[
            LoadData(path_in="resources/data/processed/cic_2018_v2/root.parquet"),
            AssignSplitPartitions(),
            BuildDataset(dataframe_in="data.train", dataset_out="train.dataset"),
            BuildDataLoader(
                dataset_in="train.dataset",
                dataloader_out="train.dataloader",
                batch_size=256,
                shuffle=True,
                collate_fn=default_collate,
            ),
            TrainOneEpoch(),
        ],
        bus=bus,
        name="training_pipeline",
    )

    state = State()
    state["device"] = get_device()

    state["model"] = TabularClassifier(
        num_features=len(schema.numerical),
        cat_cardinalities=[20] * len(schema.categorical),
        num_classes=15,
        hidden_dims=[128, 64],
        dropout=0.1,
    ).to(state["device"])
    state["loss"] = ClassificationLoss().to(state["device"])
    state["optimizer"] = torch.optim.Adam(state["model"].parameters(), lr=0.001)

    # preprocessing_pipeline.run(state)
    training_pipeline.run(state)


if __name__ == "__main__":
    main()
