import logging

import numpy as np


from src.idspy.core.pipeline import (
    FitAwareObservablePipeline,
    ObservablePipeline,
    PipelineEvent,
)
from src.idspy.common.logging import setup_logging
from src.idspy.core.state import State
from src.idspy.data.schema import Schema, ColumnRole
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
from src.idspy.data.tab_accessor import TabAccessor


setup_logging()
logger = logging.getLogger(__name__)
rng = np.random.default_rng(42)


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
    bus.subscribe(callback=DataFrameProfiler(), event_type=PipelineEvent.AFTER_STEP)

    fit_aware_pipeline = FitAwareObservablePipeline(
        steps=[
            StandardScale(),
            FrequencyMap(max_levels=3),
            LabelMap(),
        ],
        bus=bus,
        name="fit_aware_pipeline",
    )

    preprocessing_pipeline = ObservablePipeline(
        steps=[
            LoadData(
                path="resources/data/dataset_v2/cic_2018_v2.csv",
                schema=schema,
                nrows=1000000,
            ),
            DropNulls(),
            # DownsampleToMinority(class_col=schema.columns(ColumnRole.TARGET)[0]),
            StratifiedSplit(class_col=schema.columns(ColumnRole.TARGET)[0]),
            fit_aware_pipeline,
            # SaveData(path="resources/data/processed/cic_2018_v2", fmt="parquet"),
        ],
        bus=bus,
        name="preprocessing_pipeline",
    )

    training_pipeline = ObservablePipeline(
        steps=[
            AssignSplitPartitions(),
            BuildDataset(source="data.train", target="dataset.train"),
            BuildDataLoader(
                source="dataset.train",
                target="dataloader.train",
                batch_size=64,
                shuffle=True,
            ),
        ],
        bus=bus,
        name="training_pipeline",
    )

    state = State()
    preprocessing_pipeline.run(state)
    training_pipeline.run(state)


if __name__ == "__main__":
    main()
