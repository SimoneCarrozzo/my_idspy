import logging

import numpy as np

from src.idspy.core.pipeline import FitAwareObservablePipeline, ObservablePipeline
from src.idspy.core.state import State
from src.idspy.data.tabular_data import TabularSchema
from src.idspy.events.bus import EventBus
from src.idspy.events.subscribers.logging import Logger, Tracer
from src.idspy.services.setup import setup_logging
from src.idspy.steps.io import LoadTabularData, SaveTabularData
from src.idspy.steps.preprocessing.adjust import DropNulls
from src.idspy.steps.preprocessing.map import FrequencyMap, TargetMap
from src.idspy.steps.preprocessing.sample import DownsampleToMinority
from src.idspy.steps.preprocessing.scale import StandardScale
from src.idspy.steps.preprocessing.split import StratifiedSplit

setup_logging()
logger = logging.getLogger(__name__)
rng = np.random.default_rng(42)


def main():
    schema = TabularSchema(
        target="Attack",
        numeric=(
            "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
            "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
            "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES", "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
            "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS", "SRC_TO_DST_AVG_THROUGHPUT",
            "DST_TO_SRC_AVG_THROUGHPUT", "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
            "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES",
            "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "DNS_TTL_ANSWER"),
        categorical=(
            "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO", "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
            "ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_ID", "DNS_QUERY_TYPE"),
    )

    bus = EventBus()
    bus.subscribe(callback=Logger())
    bus.subscribe(callback=Tracer())

    fitted_pipeline = FitAwareObservablePipeline(
        steps=[
            StandardScale(),
            FrequencyMap(max_levels=3),
            TargetMap(),
        ],
        bus=bus,
        name="fit_aware_pipeline",
    )

    main_pipeline = ObservablePipeline(
        steps=[
            LoadTabularData(path="resources/data/dataset_v2/cic_2018_v2.csv", schema=schema),
            DropNulls(),
            DownsampleToMinority(target=schema.target),
            StratifiedSplit(target=schema.target),
            fitted_pipeline,
            SaveTabularData(path="resources/data/processed/dataset_v2/cic_2018/train.csv", input_key="data.train"),
            SaveTabularData(path="resources/data/processed/dataset_v2/cic_2018/val.csv", input_key="data.val"),
            SaveTabularData(path="resources/data/processed/dataset_v2/cic_2018/test.csv", input_key="data.test")
        ],
        bus=bus,
        name="main_pipeline",
    )

    state = State()
    main_pipeline.run(state)


if __name__ == '__main__':
    main()
