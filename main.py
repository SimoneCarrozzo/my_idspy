import logging

import numpy as np

from src.idspy.core.pipeline import FittedInstrumentedPipeline, InstrumentedPipeline
from src.idspy.core.state import State
from src.idspy.data.tabular_data import TabularSchema
from src.idspy.events.bus import EventBus
from src.idspy.events.events import EventType
from src.idspy.services.logger import setup_logging
from src.idspy.steps.io import LoadTabularData
from src.idspy.steps.preprocessing.adjust import DropNulls
from src.idspy.steps.preprocessing.map import FrequencyMap, TargetMap
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
    bus.subscribe(EventType.PIPELINE_START, lambda e: logger.info(f"Starting Pipeline: {e.id}"))
    bus.subscribe(EventType.PIPELINE_END, lambda e: logger.info(f"Finished Pipeline: {e.id}"))
    bus.subscribe(EventType.STEP_START, lambda e: logger.info(f"Starting Step: {e.id}"))
    bus.subscribe(EventType.STEP_END, lambda e: logger.info(f"Finished Step: {e.id}"))
    
    fitted_pipeline = FittedInstrumentedPipeline(
        steps=[
            StandardScale(),
            FrequencyMap(max_levels=3),
            TargetMap(),
        ],
        bus=bus,
        name="fitted_pipeline",
    )

    main_pipeline = InstrumentedPipeline(
        steps=[
            LoadTabularData(path="resources/data/dataset_v2/cic_2018_v2.csv", schema=schema),
            DropNulls(),
            StratifiedSplit(target=schema.target),
            fitted_pipeline,
        ],
        bus=bus,
        name="main_pipeline",
    )

    state = State()
    main_pipeline.run(state)


if __name__ == '__main__':
    main()
