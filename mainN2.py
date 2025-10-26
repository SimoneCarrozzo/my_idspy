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
from src.idspy.steps.transforms.sample import DownSampler2Min
from src.idspy.data.schema import Schema, ColumnRole
from src.idspy.data.tab_accessor import TabAccessor
from src.idspy.events.bus import EventBus
from src.idspy.events.events import only_id
from src.idspy.events.handlers.logging import Logger, DataFrameProfiler
from src.idspy.steps.io.saver import SaveData
from src.idspy.steps.io.loader import LoadData
from src.idspy.steps.builders.dataloader import BuildDataLoader
from src.idspy.steps.builders.dataset import BuildDataset
from src.idspy.steps.transforms.adjust import DropNulls, FilterZeroLabel, FeatureGenerator, CategoricalEncoder, FeatureGenV2
from src.idspy.steps.transforms.map import FrequencyMap, LabelMap
from src.idspy.steps.transforms.scale import StandardScale, ZScaler, MissingValueImputer, OutlierRemover 
from src.idspy.steps.transforms.split import (
    AssignSplitPartitions,
    StratifiedSplit,
    AssignSplitTarget,
)
from src.idspy.steps.model.training import TrainOneEpoch
from src.idspy.steps.model.evaluating import ValidateOneEpoch, MakePredictions
from src.idspy.steps.metrics.classification import ClassificationMetrics
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
        ColumnRole.CATEGORICAL
    )

    bus = EventBus()
    bus.subscribe(callback=Logger(), event_type=PipelineEvent.BEFORE_STEP)
    

    fit_aware_pipeline = FitAwareObservablePipeline(
        steps=[
            StandardScale(),
            #OutlierRemover(),
            #MissingValueImputer(),
            #ZScaler(),
            FrequencyMap(max_levels=20),
            LabelMap(),
        ],
        bus=bus,
        name="fit_aware_pipeline",
    )

    preprocessing_pipeline = ObservablePipeline(
        steps=[
            LoadData(
                path_in="c:/Users/simon/OneDrive/Documenti/TESI_UNI/DataSets/dataset_v2/cic_2018_v2.csv",
                schema=schema,
                nrows=1000000
            ),
            DropNulls(),
            #FilterZeroLabel(target_column=schema.columns(ColumnRole.TARGET)),
            FeatureGenV2(ratio_cols=("MIN_TTL", "MAX_TTL"), diff_cols=("DURATION_IN", "DURATION_OUT"), soglia_cols=("TCP_WIN_MAX_IN"), soglia=100),
            DownSampler2Min(class_column=schema.columns(ColumnRole.TARGET)),
            CategoricalEncoder(),
            StratifiedSplit(class_column=schema.target),
            fit_aware_pipeline,
            SaveData(
                file_path="c:/Users/simon/OneDrive/Documenti/TESI_UNI/DataSets/processati",
                file_name="cic_2018_v2",
                fmt="parquet",
            ),
        ],
        bus=bus,
        name="preprocessing_pipeline",
    )

    device = get_device()
       
    state = State(
        {
            "device": device,
            "seed": 42,
        }
    )

    preprocessing_pipeline.run(state)
    # print("[MAIN] Classi mantenute:", state.get("kept_classes", dict))
    # print("[MAIN] Classi eliminate con conteggio:", state.get("dropped_label_counts", dict))
    # print("[MAIN] Numero righe eliminate:", state.get("filter_zero_labels.dropped_rows", dict))
    # filtered_data = state.get("data")
    # print("[MAIN] Distribuzione finale classi:")
    # print(filtered_data[schema.target].value_counts())
    # print("[MAIN] Dimensioni dataset finale:", filtered_data.shape)



if __name__ == "__main__":
    main()


