import os
from dataclasses import dataclass
from pyspark.sql import SparkSession

from src.utils import read_delta


@dataclass
class XGBoostPipelineConfig:
    root_delta_path: str = os.path.join("Data", "delta")
    gold_delta_path: str = os.path.join(root_delta_path, "gold")
    model_artifact_path: str = os.path.join("artifacts", "model", "xgboost")


class XGBoostPipeline:
    def __init__(self, spark: SparkSession) -> None:
        self.spark = spark
        self.config = XGBoostPipelineConfig()
