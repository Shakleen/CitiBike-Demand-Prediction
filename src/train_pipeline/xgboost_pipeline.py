import os
from dataclasses import dataclass
from pyspark.sql import SparkSession
from xgboost.spark import SparkXGBRegressor

from src.utils import read_delta


@dataclass
class XGBoostPipelineConfig:
    root_delta_path: str = os.path.join("Data", "delta")
    gold_delta_path: str = os.path.join(root_delta_path, "gold")
    model_artifact_path: str = os.path.join("artifacts", "model", "xgboost")
    feature_column_name: str = "final_features"
    number_of_workers: int = 6
    device: str = "cuda"


class XGBoostPipeline:
    def __init__(self, spark: SparkSession) -> None:
        self.spark = spark
        self.config = XGBoostPipelineConfig()

    def get_xgboost_regressor(self, label_column_name: str) -> SparkXGBRegressor:
        return SparkXGBRegressor(
            features_col=self.config.feature_column_name,
            label_col=label_column_name,
            num_workers=self.config.number_of_workers,
            device=self.config.device,
        )

    def train(self):
        data = read_delta(self.spark, self.config.gold_delta_path)
