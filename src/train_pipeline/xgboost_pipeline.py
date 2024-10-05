import os
from dataclasses import dataclass
from pyspark.sql import SparkSession
from xgboost.spark import SparkXGBRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder
from typing import List

from src.utils import read_delta


@dataclass
class XGBoostPipelineConfig:
    root_delta_path: str = os.path.join("Data", "delta")
    gold_delta_path: str = os.path.join(root_delta_path, "gold")
    model_artifact_path: str = os.path.join("artifacts", "model", "xgboost")
    feature_column_name: str = "final_features"
    number_of_workers: int = 6
    device: str = "cuda"
    bike_demand_column_name: str = "bike_demand"
    dock_demand_column_name: str = "dock_demand"
    bike_demand_prediction_column_name: str = "predicted_bike_demand"
    dock_demand_prediction_column_name: str = "predicted_dock_demand"
    evaluation_metric_name: str = "rmse"
    search_n_estimators = [100, 200, 300]
    search_max_depths = [3, 6, 9]
    search_learning_rates =  [0.01, 0.1, 0.2, 0.3]


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
            objective="reg:squarederror",
        )

    def get_evaluator(
        self,
        label_column_name: str,
        predicted_column_name: str,
    ) -> RegressionEvaluator:
        return RegressionEvaluator(
            predictionCol=predicted_column_name,
            labelCol=label_column_name,
            metricName=self.config.evaluation_metric_name,
        )

    def get_hyperparameter_grid(self, xgb: SparkXGBRegressor) -> List:
        return (
            ParamGridBuilder()
            .addGrid(xgb.n_estimators, self.config.search_n_estimators)
            .addGrid(xgb.max_depth, self.config.search_max_depths)
            .addGrid(xgb.learning_rate, self.config.search_learning_rates)
            .build()
        )

    def train(self):
        data = read_delta(self.spark, self.config.gold_delta_path)

        regressor_bike_demand = self.get_xgboost_regressor(
            self.config.bike_demand_column_name
        )
        regressor_dock_demand = self.get_xgboost_regressor(
            self.config.dock_demand_column_name
        )
