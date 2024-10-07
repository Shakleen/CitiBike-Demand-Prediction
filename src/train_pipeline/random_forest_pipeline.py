import mlflow
import mlflow.spark
import os
from dataclasses import dataclass
from pyspark.sql import SparkSession
from pyspark.ml.regression import RandomForestRegressor
from typing import Tuple

from src.utils import read_delta
from src.train_pipeline.abstract_pipeline import AbstractPipeline, BaseConfig


@dataclass
class RandomForestPipelineConfig(BaseConfig):
    root_model_artifact_path: str = os.path.join("artifacts", "model", "random_forest")
    bike_model_artifact_path: str = os.path.join(
        root_model_artifact_path,
        "bike_model_rf",
    )
    dock_model_artifact_path: str = os.path.join(
        root_model_artifact_path,
        "dock_model_rf",
    )
    subsampling_rate: float = 0.01
    max_depth: int = 25
    num_trees: int = 100
    min_instances_per_node: int = 100
    max_bins: int = 32


class RandomForestPipeline(AbstractPipeline):
    def __init__(self, spark: SparkSession) -> None:
        super().__init__(spark, RandomForestPipelineConfig())

    def get_regressor(self, label_name: str, predict_name: str):
        return RandomForestRegressor(
            featuresCol=self.config.feature_column_name,
            labelCol=label_name,
            predictionCol=predict_name,
            seed=self.config.seed,
            subsamplingRate=self.config.subsampling_rate,
            maxDepth=self.config.max_depth,
            numTrees=self.config.num_trees,
            minInstancesPerNode=self.config.min_instances_per_node,
            maxBins=self.config.max_bins,
        )


if __name__ == "__main__":
    import pyspark
    from delta import configure_spark_with_delta_pip

    builder = (
        pyspark.sql.SparkSession.builder.appName("random_forest_train_pipeline")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.driver.memory", "15g")
        .config("spark.sql.shuffle.partitions", "6")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()

    pipeline = RandomForestPipeline(spark)
    pipeline.train()
    spark.stop()
