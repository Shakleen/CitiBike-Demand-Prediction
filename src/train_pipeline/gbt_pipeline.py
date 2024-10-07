from pyspark.sql import SparkSession
from pyspark.ml.regression import GBTRegressor
from dataclasses import dataclass
import os

from src.train_pipeline.abstract_pipeline import AbstractPipeline
from src.train_pipeline.random_forest_pipeline import RandomForestPipelineConfig


@dataclass
class GBTPipelineConfig(RandomForestPipelineConfig):
    root_model_artifact_path: str = os.path.join("artifacts", "model", "random_forest")
    bike_model_artifact_path: str = os.path.join(
        root_model_artifact_path,
        "bike_model_gbt",
    )
    dock_model_artifact_path: str = os.path.join(
        root_model_artifact_path,
        "dock_model_gbt",
    )


class GBTPipeline(AbstractPipeline):
    def __init__(self, spark: SparkSession) -> None:
        super().__init__(spark, GBTPipelineConfig())

    def get_regressor(self, label_name: str, predict_name: str):
        return GBTRegressor(
            featuresCol=self.config.feature_column_name,
            labelCol=label_name,
            predictionCol=predict_name,
            seed=self.config.seed,
            subsamplingRate=self.config.subsampling_rate,
            maxDepth=self.config.max_depth,
            minInstancesPerNode=self.config.min_instances_per_node,
            maxBins=self.config.max_bins,
            maxIter=self.config.num_trees,
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

    pipeline = GBTPipeline(spark)
    pipeline.train()
    spark.stop()
