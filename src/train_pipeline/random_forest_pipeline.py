import mlflow
import mlflow.spark
import os
from dataclasses import dataclass
from pyspark.sql import SparkSession
from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.dataframe import DataFrame
from typing import Tuple

from src.utils import read_delta


@dataclass
class RandomForestPipelineConfig:
    root_delta_path: str = os.path.join("Data", "delta")
    gold_delta_path: str = os.path.join(root_delta_path, "gold")
    root_model_artifact_path: str = os.path.join("artifacts", "model", "random_forest")
    bike_model_artifact_path: str = os.path.join(
        root_model_artifact_path,
        "bike_model_rf",
    )
    dock_model_artifact_path: str = os.path.join(
        root_model_artifact_path,
        "dock_model_rf",
    )
    feature_column_name: str = "final_features"
    bike_demand_column_name: str = "bike_demand"
    dock_demand_column_name: str = "dock_demand"
    bike_demand_prediction_column_name: str = "predicted_bike_demand"
    dock_demand_prediction_column_name: str = "predicted_dock_demand"
    evaluation_metric_name: str = "rmse"
    seed: int = 29
    subsampling_rate: float = 0.01
    train_val_test_split_ratio = [0.9, 0.05, 0.05]
    max_depth: int = 25
    num_trees: int = 250


class RandomForestPipeline:
    def __init__(self, spark: SparkSession) -> None:
        self.spark = spark
        self.config = RandomForestPipelineConfig()

    def split_train_val_test(
        self,
        data: DataFrame,
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        train_data, val_data, test_data = data.randomSplit(
            self.config.train_val_test_split_ratio,
            seed=self.config.seed,
        )
        return (train_data, val_data, test_data)

    def get_regressor(self, label_name: str, predict_name: str):
        return RandomForestRegressor(
            featuresCol=self.config.feature_column_name,
            labelCol=label_name,
            predictionCol=predict_name,
            seed=self.config.seed,
            subsamplingRate=self.config.subsampling_rate,
            maxDepth=self.config.max_depth,
            numTrees=self.config.num_trees,
        )

    def eval_model(
        self,
        model: RandomForestRegressionModel,
        data: DataFrame,
        evaluator_rmse: RegressionEvaluator,
        evaluator_r2: RegressionEvaluator,
    ) -> Tuple[float, float]:
        predictions = model.transform(data)

        rmse = evaluator_rmse.evaluate(predictions)
        r2 = evaluator_r2.evaluate(predictions)

        return (rmse, r2)

    def train_model(
        self,
        train_data: DataFrame,
        val_data: DataFrame,
        test_data: DataFrame,
        label_name: str,
        predict_name: str,
    ):
        regressor = self.get_regressor(label_name, predict_name)
        evaluator_rmse = RegressionEvaluator(
            predictionCol=predict_name,
            labelCol=label_name,
            metricName="rmse",
        )
        evaluator_r2 = RegressionEvaluator(
            predictionCol=predict_name,
            labelCol=label_name,
            metricName="r2",
        )

        model = regressor.fit(train_data)

        val_rmse, val_r2 = self.eval_model(
            model,
            val_data,
            evaluator_rmse,
            evaluator_r2,
        )
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("val_r2", val_r2)

        test_rmse, test_r2 = self.eval_model(
            model,
            test_data,
            evaluator_rmse,
            evaluator_r2,
        )
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_r2", test_r2)

        mlflow.spark.log_model(model, f"{label_name}_random_forest_model")

        return model

    def train(self):
        data = read_delta(self.spark, self.config.gold_delta_path)

        train_data, val_data, test_data = pipeline.split_train_val_test(data)

        while mlflow.start_run(run_name="random_forest_model"):
            model = self.train_model(
                train_data,
                val_data,
                test_data,
                self.config.bike_demand_column_name,
                self.config.bike_demand_prediction_column_name,
            )
            model.write().overwrite().save(self.config.bike_model_artifact_path)

            model = self.train_model(
                train_data,
                val_data,
                test_data,
                self.config.dock_demand_column_name,
                self.config.dock_demand_prediction_column_name,
            )
            model.write().overwrite().save(self.config.dock_model_artifact_path)


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
