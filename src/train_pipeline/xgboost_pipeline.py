import mlflow
import mlflow.spark
import os
from dataclasses import dataclass
from xgboost.spark import SparkXGBRegressor
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.dataframe import DataFrame
from typing import List

from src.utils import read_delta, write_delta


@dataclass
class XGBoostPipelineConfig:
    root_delta_path: str = os.path.join("Data", "delta")
    gold_delta_path: str = os.path.join(root_delta_path, "gold")
    prediction_delta_path: str = os.path.join(root_delta_path, "xgboost_prediction")
    root_model_artifact_path: str = os.path.join("artifacts", "model", "xgboost")
    bike_model_artifact_path: str = os.path.join(root_model_artifact_path, "bike_model")
    dock_model_artifact_path: str = os.path.join(root_model_artifact_path, "dock_model")
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
    search_learning_rates = [0.01, 0.1, 0.2, 0.3]
    cv_folds: int = 10
    seed: int = 29


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

    def get_hyperparameter_grid(self, xgb: SparkXGBRegressor) -> List:
        return (
            ParamGridBuilder()
            .addGrid(xgb.n_estimators, self.config.search_n_estimators)
            .addGrid(xgb.max_depth, self.config.search_max_depths)
            .addGrid(xgb.learning_rate, self.config.search_learning_rates)
            .build()
        )

    def get_best_model(self, data: DataFrame, label_col: str, pred_col: str):
        evaluator_rmse = RegressionEvaluator(
            predictionCol=pred_col,
            labelCol=label_col,
            metricName="rmse",
        )
        evaluator_r2 = RegressionEvaluator(
            predictionCol=pred_col,
            labelCol=label_col,
            metricName="r2",
        )
        estimator = self.get_xgboost_regressor(label_col)
        cv = CrossValidator(
            estimator=estimator,
            estimatorParamMaps=self.get_hyperparameter_grid(estimator),
            evaluator=evaluator_rmse,
            numFolds=self.config.cv_folds,
            seed=self.config.seed,
        )

        with mlflow.start_run():
            cv_model = cv.fit(data)

            for i, metrics in enumerate(cv_model.avgMetrics):
                mlflow.log_metric(
                    f"fold_{i+1}_{self.config.evaluation_metric_name}",
                    metrics,
                )

            best_model = cv_model.bestModel
            predictions = best_model.transform(data)

            overall_rmse = evaluator_rmse.evaluate(predictions)
            overall_r2 = evaluator_r2.evaluate(predictions)

            mlflow.log_metric("overall_rmse", overall_rmse)
            mlflow.log_metric("overall_r2", overall_r2)

        return best_model

    def train(self):
        data = read_delta(self.spark, self.config.gold_delta_path)

        bike_demand_model = self.get_best_model(
            data,
            self.config.bike_demand_column_name,
            self.config.bike_demand_prediction_column_name,
        )
        mlflow.spark.log_model(bike_demand_model, self.config.bike_model_artifact_path)

        dock_demand_model = self.get_best_model(
            data,
            self.config.dock_demand_column_name,
            self.config.dock_demand_prediction_column_name,
        )
        mlflow.spark.log_model(dock_demand_model, self.config.dock_model_artifact_path)

        write_delta(data, self.config.prediction_delta_path)


if __name__ == "__main__":
    import pyspark
    from delta import configure_spark_with_delta_pip

    builder = (
        pyspark.sql.SparkSession.builder.appName("raw_to_bronze")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.driver.memory", "15g")
        .config("spark.sql.shuffle.partitions", "6")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()

    pipeline = XGBoostPipeline(spark)
    pipeline.train()
