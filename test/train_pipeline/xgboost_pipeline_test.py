import pytest
from unittest.mock import Mock, patch
from pyspark.sql import SparkSession
from xgboost.spark import SparkXGBRegressor
from pyspark.sql.dataframe import DataFrame

from src.train_pipeline.xgboost_pipeline import XGBoostPipelineConfig, XGBoostPipeline


@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("local-tests")
        .config("spark.executor.cores", "1")
        .config("spark.executor.instances", "1")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .getOrCreate()
    )
    yield spark
    spark.stop()


@pytest.fixture
def pipeline(spark: SparkSession) -> XGBoostPipeline:
    return XGBoostPipeline(spark)


def test_config():
    config = XGBoostPipelineConfig()

    assert hasattr(config, "root_delta_path")
    assert hasattr(config, "gold_delta_path")
    assert hasattr(config, "prediction_delta_path")
    assert hasattr(config, "root_model_artifact_path")
    assert hasattr(config, "bike_model_artifact_path")
    assert hasattr(config, "dock_model_artifact_path")
    assert hasattr(config, "feature_column_name")
    assert hasattr(config, "number_of_workers")
    assert hasattr(config, "device")
    assert hasattr(config, "bike_demand_column_name")
    assert hasattr(config, "dock_demand_column_name")
    assert hasattr(config, "bike_demand_prediction_column_name")
    assert hasattr(config, "dock_demand_prediction_column_name")
    assert hasattr(config, "evaluation_metric_name")
    assert hasattr(config, "search_n_estimators")
    assert hasattr(config, "search_max_depths")
    assert hasattr(config, "search_learning_rates")
    assert hasattr(config, "cv_folds")
    assert hasattr(config, "seed")


def test_init(pipeline: XGBoostPipeline):
    assert hasattr(pipeline, "spark")
    assert hasattr(pipeline, "config")


@pytest.mark.parametrize("label", ["bike_demand", "dock_demand"])
def test_get_xgboost_regressor(pipeline: XGBoostPipeline, label: str):
    regressor = pipeline.get_xgboost_regressor(label)

    assert regressor.getLabelCol() == label
    assert regressor.getFeaturesCol() == pipeline.config.feature_column_name


def test_get_hyperparameter_grid(pipeline: XGBoostPipeline):
    xgb = pipeline.get_xgboost_regressor(pipeline.config.bike_demand_column_name)
    grid = pipeline.get_hyperparameter_grid(xgb)

    assert len(grid) == len(pipeline.config.search_learning_rates) * len(
        pipeline.config.search_n_estimators
    ) * len(pipeline.config.search_max_depths)


@pytest.mark.parametrize(
    ("predict_col", "label_col"),
    [
        ("bike_demand", "predicted_bike_demand"),
        ("dock_demand", "predicted_dock_demand"),
    ],
)
def test_get_best_model(pipeline: XGBoostPipeline, predict_col: str, label_col: str):
    mock_df = Mock(DataFrame)

    with (patch("src.train_pipeline.xgboost_pipeline.CrossValidator.fit") as patched_fit,
          patch("src.train_pipeline.xgboost_pipeline.RegressionEvaluator.evaluate") as patched_eval):
        model = pipeline.get_best_model(mock_df, predict_col, label_col)
        patched_fit.assert_called_once_with(mock_df)
