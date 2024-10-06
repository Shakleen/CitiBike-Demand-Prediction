import pytest
from unittest.mock import Mock, patch
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator
import numpy as np

from src.train_pipeline.random_forest_pipeline import (
    RandomForestPipelineConfig,
    RandomForestPipeline,
)


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
def pipeline(spark: SparkSession) -> RandomForestPipeline:
    return RandomForestPipeline(spark)


def test_init(pipeline: RandomForestPipeline):
    assert isinstance(pipeline.config, RandomForestPipelineConfig)
    assert isinstance(pipeline.spark, SparkSession)


def test_split_train_val_test(pipeline: RandomForestPipeline, spark: SparkSession):
    data = spark.createDataFrame(
        data=np.random.randn(100, 4),
        schema="F1 float, F2 float, F3 float, L float",
    )

    train_data, val_data, test_data = pipeline.split_train_val_test(data)

    assert train_data.count() > val_data.count()
    assert train_data.count() > test_data.count()


@pytest.mark.parametrize(
    ("label_col", "pred_col"),
    (
        ["bike_demand", "predicted_bike_demand"],
        ["dock_demand", "predicted_dock_demand"],
    ),
)
def test_get_regressor(pipeline: RandomForestPipeline, label_col: str, pred_col: str):
    regressor: RandomForestRegressor = pipeline.get_regressor(label_col, pred_col)
    config = pipeline.config

    assert regressor.getFeaturesCol() == config.feature_column_name
    assert regressor.getLabelCol() == label_col
    assert regressor.getPredictionCol() == pred_col
    assert regressor.getSeed() == config.seed
    assert regressor.getSubsamplingRate() == config.subsampling_rate
    assert regressor.getMaxDepth() == config.max_depth
    assert regressor.getNumTrees() == config.num_trees


def test_eval_model():
    pipeline = RandomForestPipeline(None)
    mock_evaluator_rmse = Mock(RegressionEvaluator)
    mock_evaluator_r2 = Mock(RegressionEvaluator)
    mock_data = Mock(DataFrame)
    mock_pred_data = Mock(DataFrame)
    mock_model = Mock(RandomForestRegressionModel)
    expected_rmse = 1
    expected_r2 = 2

    mock_model.transform.return_value = mock_pred_data
    mock_evaluator_r2.evaluate.return_value = expected_r2
    mock_evaluator_rmse.evaluate.return_value = expected_rmse

    rmse, r2 = pipeline.eval_model(
        mock_model,
        mock_data,
        mock_evaluator_rmse,
        mock_evaluator_r2,
    )

    assert rmse == expected_rmse
    assert r2 == expected_r2
    mock_model.transform.assert_called_once_with(mock_data)
    mock_evaluator_r2.evaluate.assert_called_once_with(mock_pred_data)
    mock_evaluator_rmse.evaluate.assert_called_once_with(mock_pred_data)
