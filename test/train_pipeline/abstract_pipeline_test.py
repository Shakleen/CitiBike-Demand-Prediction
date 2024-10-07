import pytest
from unittest.mock import Mock, patch
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel
from pyspark.sql.dataframe import DataFrame
import numpy as np
from dataclasses import dataclass

from src.train_pipeline.abstract_pipeline import AbstractPipeline, BaseConfig


class TestPipeline(AbstractPipeline):
    def __init__(self, spark: SparkSession) -> None:
        super().__init__(spark, BaseConfig())

    def get_regressor(self, label_name: str, predict_name: str):
        return None


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
def pipeline(spark: SparkSession) -> TestPipeline:
    return TestPipeline(spark)


def test_init():
    with pytest.raises(TypeError):
        mock_spark = Mock(SparkSession)
        mock_config = Mock()
        transformer = AbstractPipeline(mock_spark, mock_config)


def test_train():
    assert getattr(
        AbstractPipeline.__dict__["get_regressor"],
        "__isabstractmethod__",
        False,
    )


def test_split_train_val_test(pipeline: TestPipeline, spark: SparkSession):
    data = spark.createDataFrame(
        data=np.random.randn(100, 4),
        schema="F1 float, F2 float, F3 float, L float",
    )

    train_data, val_data, test_data = pipeline.split_train_val_test(data)

    assert train_data.count() > val_data.count()
    assert train_data.count() > test_data.count()


def test_eval_model():
    pipeline = TestPipeline(None)
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
