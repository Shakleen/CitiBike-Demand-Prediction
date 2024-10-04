import pytest
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer

from src.pipelines.data_preparation_pipeline_xgboost import (
    DataPreparationPipelineXGBoostConfig, DataPreparationPipelineXGboost,
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
def pipeline(spark: SparkSession):
    return DataPreparationPipelineXGboost(spark)


def test_config():
    config = DataPreparationPipelineXGBoostConfig()

    assert hasattr(config, "root_delta_path")
    assert hasattr(config, "gold_delta_path")
    assert hasattr(config, "data_artifact_path")
    assert hasattr(config, "categorical_column_names")
    assert hasattr(config, "numerical_column_names")
    assert hasattr(config, "cyclic_column_periods")

def test_init(pipeline: DataPreparationPipelineXGboost):
    assert hasattr(pipeline, "spark")
    assert hasattr(pipeline, "config")


def test_get_column_transformer(pipeline: DataPreparationPipelineXGboost):
    transformer = pipeline.get_column_transformer()

    assert isinstance(transformer, ColumnTransformer)