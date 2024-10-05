import pytest
from unittest.mock import Mock, patch
from pyspark.sql import SparkSession

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
    assert hasattr(config, "model_artifact_path")


def test_init(pipeline: XGBoostPipeline):
    assert hasattr(pipeline, "spark")
    assert hasattr(pipeline, "config")
