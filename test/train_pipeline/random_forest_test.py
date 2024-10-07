import pytest
from unittest.mock import Mock, patch
from pyspark.sql import SparkSession
from pyspark.ml.regression import RandomForestRegressor

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


@pytest.mark.parametrize(
    ("label_col", "pred_col"),
    (
        ["bike_demand", "predicted_bike_demand"],
        ["dock_demand", "predicted_dock_demand"],
    ),
)
def test_get_regressor(pipeline: RandomForestPipeline, label_col: str, pred_col: str):
    regressor: RandomForestRegressor = pipeline.get_regressor(label_col, pred_col)
    config: RandomForestPipelineConfig = pipeline.config

    assert regressor.getFeaturesCol() == config.feature_column_name
    assert regressor.getLabelCol() == label_col
    assert regressor.getPredictionCol() == pred_col
    assert regressor.getSeed() == config.seed
    assert regressor.getSubsamplingRate() == config.subsampling_rate
    assert regressor.getMaxDepth() == config.max_depth
    assert regressor.getNumTrees() == config.num_trees
