import pytest
from unittest.mock import Mock, patch
from pyspark.sql import SparkSession
from pyspark.ml.regression import GBTRegressor

from src.train_pipeline.gbt_pipeline import GBTPipeline, GBTPipelineConfig


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
def pipeline(spark: SparkSession) -> GBTPipeline:
    return GBTPipeline(spark)


def test_init(pipeline: GBTPipeline):
    assert isinstance(pipeline.config, GBTPipelineConfig)
    assert isinstance(pipeline.spark, SparkSession)


@pytest.mark.parametrize(
    ("label_col", "pred_col"),
    (
        ["bike_demand", "predicted_bike_demand"],
        ["dock_demand", "predicted_dock_demand"],
    ),
)
def test_get_regressor(pipeline: GBTPipeline, label_col: str, pred_col: str):
    regressor: GBTRegressor = pipeline.get_regressor(label_col, pred_col)
    config: GBTPipelineConfig = pipeline.config

    assert regressor.getFeaturesCol() == config.feature_column_name
    assert regressor.getLabelCol() == label_col
    assert regressor.getPredictionCol() == pred_col
    assert regressor.getSeed() == config.seed
    assert regressor.getSubsamplingRate() == config.subsampling_rate
    assert regressor.getMaxDepth() == config.max_depth
    assert regressor.getMaxIter() == config.num_trees
