import pytest
from unittest.mock import Mock, patch
from pyspark.sql import SparkSession

from src.components.silver_to_gold_transformer import (
    SilverToGoldTransformerConfig,
    SilverToGoldTransformer,
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
def transformer(spark: SparkSession):
    return SilverToGoldTransformer(spark)


def test_config():
    config = SilverToGoldTransformerConfig()

    assert hasattr(config, "root_delta_path")
    assert hasattr(config, "silver_delta_path")
    assert hasattr(config, "gold_delta_path")


def test_init(transformer: SilverToGoldTransformer, spark: SparkSession):
    assert hasattr(transformer, "config")
    assert hasattr(transformer, "spark")
    assert transformer.spark is spark
    assert isinstance(transformer.config, SilverToGoldTransformerConfig)
    assert isinstance(transformer.spark, SparkSession)
