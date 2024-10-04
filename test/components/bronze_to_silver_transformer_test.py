import pytest
from unittest.mock import Mock, patch
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
import pyspark.sql.types as T
import pyspark.sql.functions as F
import pandas as pd

from src.components.bronze_to_silver_transformer import (
    BronzeToSilverTransformerConfig,
    BronzeToSilverTransformer,
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
    return BronzeToSilverTransformer(spark)


def test_config():
    config = BronzeToSilverTransformerConfig()

    assert hasattr(config, "root_delta_path")
    assert hasattr(config, "bronze_delta_path")
    assert hasattr(config, "silver_delta_path")


def test_init(transformer: BronzeToSilverTransformer):
    assert hasattr(transformer, "config")
    assert isinstance(transformer.config, BronzeToSilverTransformerConfig)
    assert hasattr(transformer, "spark")
    assert isinstance(transformer.spark, SparkSession)


def test_read_bronze_delta():
    spark_mock = Mock(SparkSession)
    transformer = BronzeToSilverTransformer(spark_mock)
    dataframe_mock = Mock(DataFrame)

    spark_mock.read.format("delta").load.return_value = dataframe_mock

    df = transformer.read_bronze_delta()

    spark_mock.read.format.assert_called_with("delta")
    spark_mock.read.format("delta").load.assert_called_with(
        transformer.config.bronze_delta_path
    )

    assert df is dataframe_mock


def test_write_delta():
    spark_mock = Mock(SparkSession)
    transformer = BronzeToSilverTransformer(spark_mock)
    dataframe_mock = Mock(DataFrame)

    transformer.write_delta(dataframe_mock, transformer.config.silver_delta_path)

    dataframe_mock.write.save.assert_called_once_with(
        path=transformer.config.silver_delta_path,
        format="delta",
        mode="overwrite",
    )


def test_create_time_features(
    transformer: BronzeToSilverTransformer,
    spark: SparkSession,
):
    dataframe = spark.createDataFrame(
        [pd.Timestamp("2024-06-19 19:24:11"), pd.Timestamp("2024-06-19 19:35:37")],
        schema=T.StructType([T.StructField("timestamp", T.StringType(), True)]),
    )
    dataframe = dataframe.withColumn("timestamp", F.to_timestamp("timestamp"))

    output = transformer.create_time_features(dataframe, "timestamp")

    assert isinstance(output, DataFrame)
    assert set(output.columns) == {
        "year",
        "month",
        "dayofmonth",
        "weekday",
        "weekofyear",
        "dayofyear",
        "hour",
        "timestamp",
    }
