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
        schema=T.StructType([T.StructField("start_time", T.TimestampType(), True)]),
    )

    output = transformer.create_time_features(dataframe, "start_time")

    assert isinstance(output, DataFrame)
    assert set(output.columns) == {
        "year",
        "month",
        "dayofmonth",
        "weekday",
        "weekofyear",
        "dayofyear",
        "hour",
    }


def test_split_start_and_end_time(
    transformer: BronzeToSilverTransformer,
    spark: SparkSession,
):
    dataframe = spark.createDataFrame(
        [
            [
                "2024-06-19 19:24:11",
                "2024-06-19 19:35:37",
                1443109011456,
            ],
            [
                "2024-06-20 17:01:54",
                "2024-06-20 17:14:34",
                1443109011457,
            ],
        ],
        schema=T.StructType(
            [
                T.StructField("start_time", T.StringType(), True),
                T.StructField("end_time", T.StringType(), True),
                T.StructField("row_number", T.LongType(), True),
            ]
        ),
    )
    dataframe = dataframe.withColumn(
        "start_time", F.to_timestamp("start_time")
    ).withColumn("end_time", F.to_timestamp("end_time"))

    start_df, end_df = transformer.split_start_and_end_time(dataframe)

    assert isinstance(start_df, DataFrame)
    assert start_df.count() == dataframe.count()
    assert set(start_df.columns) == {"row_number", "start_time"}
    assert isinstance(end_df, DataFrame)
    assert end_df.count() == dataframe.count()
    assert set(end_df.columns) == {"row_number", "end_time"}
