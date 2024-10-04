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


@pytest.fixture
def time_dataframe(spark: SparkSession):
    dataframe = spark.createDataFrame(
        [
            [
                "2024-06-19 19:24:11",
                "2024-06-19 19:35:37",
                1,
            ],
            [
                "2024-06-20 17:01:54",
                "2024-06-20 17:14:34",
                2,
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

    return dataframe


@pytest.fixture
def mapper_dataframe(spark: SparkSession):
    dataframe = spark.createDataFrame(
        [
            [1, 10, 20],
            [2, 20, 10],
        ],
        schema=T.StructType(
            [
                T.StructField("row_number", T.LongType(), True),
                T.StructField("start_station_id", T.IntegerType(), True),
                T.StructField("end_station_id", T.IntegerType(), True),
            ]
        ),
    )

    return dataframe


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
    time_dataframe: DataFrame,
):
    start_df, end_df = transformer.split_start_and_end_time(time_dataframe)

    assert isinstance(start_df, DataFrame)
    assert start_df.count() == time_dataframe.count()
    assert set(start_df.columns) == {"row_number", "time"}
    assert isinstance(end_df, DataFrame)
    assert end_df.count() == time_dataframe.count()
    assert set(end_df.columns) == {"row_number", "time"}


def test_attach_station_ids(
    transformer: BronzeToSilverTransformer,
    time_dataframe: DataFrame,
    mapper_dataframe: DataFrame,
):
    start_df, end_df = transformer.split_start_and_end_time(time_dataframe)
    start_df, end_df = transformer.attach_station_ids(
        start_df, end_df, mapper_dataframe
    )

    assert isinstance(start_df, DataFrame)
    assert start_df.count() == time_dataframe.count()
    assert set(start_df.columns) == {"row_number", "time", "station_id"}
    assert isinstance(end_df, DataFrame)
    assert end_df.count() == time_dataframe.count()
    assert set(end_df.columns) == {"row_number", "time", "station_id"}


def test_count_group_by_station_and_time(
    transformer: BronzeToSilverTransformer,
    spark: SparkSession,
):
    dataframe = spark.createDataFrame(
        [
            [1, "2024-06-19 19:24:11", 10],
            [2, "2024-06-20 17:01:54", 10],
            [3, "2024-06-20 17:01:54", 10],
            [4, "2024-06-21 17:01:54", 20],
        ],
        schema=T.StructType(
            [
                T.StructField("row_number", T.LongType(), True),
                T.StructField("time", T.StringType(), True),
                T.StructField("station_id", T.IntegerType(), True),
            ]
        ),
    )
    dataframe = dataframe.withColumn("time", F.to_timestamp("time"))

    output = transformer.count_group_by_station_and_time(dataframe)

    assert isinstance(output, DataFrame)
    assert output.count() == 3
    assert set(output.columns) == {"station_id", "time", "count"}
    assert output.select("count").toPandas().to_numpy().flatten().tolist() == [1, 2, 1]

