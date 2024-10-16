import pytest
from unittest.mock import Mock, patch
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
import pyspark.sql.types as T
import pyspark.sql.functions as F
import pandas as pd

from src.data_pipeline.bronze_to_silver_transformer import (
    BronzeToSilverTransformerConfig,
    BronzeToSilverTransformer,
    create_time_features,
)

output_schema = T.StructType(
    [
        T.StructField("station_id", T.IntegerType(), True),
        T.StructField("bike_demand", T.IntegerType(), True),
        T.StructField("dock_demand", T.IntegerType(), True),
        T.StructField("year", T.IntegerType(), True),
        T.StructField("month", T.IntegerType(), True),
        T.StructField("dayofmonth", T.IntegerType(), True),
        T.StructField("weekday", T.IntegerType(), True),
        T.StructField("weekofyear", T.IntegerType(), True),
        T.StructField("dayofyear", T.IntegerType(), True),
        T.StructField("hour", T.IntegerType(), True),
        T.StructField("is_holiday", T.BooleanType(), True),
    ]
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
            ["2024-06-19 19:24:11", "2024-06-19 19:35:37", 1, 1, 2],
            ["2024-06-20 17:01:54", "2024-06-20 17:14:34", 2, 2, 1],
        ],
        schema=T.StructType(
            [
                T.StructField("start_time", T.StringType(), True),
                T.StructField("end_time", T.StringType(), True),
                T.StructField("row_number", T.LongType(), True),
                T.StructField("start_station_id", T.IntegerType(), True),
                T.StructField("end_station_id", T.IntegerType(), True),
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


@pytest.fixture
def demand_dataframe(spark: SparkSession):
    data = [
        [1, 100, 100, 2024, month, dayofmonth, weekday, 25, 171, 19, False]
        for month in range(1, 13, 1)
        for weekday in range(7)
        for dayofmonth in range(1, 32, 3)
    ]
    df = spark.createDataFrame(data, output_schema)
    return df


def test_config():
    config = BronzeToSilverTransformerConfig()

    assert hasattr(config, "root_delta_path")
    assert hasattr(config, "bronze_delta_path")
    assert hasattr(config, "station_delta_path")
    assert hasattr(config, "silver_delta_path")


def test_init(transformer: BronzeToSilverTransformer):
    assert hasattr(transformer, "config")
    assert isinstance(transformer.config, BronzeToSilverTransformerConfig)
    assert hasattr(transformer, "spark")
    assert isinstance(transformer.spark, SparkSession)


def test_create_time_features(
    spark: SparkSession,
):
    dataframe = spark.createDataFrame(
        [pd.Timestamp("2024-06-19 19:24:11"), pd.Timestamp("2024-06-19 19:35:37")],
        schema=T.StructType([T.StructField("time", T.TimestampType(), True)]),
    )

    output = create_time_features(dataframe)

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
    start_df, end_df = transformer.split_start_and_end(time_dataframe)

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


def test_combine_on_station_id_and_time(
    transformer: BronzeToSilverTransformer,
    spark: SparkSession,
):
    start_df = spark.createDataFrame(
        [
            [1, "2024-06-19 19:00:00", 100],
            [2, "2024-06-20 17:00:00", 200],
            [3, "2024-06-21 17:00:00", 300],
        ],
        schema=T.StructType(
            [
                T.StructField("station_id", T.IntegerType(), True),
                T.StructField("time", T.StringType(), True),
                T.StructField("count", T.IntegerType(), True),
            ]
        ),
    )
    start_df = start_df.withColumn("time", F.to_timestamp("time"))

    end_df = spark.createDataFrame(
        [
            [1, "2024-06-19 19:00:00", 100],
            [2, "2024-06-20 17:00:00", 200],
            [4, "2024-06-21 17:00:00", 400],
        ],
        schema=T.StructType(
            [
                T.StructField("station_id", T.IntegerType(), True),
                T.StructField("time", T.StringType(), True),
                T.StructField("count", T.IntegerType(), True),
            ]
        ),
    )
    end_df = end_df.withColumn("time", F.to_timestamp("time"))

    output = transformer.combine_on_station_id_and_time(start_df, end_df)

    assert isinstance(output, DataFrame)
    assert output.count() == 4
    assert set(output.columns) == {"station_id", "time", "bike_demand", "dock_demand"}
    assert output.select("bike_demand").toPandas().to_numpy().flatten().tolist() == [
        100,
        200,
        300,
        0,
    ]
    assert output.select("dock_demand").toPandas().to_numpy().flatten().tolist() == [
        100,
        200,
        0,
        400,
    ]


def test_holiday_weekend(
    transformer: BronzeToSilverTransformer,
    demand_dataframe: DataFrame,
):
    expected = [
        weekday > 4
        for _ in range(1, 13, 1)
        for weekday in range(7)
        for _ in range(1, 32, 3)
    ]

    output = transformer.holiday_weekend(demand_dataframe)

    assert (
        output.select("is_holiday").toPandas().to_numpy().flatten().tolist() == expected
    )


def test_holiday_MLK_and_presidents_day(
    transformer: BronzeToSilverTransformer,
    demand_dataframe: DataFrame,
):
    expected = [
        month < 3 and weekday == 0 and 15 <= dayofmonth <= 21
        for month in range(1, 13, 1)
        for weekday in range(7)
        for dayofmonth in range(1, 32, 3)
    ]

    output = transformer.holiday_MLK_and_presidents_day(demand_dataframe)

    assert (
        output.select("is_holiday").toPandas().to_numpy().flatten().tolist() == expected
    )


def test_holiday_labor(
    transformer: BronzeToSilverTransformer,
    demand_dataframe: DataFrame,
):
    expected = [
        month == 9 and weekday == 0 and dayofmonth <= 7
        for month in range(1, 13, 1)
        for weekday in range(7)
        for dayofmonth in range(1, 32, 3)
    ]

    output = transformer.holiday_labor(demand_dataframe)

    assert (
        output.select("is_holiday").toPandas().to_numpy().flatten().tolist() == expected
    )


def test_holiday_columbus(
    transformer: BronzeToSilverTransformer,
    demand_dataframe: DataFrame,
):
    expected = [
        month == 10 and weekday == 0 and 7 < dayofmonth <= 14
        for month in range(1, 13, 1)
        for weekday in range(7)
        for dayofmonth in range(1, 32, 3)
    ]

    output = transformer.holiday_columbus(demand_dataframe)

    assert (
        output.select("is_holiday").toPandas().to_numpy().flatten().tolist() == expected
    )


def test_holiday_thanksgiving(
    transformer: BronzeToSilverTransformer,
    demand_dataframe: DataFrame,
):
    expected = [
        month == 11 and weekday == 3 and dayofmonth >= 22
        for month in range(1, 13, 1)
        for weekday in range(7)
        for dayofmonth in range(1, 32, 3)
    ]

    output = transformer.holiday_thanksgiving(demand_dataframe)

    assert (
        output.select("is_holiday").toPandas().to_numpy().flatten().tolist() == expected
    )


def test_add_station_coordinates(
    transformer: BronzeToSilverTransformer,
    spark: SparkSession,
):
    station_df = spark.createDataFrame(
        data=[
            [1, "W 52 St & 11 Ave", 40.76727294921875, -73.99392700195312],
            [2, "Franklin St & W Broadway", 40.7191162109375, -74.00666809082031],
            [3, "St James Pl & Pearl St", 40.71117401123047, -74.00016784667969],
            [4, "Atlantic Ave & Fort Greene Pl", 40.6838264465332, -73.97632598876953],
            [5, "W 17 St & 8 Ave", 40.74177551269531, -74.00149536132812],
        ],
        schema=T.StructType(
            [
                T.StructField("id", T.IntegerType(), True),
                T.StructField("name", T.StringType(), True),
                T.StructField("latitude", T.FloatType(), True),
                T.StructField("longitude", T.FloatType(), True),
            ]
        ),
    )

    df = spark.createDataFrame(
        data=[
            [1, 100, 100, 2024, 1, 1, 1, 25, 171, 19, False],
            [2, 100, 100, 2024, 1, 1, 1, 25, 171, 19, False],
            [3, 100, 100, 2024, 1, 1, 1, 25, 171, 19, False],
            [4, 100, 100, 2024, 1, 1, 1, 25, 171, 19, False],
            [5, 100, 100, 2024, 1, 1, 1, 25, 171, 19, False],
        ],
        schema=output_schema,
    )

    output = transformer.add_station_coordinates(df, station_df)

    assert set(output.columns) == {
        "bike_demand",
        "dock_demand",
        "year",
        "month",
        "dayofmonth",
        "weekday",
        "weekofyear",
        "dayofyear",
        "hour",
        "is_holiday",
        "latitude",
        "longitude",
    }
