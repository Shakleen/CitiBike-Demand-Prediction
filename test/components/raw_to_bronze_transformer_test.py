import pytest
from unittest.mock import Mock, patch
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructField,
    StructType,
    LongType,
    IntegerType,
    StringType,
    FloatType,
    TimestampType,
)
from pyspark.sql.functions import col
from pyspark.sql.dataframe import DataFrame

from src.components.raw_to_bronze_transformer import (
    RawToBronzeTransformerConfig,
    RawToBronzeTransformer,
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
    return RawToBronzeTransformer(spark)


@pytest.fixture
def dataframe(spark: SparkSession):
    return spark.createDataFrame(
        data=[
            [
                "2021-01-19 19:43:36.986",
                "2021-01-19 19:45:50.414",
                "Rivington St & Ridge St",
                40.718502044677734,
                -73.9832992553711,
                "Allen St & Rivington St",
                40.72019577026367,
                -73.98997497558594,
                None,
                5414,
                1,
                1,
                "file:///media/ishrak/New%20Volume/Studies/Projects/CitiBike-Demand-Prediction/Data/CSVs/post_2020/202101-citibike-tripdata_1.csv",
                "202101-citibike-tripdata_1.csv",
            ],
            [
                "2021-01-29 06:38:32.423",
                "2021-01-29 06:40:28.603",
                "Clark St & Henry St",
                40.697601318359375,
                -73.99344635009766,
                None,
                40.70037841796875,
                -73.9954833984375,
                4789,
                4829,
                1,
                2,
                "file:///media/ishrak/New%20Volume/Studies/Projects/CitiBike-Demand-Prediction/Data/CSVs/post_2020/202101-citibike-tripdata_1.csv",
                "202101-citibike-tripdata_1.csv",
            ],
            [
                "1/1/2015 0:01",
                "1/1/2015 0:24",
                "Rivington St & Ridge St",
                None,
                None,
                "Allen St & Rivington St",
                40.72019577026367,
                -73.98997497558594,
                5406,
                5414,
                1,
                3,
                "file:///media/ishrak/New%20Volume/Studies/Projects/CitiBike-Demand-Prediction/Data/CSVs/post_2020/202101-citibike-tripdata_1.csv",
                "201501-citibike-tripdata_1.csv",
            ],
            [
                "1/1/2015 0:02",
                "1/1/2015 0:08",
                None,
                40.697601318359375,
                -73.99344635009766,
                "Columbia Heights & Cranberry St",
                40.70037841796875,
                -73.9954833984375,
                4789,
                4829,
                1,
                4,
                "file:///media/ishrak/New%20Volume/Studies/Projects/CitiBike-Demand-Prediction/Data/CSVs/post_2020/202101-citibike-tripdata_1.csv",
                "201501-citibike-tripdata_1.csv",
            ],
            [
                "9/1/2014 00:00:25",
                "9/1/2014 00:00:25",
                "Rivington St & Ridge St",
                40.718502044677734,
                -73.9832992553711,
                None,
                None,
                None,
                5406,
                None,
                1,
                5,
                "file:///media/ishrak/New%20Volume/Studies/Projects/CitiBike-Demand-Prediction/Data/CSVs/post_2020/202101-citibike-tripdata_1.csv",
                "201409-citibike-tripdata_1.csv",
            ],
            [
                "9/1/2014 00:00:28",
                "9/1/2014 00:00:28",
                "Clark St & Henry St",
                40.697601318359375,
                -73.99344635009766,
                "Columbia Heights & Cranberry St",
                None,
                None,
                4789,
                4829,
                1,
                6,
                "file:///media/ishrak/New%20Volume/Studies/Projects/CitiBike-Demand-Prediction/Data/CSVs/post_2020/202101-citibike-tripdata_1.csv",
                "201409-citibike-tripdata_1.csv",
            ],
        ],
        schema=StructType(
            StructType(
                [
                    StructField("start_time", StringType(), True),
                    StructField("end_time", StringType(), True),
                    StructField("start_station_name", StringType(), True),
                    StructField("start_station_latitude", FloatType(), True),
                    StructField("start_station_longitude", FloatType(), True),
                    StructField("end_station_name", StringType(), True),
                    StructField("end_station_latitude", FloatType(), True),
                    StructField("end_station_longitude", FloatType(), True),
                    StructField("start_station_id", IntegerType(), True),
                    StructField("end_station_id", IntegerType(), True),
                    StructField("member", IntegerType(), True),
                    StructField("row_number", LongType(), True),
                    StructField("file_path", StringType(), True),
                    StructField("file_name", StringType(), True),
                ]
            )
        ),
    )


def test_config():
    config = RawToBronzeTransformerConfig()

    assert hasattr(config, "root_delta_path")
    assert hasattr(config, "raw_data_path")
    assert hasattr(config, "bronze_data_path")
    assert hasattr(config, "station_data_path")


def test_init(transformer: RawToBronzeTransformer, spark: SparkSession):
    assert hasattr(transformer, "config")
    assert hasattr(transformer, "spark")
    assert transformer.spark is spark
    assert isinstance(transformer.config, RawToBronzeTransformerConfig)
    assert isinstance(transformer.spark, SparkSession)


def test_read_raw_delta():
    spark_mock = Mock(SparkSession)
    transformer = RawToBronzeTransformer(spark_mock)
    dataframe_mock = Mock(DataFrame)

    spark_mock.read.format("delta").load.return_value = dataframe_mock

    df = transformer.read_raw_delta()

    spark_mock.read.format.assert_called_with("delta")
    spark_mock.read.format("delta").load.assert_called_with(
        transformer.config.raw_data_path
    )

    assert df is dataframe_mock


def test_write_delta():
    dataframe = Mock(DataFrame)
    spark_mock = Mock(SparkSession)
    transformer = RawToBronzeTransformer(spark_mock)

    transformer.write_delta(dataframe, transformer.config.bronze_data_path)

    dataframe.write.save.assert_called_once_with(
        path=transformer.config.bronze_data_path,
        format="delta",
        mode="overwrite",
    )


def test_create_file_name_column(
    transformer: RawToBronzeTransformer,
    dataframe: DataFrame,
):
    df = transformer.create_file_name_column(dataframe)

    assert "file_name" in df.columns


def test_get_dataframe_timeformat_type_1(
    dataframe: DataFrame,
    transformer: RawToBronzeTransformer,
):
    output = transformer.get_dataframe_timeformat_type_1(dataframe)

    assert isinstance(output, DataFrame)
    assert output.count() == 2


def test_get_dataframe_timeformat_type_2(
    dataframe: DataFrame,
    transformer: RawToBronzeTransformer,
):
    output = transformer.get_dataframe_timeformat_type_2(dataframe)

    assert isinstance(output, DataFrame)
    assert output.count() == 2


def test_get_dataframe_timeformat_type_3(
    dataframe: DataFrame,
    transformer: RawToBronzeTransformer,
):
    output = transformer.get_dataframe_timeformat_type_3(dataframe)

    assert isinstance(output, DataFrame)
    assert output.count() == 2


@pytest.mark.parametrize(
    ("time_format", "count"),
    [
        ("yyyy-MM-dd HH:mm:ss", 2),
        ("M/d/yyyy H:mm", 4),
        ("M/d/yyyy HH:mm:ss", 2),
    ],
)
def test_set_timestamp_for_format(
    dataframe: DataFrame,
    transformer: RawToBronzeTransformer,
    time_format: str,
    count: int,
):
    output = transformer.set_timestamp_for_format(dataframe, time_format)

    assert isinstance(output, DataFrame)
    assert output.schema[0] == StructField("start_time", TimestampType(), True)
    assert output.schema[1] == StructField("end_time", TimestampType(), True)
    assert (
        output.filter(col("start_time").isNotNull())
        .filter(col("end_time").isNotNull())
        .count()
        == count
    )


def test_set_timestamp_datatype(
    dataframe: DataFrame,
    transformer: RawToBronzeTransformer,
):
    output = transformer.set_timestamp_datatype(dataframe)

    assert isinstance(output, DataFrame)
    assert output.schema[0] == StructField("start_time", TimestampType(), True)
    assert output.schema[1] == StructField("end_time", TimestampType(), True)
    assert (
        output.filter(col("start_time").isNotNull())
        .filter(col("end_time").isNotNull())
        .count()
        == dataframe.count()
    )


def test_get_station_dataframe(
    dataframe: DataFrame,
    transformer: RawToBronzeTransformer,
):
    output = transformer.get_station_dataframe(dataframe)

    assert isinstance(output, DataFrame)
    assert output.columns == ["id", "name", "latitude", "longitude"]


def test_split_station_and_time(
    dataframe: DataFrame,
    transformer: RawToBronzeTransformer,
):
    station_df, df = transformer.split_station_and_time(dataframe)

    assert isinstance(station_df, DataFrame)
    assert isinstance(df, DataFrame)
    assert set(station_df.columns) == {"id", "name", "latitude", "longitude"}
    assert set(df.columns) == {"start_time", "end_time", "row_number", "start_station_id", "end_station_id"}
