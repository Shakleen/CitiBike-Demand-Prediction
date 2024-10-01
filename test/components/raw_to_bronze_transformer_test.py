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

schema = StructType(
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
def dataframe_2(spark: SparkSession):
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
                5406,
                5414,
                1,
                1666447310848,
                "file:///media/ishrak/New%20Volume/Studies/Projects/CitiBike-Demand-Prediction/Data/CSVs/post_2020/202101-citibike-tripdata_1.csv",
                "202101-citibike-tripdata_1.csv",
            ],
            [
                "2021-01-29 06:38:32.423",
                "2021-01-29 06:40:28.603",
                "Clark St & Henry St",
                40.697601318359375,
                -73.99344635009766,
                "Columbia Heights & Cranberry St",
                40.70037841796875,
                -73.9954833984375,
                4789,
                4829,
                1,
                1666447310849,
                "file:///media/ishrak/New%20Volume/Studies/Projects/CitiBike-Demand-Prediction/Data/CSVs/post_2020/202101-citibike-tripdata_1.csv",
                "202101-citibike-tripdata_1.csv",
            ],
            [
                "1/1/2015 0:01",
                "1/1/2015 0:24",
                "Rivington St & Ridge St",
                40.718502044677734,
                -73.9832992553711,
                "Allen St & Rivington St",
                40.72019577026367,
                -73.98997497558594,
                5406,
                5414,
                1,
                1666447310848,
                "file:///media/ishrak/New%20Volume/Studies/Projects/CitiBike-Demand-Prediction/Data/CSVs/post_2020/202101-citibike-tripdata_1.csv",
                "201501-citibike-tripdata_1.csv",
            ],
            [
                "1/1/2015 0:02",
                "1/1/2015 0:08",
                "Clark St & Henry St",
                40.697601318359375,
                -73.99344635009766,
                "Columbia Heights & Cranberry St",
                40.70037841796875,
                -73.9954833984375,
                4789,
                4829,
                1,
                1666447310849,
                "file:///media/ishrak/New%20Volume/Studies/Projects/CitiBike-Demand-Prediction/Data/CSVs/post_2020/202101-citibike-tripdata_1.csv",
                "201501-citibike-tripdata_1.csv",
            ],
            [
                "9/1/2014 00:00:25",
                "9/1/2014 00:00:25",
                "Rivington St & Ridge St",
                40.718502044677734,
                -73.9832992553711,
                "Allen St & Rivington St",
                40.72019577026367,
                -73.98997497558594,
                5406,
                5414,
                1,
                1666447310848,
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
                40.70037841796875,
                -73.9954833984375,
                4789,
                4829,
                1,
                1666447310849,
                "file:///media/ishrak/New%20Volume/Studies/Projects/CitiBike-Demand-Prediction/Data/CSVs/post_2020/202101-citibike-tripdata_1.csv",
                "201409-citibike-tripdata_1.csv",
            ],
        ],
        schema=schema,
    )


@pytest.fixture
def dataframe(spark: SparkSession):
    return spark.createDataFrame(
        data=[
            [
                "2019-08-01 00:00:01.4680",
                "2019-08-01 00:06:35.3780",
                "Forsyth St & Broome St",
                40.71894073486328,
                -73.99266052246094,
                "Market St & Cherry St",
                40.71076202392578,
                -73.99400329589844,
                531,
                408,
                1,
                85899345920,
                "file:///media/ishrak/New%20Volume/Studies/Projects/CitiBike-Demand-Prediction/Data/CSVs/pre_2020/201908-citibike-tripdata_1.csv",
            ],
            [
                "2019-08-01 00:00:01.9290",
                "2019-08-01 00:10:29.7840",
                "Lafayette Ave & Fort Greene Pl",
                40.686920166015625,
                -73.9766845703125,
                "Bergen St & Smith St",
                40.686744689941406,
                -73.99063110351562,
                274,
                3409,
                1,
                85899345921,
                "file:///media/ishrak/New%20Volume/Studies/Projects/CitiBike-Demand-Prediction/Data/CSVs/pre_2020/201908-citibike-tripdata_1.csv",
            ],
            [
                "2019-08-01 00:00:04.0480",
                "2019-08-01 00:18:56.1650",
                "Front St & Washington St",
                40.70254898071289,
                -73.9894027709961,
                "President St & Henry St",
                40.68280029296875,
                -73.9999008178711,
                2000,
                3388,
                1,
                85899345922,
                "file:///media/ishrak/New%20Volume/Studies/Projects/CitiBike-Demand-Prediction/Data/CSVs/pre_2020/201908-citibike-tripdata_1.csv",
            ],
            [
                "2019-08-01 00:00:04.1630",
                "2019-08-01 00:29:44.7940",
                "9 Ave & W 45 St",
                40.76019287109375,
                -73.99125671386719,
                "Rivington St & Chrystie St",
                40.721099853515625,
                -73.99192810058594,
                479,
                473,
                1,
                85899345923,
                "file:///media/ishrak/New%20Volume/Studies/Projects/CitiBike-Demand-Prediction/Data/CSVs/pre_2020/201908-citibike-tripdata_1.csv",
            ],
            [
                "2019-08-01 00:00:05.4580",
                "2019-08-01 00:25:23.4550",
                "1 Ave & E 94 St",
                40.78172302246094,
                -73.94593811035156,
                "1 Ave & E 94 St",
                40.78172302246094,
                -73.94593811035156,
                3312,
                3312,
                1,
                85899345924,
                "file:///media/ishrak/New%20Volume/Studies/Projects/CitiBike-Demand-Prediction/Data/CSVs/pre_2020/201908-citibike-tripdata_1.csv",
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
                ]
            )
        ),
    )


def test_config():
    config = RawToBronzeTransformerConfig()

    assert hasattr(config, "root_delta_path")
    assert hasattr(config, "raw_data_path")
    assert hasattr(config, "bronze_data_path")


def test_init(transformer: RawToBronzeTransformer, spark: SparkSession):
    assert hasattr(transformer, "config")
    assert hasattr(transformer, "spark")
    assert transformer.spark is spark
    assert isinstance(transformer.config, RawToBronzeTransformerConfig)
    assert isinstance(transformer.spark, SparkSession)


def test_read_raw_delta(dataframe: DataFrame):
    spark_mock = Mock(SparkSession)
    transformer = RawToBronzeTransformer(spark_mock)

    spark_mock.read.format("delta").load.return_value = dataframe

    df = transformer.read_raw_delta()

    spark_mock.read.format.assert_called_with("delta")
    spark_mock.read.format("delta").load.assert_called_with(
        transformer.config.raw_data_path
    )

    assert df is dataframe


def test_write_delta():
    dataframe = Mock(DataFrame)
    spark_mock = Mock(SparkSession)
    transformer = RawToBronzeTransformer(spark_mock)

    transformer.write_delta(dataframe)

    dataframe.write.save.assert_called_once_with(
        path=transformer.config.bronze_data_path, format="delta", mode="overwrite"
    )


def test_create_file_name_column(
    transformer: RawToBronzeTransformer,
    dataframe: DataFrame,
):
    df = transformer.create_file_name_column(dataframe)

    assert "file_name" in df.columns


def test_get_dataframe_timeformat_type_1(
    dataframe_2: DataFrame,
    transformer: RawToBronzeTransformer,
):
    output = transformer.get_dataframe_timeformat_type_1(dataframe_2)

    assert isinstance(output, DataFrame)
    assert output.count() == 2


def test_get_dataframe_timeformat_type_2(
    dataframe_2: DataFrame,
    transformer: RawToBronzeTransformer,
):
    output = transformer.get_dataframe_timeformat_type_2(dataframe_2)

    assert isinstance(output, DataFrame)
    assert output.count() == 2


def test_get_dataframe_timeformat_type_3(
    dataframe_2: DataFrame,
    transformer: RawToBronzeTransformer,
):
    output = transformer.get_dataframe_timeformat_type_3(dataframe_2)

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
    dataframe_2: DataFrame,
    transformer: RawToBronzeTransformer,
    time_format: str,
    count: int,
):
    output = transformer.set_timestamp_for_format(dataframe_2, time_format)

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
    dataframe_2: DataFrame,
    transformer: RawToBronzeTransformer,
):
    output = transformer.set_timestamp_datatype(dataframe_2)

    assert isinstance(output, DataFrame)
    assert output.schema[0] == StructField("start_time", TimestampType(), True)
    assert output.schema[1] == StructField("end_time", TimestampType(), True)
    assert (
        output.filter(col("start_time").isNotNull())
        .filter(col("end_time").isNotNull())
        .count()
        == dataframe_2.count()
    )


def test_get_station_dataframe(
    dataframe_2: DataFrame,
    transformer: RawToBronzeTransformer,
):
    output = transformer.get_station_dataframe(dataframe_2)

    assert isinstance(output, DataFrame)
    assert output.columns == ["id", "name", "latitude", "longitude"]
