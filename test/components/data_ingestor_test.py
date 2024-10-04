import pytest
from unittest.mock import Mock, patch
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame

from src.components.data_ingestor import DataIngestor, DataIngestorConfig


@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("local-tests")
        .config("spark.executor.cores", "1")
        .config("spark.executor.instances", "1")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    yield spark
    spark.stop()


@pytest.fixture
def ingestor(spark: SparkSession):
    return DataIngestor(spark)


@pytest.fixture
def dataframe_pre2020(spark: SparkSession, ingestor: DataIngestor):
    return spark.createDataFrame(
        data=[
            [
                219,
                "2019-02-01 00:00:06.2570",
                "2019-02-01 00:03:46.1090",
                3494.0,
                "E 115 St & Lexington Ave",
                40.79791259765625,
                -73.94229888916016,
                3501.0,
                "E 118 St & Madison Ave",
                40.80148696899414,
                -73.94425201416016,
                33450,
                "Subscriber",
                1989,
                1,
            ],
            [
                143,
                "2019-02-01 00:00:28.0320",
                "2019-02-01 00:02:51.7460",
                438.0,
                "St Marks Pl & 1 Ave",
                40.72779083251953,
                -73.98564910888672,
                236.0,
                "St Marks Pl & 2 Ave",
                40.72842025756836,
                -73.98713684082031,
                25626,
                "Subscriber",
                1990,
                1,
            ],
            [
                296,
                "2019-02-01 00:01:13.9870",
                "2019-02-01 00:06:10.7340",
                3571.0,
                "Bedford Ave & Bergen St",
                40.676368713378906,
                -73.95291900634766,
                3549.0,
                "Grand Ave & Bergen St",
                40.678043365478516,
                -73.96240997314453,
                35568,
                "Subscriber",
                1987,
                1,
            ],
            [
                478,
                "2019-02-01 00:01:14.1520",
                "2019-02-01 00:09:12.7870",
                167.0,
                "E 39 St & 3 Ave",
                40.7489013671875,
                -73.9760513305664,
                477.0,
                "W 41 St & 8 Ave",
                40.756404876708984,
                -73.99002838134766,
                25045,
                "Subscriber",
                1964,
                2,
            ],
            [
                225,
                "2019-02-01 00:01:49.3410",
                "2019-02-01 00:05:34.4980",
                3458.0,
                "W 55 St & 6 Ave",
                40.763092041015625,
                -73.97834777832031,
                3443.0,
                "W 52 St & 6 Ave",
                40.761329650878906,
                -73.97982025146484,
                34006,
                "Subscriber",
                1979,
                1,
            ],
        ],
        schema=ingestor.config.pre_2020_schema,
    )


@pytest.fixture
def dataframe_post2020(spark: SparkSession, ingestor: DataIngestor):
    return spark.createDataFrame(
        data=[
            [
                "26F472DB7812B0EF",
                "classic_bike",
                "2023-09-16 14:14:31.636",
                "2023-09-16 14:24:10.687",
                "E 68 St & Madison Ave",
                6932.14990234375,
                "E 84 St & Park Ave",
                7243.0400390625,
                40.76915740966797,
                -73.96703338623047,
                40.77862548828125,
                -73.95771789550781,
                "casual",
            ],
            [
                "6B14664F37A457AE",
                "classic_bike",
                "2023-09-16 19:36:49.310",
                "2023-09-16 19:59:20.831",
                "W 106 St & Central Park West",
                7606.009765625,
                "E 84 St & Park Ave",
                7243.0400390625,
                40.798187255859375,
                -73.9605941772461,
                40.77862548828125,
                -73.95771789550781,
                "casual",
            ],
            [
                "07E1D1C53A48CD9D",
                "classic_bike",
                "2023-09-16 15:01:48.745",
                "2023-09-16 16:08:48.989",
                "E 72 St & Park Ave",
                6998.080078125,
                "E 84 St & Park Ave",
                7243.0400390625,
                40.771183013916016,
                -73.96409606933594,
                40.77862548828125,
                -73.95771789550781,
                "casual",
            ],
            [
                "40DB62D1510D3221",
                "classic_bike",
                "2023-09-16 15:03:02.733",
                "2023-09-16 16:08:33.119",
                "E 72 St & Park Ave",
                6998.080078125,
                "E 84 St & Park Ave",
                7243.0400390625,
                40.771183013916016,
                -73.96409606933594,
                40.77862548828125,
                -73.95771789550781,
                "casual",
            ],
            [
                "BBD4714878DFE088",
                "classic_bike",
                "2023-09-16 15:03:14.158",
                "2023-09-16 16:08:40.688",
                "E 72 St & Park Ave",
                6998.080078125,
                "E 84 St & Park Ave",
                7243.0400390625,
                40.771183013916016,
                -73.96409606933594,
                40.77862548828125,
                -73.95771789550781,
                "casual",
            ],
        ],
        schema=ingestor.config.post_2020_schema,
    )


def test_config():
    config = DataIngestorConfig()

    assert hasattr(config, "root_data_path")
    assert hasattr(config, "root_csv_path")
    assert hasattr(config, "pre_2020_csv_dir")
    assert hasattr(config, "pre_2020_schema")
    assert hasattr(config, "post_2020_csv_dir")
    assert hasattr(config, "post_2020_schema")
    assert hasattr(config, "raw_delta_path")
    assert hasattr(config, "column_order")


def test_init(spark: SparkSession, ingestor: DataIngestor):
    assert hasattr(ingestor, "config")
    assert hasattr(ingestor, "spark")

    assert isinstance(ingestor.config, DataIngestorConfig)
    assert isinstance(ingestor.spark, SparkSession)
    assert ingestor.spark is spark


def test_read_csv(dataframe_pre2020: DataFrame):
    spark = Mock(SparkSession)
    spark.read.csv.return_value = dataframe_pre2020

    ingestor = DataIngestor(spark)
    df = ingestor.read_csv(
        ingestor.config.pre_2020_csv_dir, ingestor.config.pre_2020_schema
    )

    spark.read.csv.assert_called_once_with(
        ingestor.config.pre_2020_csv_dir,
        schema=ingestor.config.pre_2020_schema,
        header=True,
    )
    assert df is dataframe_pre2020


def test_standardize_columns_pre2020(
    dataframe_pre2020: DataFrame,
    ingestor: DataIngestor,
):
    df = ingestor.standardize_columns_for_pre2020(dataframe_pre2020)

    assert df.columns == ingestor.config.column_order


def test_standardize_columns_post2020(
    dataframe_post2020: DataFrame,
    ingestor: DataIngestor,
):
    df = ingestor.standardize_columns_for_post2020(dataframe_post2020)

    assert df.columns == ingestor.config.column_order


def test_combine_dataframes(
    dataframe_pre2020: DataFrame,
    dataframe_post2020: DataFrame,
    ingestor: DataIngestor,
):
    expected_columns = set(
        [
            "start_time",
            "end_time",
            "start_station_name",
            "start_station_latitude",
            "start_station_longitude",
            "end_station_name",
            "end_station_latitude",
            "end_station_longitude",
            "start_station_id",
            "end_station_id",
            "member",
            "row_number",
            "file_path",
        ]
    )

    dataframe_pre2020 = ingestor.standardize_columns_for_pre2020(dataframe_pre2020)
    dataframe_post2020 = ingestor.standardize_columns_for_post2020(dataframe_post2020)
    df = ingestor.combine_dataframes(dataframe_pre2020, dataframe_post2020)

    assert set(df.columns) == expected_columns
