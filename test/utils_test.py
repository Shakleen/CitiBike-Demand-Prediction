import pytest
from unittest.mock import Mock, patch
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame

from src.utils import read_delta, write_delta


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


@pytest.mark.parametrize("path", ["path/to/test", "path\\to\\test"])
def test_read_delta(path: str):
    spark_mock = Mock(SparkSession)
    dataframe_mock = Mock(DataFrame)

    spark_mock.read.format("delta").load.return_value = dataframe_mock

    df = read_delta(spark_mock, path)

    spark_mock.read.format.assert_called_with("delta")
    spark_mock.read.format("delta").load.assert_called_with(path)

    assert df is dataframe_mock


@pytest.mark.parametrize("path", ["path/to/test", "path\\to\\test"])
def test_write_delta(path: str):
    dataframe_mock = Mock(DataFrame)

    write_delta(dataframe_mock, path)

    dataframe_mock.write.save.assert_called_once_with(
        path=path,
        format="delta",
        mode="overwrite",
    )
