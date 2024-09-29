import pytest
from unittest.mock import Mock, patch
from pyspark.sql import SparkSession

from src.components.data_ingestor import DataIngestor, DataIngestorConfig


@pytest.fixture(scope="session")
def mock_dataframe():
    df = Mock(name="dataframe")
    df.withColumnRenamed = df
    df.withColumn = df
    df.drop = df
    return df


@pytest.fixture(scope="session")
def spark_mock(mock_dataframe):
    spark = Mock(SparkSession)
    spark.read = Mock()
    spark.read.csv = mock_dataframe
    return spark


@pytest.fixture()
def ingestor(spark_mock):
    return DataIngestor(spark_mock)


def test_config(ingestor: DataIngestor):
    config = DataIngestorConfig()
    assert ingestor.config == config


def test_read_csv(spark_mock: SparkSession, ingestor: DataIngestor):
    df = ingestor.read_csv(
        ingestor.config.pre_2020_csv_dir,
        ingestor.config.pre_2020_schema,
    )
    spark_mock.read.csv.assert_called_once_with(
        ingestor.config.pre_2020_csv_dir,
        schema=ingestor.config.pre_2020_schema,
        header=True,
    )
    assert df is spark_mock.read.csv.return_value


def test_read_pre_and_post_csv_dir(spark_mock: SparkSession, ingestor: DataIngestor):
    output = ingestor.read_pre_and_post_csv_dir()
    assert output[0] is spark_mock.read.csv.return_value
    assert output[1] is spark_mock.read.csv.return_value
