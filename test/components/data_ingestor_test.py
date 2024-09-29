import pytest
from unittest.mock import Mock, patch
from pyspark.sql import SparkSession

from src.components.data_ingestor import DataIngestor, DataIngestorConfig


@pytest.fixture(scope="session")
def spark_mock():
    spark = Mock(SparkSession)
    spark.read = Mock()
    spark.read.csv = Mock()
    return spark


@pytest.fixture()
def ingestor(spark_mock):
    return DataIngestor(spark_mock)


def test_config(ingestor: DataIngestor):
    config = DataIngestorConfig()
    assert ingestor.config == config


def test_read_csv(spark_mock, ingestor: DataIngestor):
    df = ingestor.read_csv(spark_mock, "path/to/csv", ingestor.config.pre_2020_schema)
    spark_mock.read.csv.assert_called_once_with(
        "path/to/csv",
        schema=ingestor.config.pre_2020_schema,
        header=True,
    )
