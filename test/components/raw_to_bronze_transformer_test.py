import pytest
from unittest.mock import Mock, patch
from pyspark.sql import SparkSession

from src.components.raw_to_bronze_transformer import (
    RawToBronzeTransformerConfig,
    RawToBronzeTransformer,
)


@pytest.fixture(scope="session")
def dataframe_mock():
    df = Mock(name="dataframe")
    df.withColumnRenamed.return_value = df
    df.withColumn.return_value = df
    df.drop.return_value = df
    df.union.return_value = df
    df.write.return_value = df
    df.save.return_value = df
    return df


@pytest.fixture(scope="session")
def spark_mock(dataframe_mock):
    spark = Mock(SparkSession)
    spark.read = Mock()
    spark.read.csv = dataframe_mock
    return spark


@pytest.fixture()
def transformer(spark_mock):
    return RawToBronzeTransformer(spark_mock)


def test_config():
    config = RawToBronzeTransformerConfig()

    assert hasattr(config, "root_delta_path")
    assert hasattr(config, "raw_data_path")
    assert hasattr(config, "bronze_data_path")


def test_init(spark_mock, transformer: RawToBronzeTransformer):
    assert transformer.config == RawToBronzeTransformerConfig()
    assert transformer.spark is spark_mock
