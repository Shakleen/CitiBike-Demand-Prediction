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
    df.select.return_value = df
    return df


@pytest.fixture(scope="session")
def spark_mock(dataframe_mock):
    spark = Mock(SparkSession)
    spark.read = spark
    spark.read.format = spark
    spark.read.format.load = spark
    return spark


@pytest.fixture()
def transformer(spark_mock):
    return RawToBronzeTransformer(spark_mock)


def test_config():
    config = RawToBronzeTransformerConfig()

    assert hasattr(config, "root_delta_path")
    assert hasattr(config, "raw_data_path")
    assert hasattr(config, "bronze_data_path")


def test_init(spark_mock: SparkSession, transformer: RawToBronzeTransformer):
    assert transformer.config == RawToBronzeTransformerConfig()
    assert transformer.spark is spark_mock


def test_read_raw_delta(spark_mock: SparkSession, transformer: RawToBronzeTransformer):
    _ = transformer.read_raw_delta()
    spark_mock.read.assert_called_once()
    spark_mock.read.format.assert_called_once_with("delta")
    spark_mock.read.format.load.assert_called_once()


def test_write_delta(dataframe_mock, transformer: RawToBronzeTransformer):
    write_mock = Mock()
    dataframe_mock.write = write_mock

    transformer.write_delta(dataframe_mock)

    write_mock.save.assert_called_once_with(
        path=transformer.config.bronze_data_path,
        format="delta",
        mode="overwrite",
    )

def test_create_file_name_column(mocker, dataframe_mock, transformer: RawToBronzeTransformer):
    reg_exp_mock = Mock()
    mocker.patch("src.components.raw_to_bronze_transformer.col", return_value=Mock())
    mocker.patch("src.components.raw_to_bronze_transformer.regexp_extract", return_value=reg_exp_mock)
    
    output = transformer.create_file_name_column(dataframe_mock)
    
    dataframe_mock.withColumn.assert_called_once_with("file_name", reg_exp_mock)
    assert output == dataframe_mock
