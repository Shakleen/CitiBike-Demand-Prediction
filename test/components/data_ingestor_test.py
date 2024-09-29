import pytest
from unittest.mock import Mock, patch
from pyspark.sql import SparkSession

from src.components.data_ingestor import DataIngestor, DataIngestorConfig


@pytest.fixture(scope="session")
def mock_dataframe():
    df = Mock(name="dataframe")
    df.withColumnRenamed.return_value = df
    df.withColumn.return_value = df
    df.drop.return_value = df
    df.union.return_value = df
    df.write.return_value = df
    df.save.return_value = df
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
    assert len(output) == 2
    assert output[0] is spark_mock.read.csv.return_value
    assert output[1] is spark_mock.read.csv.return_value


def test_combine_dataframes(mocker, mock_dataframe, ingestor: DataIngestor):
    mocker.patch(
        "src.components.data_ingestor.monotonically_increasing_id",
        return_value=Mock(),
    )
    mocker.patch(
        "src.components.data_ingestor.input_file_name",
        return_value=Mock(),
    )

    output = ingestor.combine_dataframes(mock_dataframe, mock_dataframe)

    mock_dataframe.union.assert_called_once()
    assert output is mock_dataframe


def test_fix_column_names_and_dtypes(mocker, mock_dataframe, ingestor: DataIngestor):
    mocker.patch("src.components.data_ingestor.col", return_value=Mock())
    mocker.patch("src.components.data_ingestor.IntegerType", return_value=Mock())
    mocker.patch("src.components.data_ingestor.when", return_value=Mock())

    output = ingestor.fix_column_names_and_dtypes(mock_dataframe, mock_dataframe)

    assert len(output) == 2
    assert output[0] is mock_dataframe
    assert output[1] is mock_dataframe