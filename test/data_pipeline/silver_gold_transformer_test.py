import pytest
from unittest.mock import Mock, patch
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark.ml import Pipeline

from src.date_pipeline.silver_to_gold_transformer import (
    SilverToGoldTransformerConfig,
    SilverToGoldTransformer,
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
    return SilverToGoldTransformer(spark)


@pytest.fixture
def dataframe(spark: SparkSession):
    df = spark.createDataFrame(
        data=[
            [
                2,
                0,
                40.76727294921875,
                -73.99392700195312,
                2013,
                6,
                1,
                5,
                22,
                152,
                8,
                True,
            ],
            [
                1,
                1,
                40.76727294921875,
                -73.99392700195312,
                2013,
                6,
                1,
                5,
                22,
                152,
                11,
                True,
            ],
            [
                2,
                4,
                40.76727294921875,
                -73.99392700195312,
                2013,
                6,
                1,
                5,
                22,
                152,
                14,
                True,
            ],
            [
                3,
                1,
                40.76727294921875,
                -73.99392700195312,
                2013,
                6,
                1,
                5,
                22,
                152,
                15,
                True,
            ],
            [
                3,
                4,
                40.76727294921875,
                -73.99392700195312,
                2013,
                6,
                1,
                5,
                22,
                152,
                18,
                True,
            ],
        ],
        schema=T.StructType(
            [
                T.StructField("bike_demand", T.LongType(), True),
                T.StructField("dock_demand", T.LongType(), True),
                T.StructField("latitude", T.FloatType(), True),
                T.StructField("longitude", T.FloatType(), True),
                T.StructField("year", T.IntegerType(), True),
                T.StructField("month", T.IntegerType(), True),
                T.StructField("dayofmonth", T.IntegerType(), True),
                T.StructField("weekday", T.IntegerType(), True),
                T.StructField("weekofyear", T.IntegerType(), True),
                T.StructField("dayofyear", T.IntegerType(), True),
                T.StructField("hour", T.IntegerType(), True),
                T.StructField("is_holiday", T.BooleanType(), True),
            ]
        ),
    )

    return df


def test_config():
    config = SilverToGoldTransformerConfig()

    assert hasattr(config, "root_delta_path")
    assert hasattr(config, "silver_delta_path")
    assert hasattr(config, "gold_delta_path")
    assert hasattr(config, "pipeline_artifact_path")
    assert hasattr(config, "numerical_columns")
    assert hasattr(config, "categorical_columns")
    assert hasattr(config, "label_columns")


def test_init(transformer: SilverToGoldTransformer, spark: SparkSession):
    assert hasattr(transformer, "config")
    assert hasattr(transformer, "spark")
    assert transformer.spark is spark
    assert isinstance(transformer.config, SilverToGoldTransformerConfig)
    assert isinstance(transformer.spark, SparkSession)


def test_cyclic_encode(transformer: SilverToGoldTransformer, dataframe: DataFrame):
    expected = {
        "dayofmonth_sin",
        "dayofmonth_cos",
        "weekofyear_sin",
        "weekofyear_cos",
        "dayofyear_sin",
        "dayofyear_cos",
        "hour_sin",
        "hour_cos",
    }
    output = transformer.cyclic_encode(dataframe)

    assert len(set(output.columns).intersection(expected)) == len(expected)

    assert (
        len(
            set(output.columns).intersection(
                {"dayofmonth", "weekofyear", "dayofyear", "hour"}
            )
        )
        == 0
    )


def test_get_pipeline(transformer: SilverToGoldTransformer):
    pipeline = transformer.get_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.getStages()) == 5
