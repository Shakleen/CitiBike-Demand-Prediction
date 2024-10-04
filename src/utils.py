from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame


def read_delta(spark: SparkSession, path: str) -> DataFrame:
    return spark.read.format("delta").load(path)


def write_delta(df: DataFrame, path: str):
    df.write.save(path=path, format="delta", mode="overwrite")
