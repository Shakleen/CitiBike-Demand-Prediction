from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
import pickle as pkl


def read_delta(spark: SparkSession, path: str) -> DataFrame:
    return spark.read.format("delta").load(path)


def write_delta(df: DataFrame, path: str):
    df.write.save(path=path, format="delta", mode="overwrite")


def save_as_pickle(obj, path: str):
    with open(path, "wb") as file:
        pkl.dump(obj, file)


def read_pickle(path: str):
    obj = None

    with open(path, "rb") as file:
        obj = pkl.load(file)

    return obj
