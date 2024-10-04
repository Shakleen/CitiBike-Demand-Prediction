import os
from dataclasses import dataclass
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
import pyspark.sql.functions as F


@dataclass
class BronzeToSilverTransformerConfig:
    root_delta_path: str = os.path.join("data", "delta")
    bronze_delta_path: str = os.path.join(root_delta_path, "bronze")
    silver_delta_path: str = os.path.join(root_delta_path, "silver")


class BronzeToSilverTransformer:
    def __init__(self, spark: SparkSession) -> None:
        self.spark = spark
        self.config = BronzeToSilverTransformerConfig()

    def read_bronze_delta(self) -> DataFrame:
        return self.spark.read.format("delta").load(self.config.bronze_delta_path)

    def write_delta(self, df: DataFrame, path: str):
        df.write.save(path=path, format="delta", mode="overwrite")

    def create_time_features(self, df: DataFrame, column_name: str) -> DataFrame:
        return (
            df.withColumn("year", F.year(column_name))
            .withColumn("month", F.month(column_name))
            .withColumn("dayofmonth", F.dayofmonth(column_name))
            .withColumn("weekday", F.weekday(column_name))
            .withColumn("weekofyear", F.weekofyear(column_name))
            .withColumn("dayofyear", F.dayofyear(column_name))
            .withColumn("hour", F.hour(column_name))
        )
