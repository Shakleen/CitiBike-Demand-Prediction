import os
from dataclasses import dataclass
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import (
    col,
    count,
    regexp_extract,
    to_timestamp,
    when,
    add_months,
    year,
    month,
)

if __name__ == "__main__":
    from src.logger import logging


@dataclass
class RawToBronzeTransformerConfig:
    root_delta_path: str = os.path.join("Data", "delta")
    raw_data_path: str = os.path.join(root_delta_path, "raw")
    bronze_data_path: str = os.path.join(root_delta_path, "bronze")
    station_data_path: str = os.path.join(root_delta_path, "station")


class RawToBronzeTransformer:
    def __init__(self, spark: SparkSession):
        self.config = RawToBronzeTransformerConfig()
        self.spark = spark

    def read_raw_delta(self) -> DataFrame:
        return self.spark.read.format("delta").load(self.config.raw_data_path)

    def create_station_dataframe(self, df: DataFrame) -> DataFrame:
        s1 = (
            df.withColumnRenamed("start_station_id", "id")
            .withColumnRenamed("start_station_name", "name")
            .withColumnRenamed("start_station_latitude", "latitude")
            .withColumnRenamed("start_station_longitude", "longitude")
            .select("id", "name", "latitude", "longitude")
        )

        s2 = (
            df.withColumnRenamed("end_station_id", "id")
            .withColumnRenamed("end_station_name", "name")
            .withColumnRenamed("end_station_latitude", "latitude")
            .withColumnRenamed("end_station_longitude", "longitude")
            .select("id", "name", "latitude", "longitude")
        )

        return s1.union(s2)

    def write_delta(self, df: DataFrame, path: str):
        df.write.save(
            path=path,
            format="delta",
            mode="overwrite",
        )

    def create_file_name_column(self, df: DataFrame) -> DataFrame:
        regex_str = "[^\\/]+$"
        return df.withColumn(
            "file_name", regexp_extract(col("file_path"), regex_str, 0)
        )
