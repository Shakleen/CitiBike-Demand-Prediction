import os
from dataclasses import dataclass
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame

if __name__ == "__main__":
    from src.logger import logging


@dataclass
class RawToBronzeTransformerConfig:
    root_delta_path: str = os.path.join("Data", "delta")
    raw_data_path: str = os.path.join(root_delta_path, "raw")
    bronze_data_path: str = os.path.join(root_delta_path, "bronze")


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
