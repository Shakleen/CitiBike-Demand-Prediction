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