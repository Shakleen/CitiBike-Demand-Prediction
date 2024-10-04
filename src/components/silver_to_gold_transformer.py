import os
from dataclasses import dataclass
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
import pyspark.sql.functions as F
from typing import Tuple

from src.utils import read_delta, write_delta


@dataclass
class SilverToGoldTransformerConfig:
    root_delta_path: str = os.path.join("Data", "delta")
    silver_delta_path: str = os.path.join(root_delta_path, "silver")
    gold_delta_path: str = os.path.join(root_delta_path, "gold")


class SilverToGoldTransformer:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.config = SilverToGoldTransformerConfig()

    def cyclic_encode(self, df):
        return (
            df.withColumn(
                "dayofmonth_sin", F.sin((2 * F.pi() * F.col("dayofmonth")) / 31)
            )
            .withColumn("dayofmonth_cos", F.cos((2 * F.pi() * F.col("dayofmonth")) / 31))
            .withColumn("weekofyear_sin", F.sin((2 * F.pi() * F.col("weekofyear")) / 53))
            .withColumn("weekofyear_cos", F.cos((2 * F.pi() * F.col("weekofyear")) / 53))
            .withColumn("dayofyear_sin", F.sin((2 * F.pi() * F.col("dayofyear")) / 366))
            .withColumn("dayofyear_cos", F.cos((2 * F.pi() * F.col("dayofyear")) / 366))
            .withColumn("hour_sin", F.sin((2 * F.pi() * F.col("hour")) / 24))
            .withColumn("hour_cos", F.cos((2 * F.pi() * F.col("hour")) / 24))
            .drop("dayofmonth", "weekofyear", "dayofyear", "hour")
        )

    def transform(self):
        df = read_delta(self.spark, self.config.silver_delta_path)

        df = self.cyclic_encode(df)
