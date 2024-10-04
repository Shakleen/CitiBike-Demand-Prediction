import os
from dataclasses import dataclass
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
import pyspark.sql.functions as F
from typing import Tuple


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

    def create_time_features(self, df: DataFrame) -> DataFrame:
        return (
            df.withColumn("year", F.year("time"))
            .withColumn("month", F.month("time"))
            .withColumn("dayofmonth", F.dayofmonth("time"))
            .withColumn("weekday", F.weekday("time"))
            .withColumn("weekofyear", F.weekofyear("time"))
            .withColumn("dayofyear", F.dayofyear("time"))
            .withColumn("hour", F.hour("time"))
            .drop("time")
        )

    def split_start_and_end_time(self, df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        start_df = df.select("row_number", "start_time").withColumnRenamed(
            "start_time", "time"
        )
        end_df = df.select("row_number", "end_time").withColumnRenamed(
            "end_time", "time"
        )
        return (start_df, end_df)

    def attach_station_ids(
        self,
        start_df: DataFrame,
        end_df: DataFrame,
        mapper_df: DataFrame,
    ) -> Tuple[DataFrame, DataFrame]:
        start_df = start_df.join(
            mapper_df.select("row_number", "start_station_id"),
            on="row_number",
            how="inner",
        ).withColumnRenamed("start_station_id", "station_id")
        end_df = end_df.join(
            mapper_df.select("row_number", "end_station_id"),
            on="row_number",
            how="inner",
        ).withColumnRenamed("end_station_id", "station_id")
        return (start_df, end_df)

    def count_group_by_station_and_time(self, df: DataFrame) -> DataFrame:
        df = df.withColumn("time", F.date_trunc("hour", "time"))
        count_df = df.groupBy("station_id", "time").agg(
            F.count("row_number").alias("count")
        )
        return count_df

    def combine_on_station_id_and_time(
        self,
        start_df: DataFrame,
        end_df: DataFrame,
    ) -> DataFrame:
        start_df = start_df.withColumnRenamed("count", "bike_demand")
        end_df = end_df.withColumnRenamed("count", "dock_demand")
        combined_df = start_df.join(
            end_df,
            on=["station_id", "time"],
            how="fullouter",
        ).fillna(0)
        return combined_df

    def holiday_weekend(self, df: DataFrame) -> DataFrame:
        return df.withColumn(
            "is_holiday",
            F.when(F.col("weekday") > 4, F.lit(True)).otherwise(F.col("is_holiday")),
        )

    def holiday_MLK_and_presidents_day(self, df: DataFrame) -> DataFrame:
        return df.withColumn(
            "is_holiday",
            F.when(
                ((F.col("month")).isin(1, 2))  # January or February
                & ((F.col("weekday")) == 0)  # Monday
                & ((F.col("dayofmonth")) >= 15)  # 3rd week
                & ((F.col("dayofmonth")) <= 21),  # 3rd week
                F.lit(True),
            ).otherwise(F.col("is_holiday")),
        )
