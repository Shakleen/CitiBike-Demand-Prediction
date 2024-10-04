import os
from dataclasses import dataclass
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import (
    col,
    regexp_extract,
    to_timestamp,
    min,
    when,
    month,
    year,
    add_months,
)
from typing import Tuple
from src.utils import read_delta, write_delta

if __name__ == "__main__":
    from src.logger import logging


@dataclass
class RawToBronzeTransformerConfig:
    root_delta_path: str = os.path.join("Data", "delta")
    raw_delta_path: str = os.path.join(root_delta_path, "raw")
    bronze_delta_path: str = os.path.join(root_delta_path, "bronze")
    station_delta_path: str = os.path.join(root_delta_path, "station")


class RawToBronzeTransformer:
    def __init__(self, spark: SparkSession):
        self.config = RawToBronzeTransformerConfig()
        self.spark = spark

    def create_file_name_column(self, df: DataFrame) -> DataFrame:
        regex_str = "[^\\/]+$"
        return df.withColumn(
            "file_name", regexp_extract(col("file_path"), regex_str, 0)
        )

    def get_dataframe_timeformat_type_1(self, df: DataFrame) -> DataFrame:
        # Create a list of conditions
        conditions = [
            col("file_name").startswith(file_name_prefix)
            for file_name_prefix in ["202", "2013", "2017", "2018", "2019"]
        ]

        # Add conditions for 2014 and 2015 months
        conditions += [col("file_name").startswith(f"20140{i}") for i in range(1, 9)]
        conditions += [col("file_name").startswith(f"2016{i}") for i in range(10, 13)]

        # Combine all conditions using the 'or' operation
        filter_condition = conditions[0]
        for condition in conditions[1:]:
            filter_condition = filter_condition | condition

        # Apply the filter condition to the DataFrame
        df_1 = df.where(filter_condition)

        return df_1

    def get_dataframe_timeformat_type_2(self, df: DataFrame) -> DataFrame:
        conditions = [
            col("file_name").startswith(file_name_prefix)
            for file_name_prefix in ["201501", "201502", "201503", "201506"]
        ]

        filter_condition = conditions[0]
        for condition in conditions[1:]:
            filter_condition = filter_condition | condition

        df_2 = df.where(filter_condition)

        return df_2

    def get_dataframe_timeformat_type_3(self, df: DataFrame) -> DataFrame:
        # Create a list of conditions for 2014
        conditions_2014 = [
            col("file_name").startswith(f"2014{i:02}") for i in range(9, 13)
        ]

        # Create a list of conditions for 2015, excluding 201506
        conditions_2015 = [
            col("file_name").startswith(f"2015{i:02}") for i in range(4, 13) if i != 6
        ]

        # Create a list of conditions for 2016
        conditions_2016 = [
            col("file_name").startswith(f"2016{i:02}") for i in range(1, 10)
        ]

        # Combine all conditions into a single list
        all_conditions = conditions_2014 + conditions_2015 + conditions_2016

        # Combine all conditions using the 'or' operation
        filter_condition = all_conditions[0]
        for condition in all_conditions[1:]:
            filter_condition = filter_condition | condition

        # Apply the filter condition to the DataFrame
        df_3 = df.where(filter_condition)

        return df_3

    def set_timestamp_for_format(self, df: DataFrame, time_format: str) -> DataFrame:
        return df.withColumn(
            "start_time", to_timestamp("start_time", time_format)
        ).withColumn("end_time", to_timestamp("end_time", time_format))

    def set_timestamp_datatype(self, df: DataFrame) -> DataFrame:
        df_1 = self.get_dataframe_timeformat_type_1(df)
        df_1 = self.set_timestamp_for_format(df_1, "yyyy-MM-dd HH:mm:ss")
        df_1 = self.add_one_month_for_202108(df_1)

        df_2 = self.get_dataframe_timeformat_type_2(df)
        df_2 = self.set_timestamp_for_format(df_2, "M/d/yyyy H:mm")

        df_3 = self.get_dataframe_timeformat_type_3(df)
        df_3 = self.set_timestamp_for_format(df_3, "M/d/yyyy HH:mm:ss")

        return df_1.union(df_2).union(df_3)

    def add_one_month_for_202108(self, df: DataFrame) -> DataFrame:
        return df.withColumn(
            "start_time",
            when(
                (col("file_name").startswith("202108"))
                & (year("start_time") == 2021)
                & (month("start_time") == 7),
                add_months(col("start_time"), 1),
            ).otherwise(col("start_time")),
        ).withColumn(
            "end_time",
            when(
                (col("file_name").startswith("202108"))
                & (year("end_time") == 2021)
                & (month("end_time") == 7),
                add_months(col("end_time"), 1),
            ).otherwise(col("end_time")),
        )

    def get_station_dataframe(self, df: DataFrame) -> DataFrame:
        return (
            df.withColumnRenamed("start_station_id", "id")
            .withColumnRenamed("start_station_name", "name")
            .withColumnRenamed("start_station_latitude", "latitude")
            .withColumnRenamed("start_station_longitude", "longitude")
            .select("id", "name", "latitude", "longitude")
        ).union(
            (
                df.withColumnRenamed("end_station_id", "id")
                .withColumnRenamed("end_station_name", "name")
                .withColumnRenamed("end_station_latitude", "latitude")
                .withColumnRenamed("end_station_longitude", "longitude")
                .select("id", "name", "latitude", "longitude")
            )
        )

    def split_station_and_time(
        self, df: DataFrame
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        # Dropping rows with null station ids
        df = df.dropna(subset=["start_station_id", "end_station_id"], how="any")

        # Separating station Data
        station_df = self.get_station_dataframe(df)
        station_df = station_df.groupBy("id").agg(
            min("name").alias("name"),
            min("latitude").alias("latitude"),
            min("longitude").alias("longitude"),
        )

        # Dropping station related columns
        df = df.drop(
            "start_station_name",
            "start_station_latitude",
            "start_station_longitude",
            "end_station_name",
            "end_station_latitude",
            "end_station_longitude",
            "member",
            "file_path",
            "file_name",
        )

        return (station_df, df)

    def transform(self):
        logging.info("Reading raw delta table")
        df = read_delta(self.spark, self.config.raw_delta_path)

        logging.info("Creating file name column")
        df = self.create_file_name_column(df)

        df = self.set_timestamp_datatype(df)

        station_df, df = self.split_station_and_time(df)

        logging.info("Saving as deltalakes")
        write_delta(station_df, self.config.station_delta_path)
        write_delta(df, self.config.bronze_delta_path)


if __name__ == "__main__":
    import pyspark
    from delta import configure_spark_with_delta_pip

    builder = (
        pyspark.sql.SparkSession.builder.appName("raw_to_bronze")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.driver.memory", "15g")
        .config("spark.sql.shuffle.partitions", "6")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
    )

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    transformer = RawToBronzeTransformer(spark)
    transformer.transform()
