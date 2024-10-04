import os
import pyspark
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructField,
    StructType,
    IntegerType,
    StringType,
    FloatType,
)
from pyspark.sql.functions import (
    monotonically_increasing_id,
    input_file_name,
    col,
    when,
)
from delta import configure_spark_with_delta_pip
from dataclasses import dataclass
from typing import Tuple

from src.logger import logging
from src.utils import write_delta


@dataclass
class DataIngestorConfig:
    root_data_path: str = "Data"
    root_csv_path: str = os.path.join(root_data_path, "CSVs")
    pre_2020_csv_dir: str = os.path.join(root_csv_path, "pre_2020")
    pre_2020_schema = StructType(
        [
            StructField("tripduration", IntegerType(), False),
            StructField("starttime", StringType(), False),
            StructField("stoptime", StringType(), False),
            StructField("start station id", FloatType(), False),
            StructField("start station name", StringType(), False),
            StructField("start station latitude", FloatType(), False),
            StructField("start station longitude", FloatType(), False),
            StructField("end station id", FloatType(), False),
            StructField("end station name", StringType(), False),
            StructField("end station latitude", FloatType(), False),
            StructField("end station longitude", FloatType(), False),
            StructField("bikeid", IntegerType(), False),
            StructField("usertype", StringType(), False),
            StructField("birth year", IntegerType(), False),
            StructField("gender", IntegerType(), False),
        ]
    )

    post_2020_csv_dir: str = os.path.join(root_csv_path, "post_2020")
    post_2020_schema = StructType(
        [
            StructField("ride_id", StringType(), False),
            StructField("rideable_type", StringType(), False),
            StructField("started_at", StringType(), False),
            StructField("ended_at", StringType(), False),
            StructField("start_station_name", StringType(), False),
            StructField("start_station_id", FloatType(), False),
            StructField("end_station_name", StringType(), False),
            StructField("end_station_id", FloatType(), False),
            StructField("start_lat", FloatType(), False),
            StructField("start_lng", FloatType(), False),
            StructField("end_lat", FloatType(), False),
            StructField("end_lng", FloatType(), False),
            StructField("member_casual", StringType(), False),
        ]
    )

    raw_delta_path: str = os.path.join(root_data_path, "delta", "raw")
    column_order = [
        "start_time",
        "end_time",
        "start_station_name",
        "start_station_latitude",
        "start_station_longitude",
        "end_station_name",
        "end_station_latitude",
        "end_station_longitude",
        "start_station_id",
        "end_station_id",
        "member",
    ]


class DataIngestor:
    def __init__(self, spark: SparkSession):
        self.config = DataIngestorConfig()
        self.spark = spark

    def read_csv(self, csv_dir: str, schema: StructType) -> DataFrame:
        return self.spark.read.csv(csv_dir, schema=schema, header=True)

    def read_pre_and_post_csv_dir(self) -> Tuple[DataFrame, DataFrame]:
        logging.info("Reading pre-2020 and post-2020 data")
        pre_2020_df = self.read_csv(
            self.config.pre_2020_csv_dir,
            self.config.pre_2020_schema,
        )
        post_2020_df = self.read_csv(
            self.config.post_2020_csv_dir,
            self.config.post_2020_schema,
        )
        return pre_2020_df, post_2020_df

    def combine_dataframes(self, df_1: DataFrame, df_2: DataFrame) -> DataFrame:
        logging.info("Combining pre-2020 and post-2020 dataframes")
        df = df_1.union(df_2)
        df = df.withColumn("row_number", monotonically_increasing_id())
        df = df.withColumn("file_path", input_file_name())
        return df

    def standardize_columns_for_post2020(self, post_2020_df: DataFrame):
        return (
            post_2020_df.withColumnRenamed("started_at", "start_time")
            .withColumnRenamed("ended_at", "end_time")
            .withColumnRenamed("start_lat", "start_station_latitude")
            .withColumnRenamed("start_lng", "start_station_longitude")
            .withColumnRenamed("end_lat", "end_station_latitude")
            .withColumnRenamed("end_lng", "end_station_longitude")
            .withColumn("start_station_id", col("start_station_id").cast(IntegerType()))
            .withColumn("end_station_id", col("end_station_id").cast(IntegerType()))
            .withColumn(
                "member", when(col("member_casual") == "casual", 0).otherwise(1)
            )
            .drop("ride_id", "rideable_type", "member_casual")
            .select(self.config.column_order)
        )

    def standardize_columns_for_pre2020(self, pre_2020_df: DataFrame):
        return (
            pre_2020_df.withColumnRenamed("starttime", "start_time")
            .withColumnRenamed("stoptime", "end_time")
            .withColumnRenamed("start station name", "start_station_name")
            .withColumnRenamed("start station latitude", "start_station_latitude")
            .withColumnRenamed("start station longitude", "start_station_longitude")
            .withColumnRenamed("end station name", "end_station_name")
            .withColumnRenamed("end station latitude", "end_station_latitude")
            .withColumnRenamed("end station longitude", "end_station_longitude")
            .withColumn("start_station_id", col("start station id").cast(IntegerType()))
            .withColumn("end_station_id", col("end station id").cast(IntegerType()))
            .withColumn("member", when(col("usertype") == "Subscriber", 1).otherwise(0))
            .drop(
                "tripduration",
                "bikeid",
                "birth year",
                "gender",
                "start station id",
                "end station id",
                "usertype",
            )
            .select(self.config.column_order)
        )

    def ingest(self):
        logging.info("Initiated data ingestion from CSV files")
        pre_2020_df, post_2020_df = self.read_pre_and_post_csv_dir()

        pre_2020_df = self.standardize_columns_for_pre2020(pre_2020_df)
        post_2020_df = self.standardize_columns_for_post2020(post_2020_df)

        df = self.combine_dataframes(pre_2020_df, post_2020_df)
        write_delta(df, self.config.raw_delta_path)


if __name__ == "__main__":
    builder = (
        pyspark.sql.SparkSession.builder.appName("csv_to_raw")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    ingestor = DataIngestor(spark)
    ingestor.ingest()
