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


class RawToBronzeTransformer:
    def __init__(self, spark: SparkSession):
        self.config = RawToBronzeTransformerConfig()
        self.spark = spark

    def read_raw_delta(self) -> DataFrame:
        return self.spark.read.format("delta").load(self.config.raw_data_path)

    def write_delta(self, df: DataFrame):
        df.write.save(
            path=self.config.bronze_data_path,
            format="delta",
            mode="overwrite",
        )

    def create_file_name_column(self, df: DataFrame) -> DataFrame:
        regex_str = "[^\\/]+$"
        return df.withColumn(
            "file_name", regexp_extract(col("file_path"), regex_str, 0)
        )

    def convert_timestamp_type_1(self, df: DataFrame) -> DataFrame:
        df_1 = df.where(col("file_name").startswith("202"))

        for i in range(2013, 2020, 1):
            if i in [2014, 2015, 2016]:
                continue

            temp_df = df.where(col("file_name").startswith(str(i)))
            df_1 = df_1.union(temp_df)

        for i in range(1, 9):
            temp_df = df.where(col("file_name").startswith(f"20140{i}"))
            df_1 = df_1.union(temp_df)

        for i in range(10, 13):
            temp_df = df.where(col("file_name").startswith(f"2016{i}"))
            df_1 = df_1.union(temp_df)

        df_1 = df_1.withColumn(
            "start_time", to_timestamp("start_time", "yyyy-MM-dd HH:mm:ss")
        ).withColumn("end_time", to_timestamp("end_time", "yyyy-MM-dd HH:mm:ss"))

        return (
            df_1.withColumn(
                "start_time",
                when(
                    (col("file_name").startswith("202108"))
                    & (year("start_time") == 2021)
                    & (month("start_time") == 7),
                    add_months(col("start_time"), 1),
                ).otherwise(col("start_time")),
            )
            .withColumn(
                "end_time",
                when(
                    (col("file_name").startswith("202108"))
                    & (year("end_time") == 2021)
                    & (month("end_time") == 7),
                    add_months(col("end_time"), 1),
                ).otherwise(col("end_time")),
            )
            .drop("file_path", "file_name")
        )

    def convert_timestamp_type_2(self, df: DataFrame) -> DataFrame:
        df_2 = None

        for i in ["201501", "201502", "201503", "201506"]:
            temp_df = df.where(col("file_name").startswith(i))

            if df_2 is None:
                df_2 = temp_df
            else:
                df_2 = df_2.union(temp_df)

        return (
            df_2.withColumn("start_time", to_timestamp("start_time", "M/d/yyyy H:mm"))
            .withColumn("end_time", to_timestamp("end_time", "M/d/yyyy H:mm"))
            .drop("file_path", "file_name")
        )

    def convert_timestamp_type_3(self, df: DataFrame) -> DataFrame:
        df_3 = None

        for i in range(9, 13):
            temp_df = df.where(col("file_name").startswith(f"2014{i:02}"))

            if df_3 is None:
                df_3 = temp_df
            else:
                df_3 = df_3.union(temp_df)

        for i in range(4, 13):
            if i == 6:
                continue

            temp_df = df.where(col("file_name").startswith(f"2015{i:02}"))
            df_3 = df_3.union(temp_df)

        for i in range(1, 10):
            temp_df = df.where(col("file_name").startswith(f"2016{i:02}"))
            df_3 = df_3.union(temp_df)

        return (
            df_3.withColumn(
                "start_time", to_timestamp("start_time", "M/d/yyyy HH:mm:ss")
            )
            .withColumn("end_time", to_timestamp("end_time", "M/d/yyyy HH:mm:ss"))
            .drop("file_path", "file_name")
        )

    def standardize_time_format(self, df: DataFrame) -> DataFrame:
        df_1 = self.convert_timestamp_type_1(df)
        df_2 = self.convert_timestamp_type_2(df)
        df_3 = self.convert_timestamp_type_3(df)
        return df_1.union(df_2).union(df_3)

    def transform(self):
        logging.info("Reading raw delta table")
        df = self.read_raw_delta()
        logging.info("Creating file name column")
        df = self.create_file_name_column(df)
        logging.info("Standardizing time")
        df = self.standardize_time_format(df)
        logging.info("Dropping rows with null values for stations")
        df = df.na.drop(
            subset=[
                "start_station_id",
                "start_station_name",
                "start_station_latitude",
                "start_station_longitude",
                "end_station_id",
                "end_station_name",
                "end_station_latitude",
                "end_station_longitude",
            ]
        )
        logging.info("Writing to bronze delta table")
        self.write_delta(df)


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
