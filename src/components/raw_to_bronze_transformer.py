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

    def set_timestamp_for_format(
        self, df: DataFrame, time_format: str
    ) -> DataFrame:
        return df.withColumn(
            "start_time", to_timestamp("start_time", time_format)
        ).withColumn("end_time", to_timestamp("end_time", time_format))

    def set_timestamp_datatype(self, df: DataFrame) -> DataFrame:
        df_1 = self.get_dataframe_timeformat_type_1(df)
        df_1 = self.set_timestamp_for_format(df_1, "yyyy-MM-dd HH:mm:ss")

        df_2 = self.get_dataframe_timeformat_type_2(df)
        df_2 = self.set_timestamp_for_format(df_2, "M/d/yyyy H:mm")

        df_3 = self.get_dataframe_timeformat_type_3(df)
        df_3 = self.set_timestamp_for_format(df_3, "M/d/yyyy HH:mm:ss")

        return df_1.union(df_2).union(df_3)

    def transform(self):
        logging.info("Reading raw delta table")
        df = self.read_raw_delta()
        logging.info("Creating file name column")
        df = self.create_file_name_column(df)

        self.set_timestamp_datatype(df)

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
