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
    coalesce,
)
from typing import Tuple

if __name__ == "__main__":
    from src.logger import logging


@dataclass
class RawToBronzeTransformerConfig:
    root_delta_path: str = os.path.join("Data", "delta")
    raw_data_path: str = os.path.join(root_delta_path, "raw")
    bronze_data_path: str = os.path.join(root_delta_path, "bronze")
    station_data_path: str = os.path.join(root_delta_path, "station")
    row_to_station_data_path: str = os.path.join(root_delta_path, "row_to_station")


class RawToBronzeTransformer:
    def __init__(self, spark: SparkSession):
        self.config = RawToBronzeTransformerConfig()
        self.spark = spark

    def read_raw_delta(self) -> DataFrame:
        return self.spark.read.format("delta").load(self.config.raw_data_path)

    def write_delta(self, df: DataFrame, path: str):
        df.write.save(path=path, format="delta", mode="overwrite")

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

        df_2 = self.get_dataframe_timeformat_type_2(df)
        df_2 = self.set_timestamp_for_format(df_2, "M/d/yyyy H:mm")

        df_3 = self.get_dataframe_timeformat_type_3(df)
        df_3 = self.set_timestamp_for_format(df_3, "M/d/yyyy HH:mm:ss")

        return df_1.union(df_2).union(df_3)

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

    def drop_duplicates_and_all_nulls(self, df: DataFrame) -> DataFrame:
        return df.dropDuplicates().dropna(how="all")

    def fill_in_station_id_using_name(self, df: DataFrame) -> DataFrame:
        # Create a mapping DataFrame with distinct non-null name and id pairs
        mapping_df = df.filter(df["id"].isNotNull()).select("name", "id").distinct()

        # Rename the id column in the mapping DataFrame to avoid conflicts
        mapping_df = mapping_df.withColumnRenamed("id", "mapped_id")

        # Join the original DataFrame with the mapping DataFrame
        df_filled = df.alias("df1").join(mapping_df.alias("df2"), on="name", how="left")

        # Use coalesce to fill null values in the id column
        df_filled = df_filled.withColumn(
            "id", coalesce(df_filled["df1.id"], df_filled["df2.mapped_id"])
        )

        # Drop the extra columns from the join
        df_filled = df_filled.drop("mapped_id")

        return df_filled

    def fill_in_using_station_id(self, df: DataFrame) -> DataFrame:
        # Create a mapping DataFrame with distinct non-null id and corresponding non-null values
        mapping_df = (
            df.filter(df["id"].isNotNull())
            .select("id", "name", "latitude", "longitude")
            .distinct()
        )
        mapping_df = (
            mapping_df.withColumnRenamed("name", "mapped_name")
            .withColumnRenamed("latitude", "mapped_latitude")
            .withColumnRenamed("longitude", "mapped_longitude")
        )

        # Show the mapping DataFrame
        mapping_df.show()

        # Join the original DataFrame with the mapping DataFrame on the id column
        df_filled = df.alias("df1").join(mapping_df.alias("df2"), on="id", how="left")

        # Use coalesce to fill null values in the name, latitude, and longitude columns
        df_filled = (
            df_filled.withColumn(
                "name", coalesce(df_filled["df1.name"], df_filled["mapped_name"])
            )
            .withColumn(
                "latitude",
                coalesce(df_filled["df1.latitude"], df_filled["mapped_latitude"]),
            )
            .withColumn(
                "longitude",
                coalesce(df_filled["df1.longitude"], df_filled["mapped_longitude"]),
            )
        )

        # Drop the extra columns from the join
        return (
            df_filled.drop("mapped_name")
            .drop("mapped_latitude")
            .drop("mapped_longitude")
            .dropDuplicates()
            .dropna(how="any")
        )

    def split_station_and_time(
        self, df: DataFrame
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        # Separating station Data
        station_df = self.get_station_dataframe(df)
        station_df = self.drop_duplicates_and_all_nulls(station_df)
        station_df = self.fill_in_station_id_using_name(station_df)
        station_df = self.fill_in_using_station_id(station_df)

        # Dropping rows with null station ids
        df = df.dropna(subset=["start_station_id", "end_station_id"], how="any")

        # Mapping df to station ids
        row_to_station_df = df.select(
            "row_number", "start_station_id", "end_station_id"
        )

        # Dropping station related columns
        df = df.drop(
            "start_station_id",
            "start_station_name",
            "start_station_latitude",
            "start_station_longitude",
            "end_station_id",
            "end_station_name",
            "end_station_latitude",
            "end_station_longitude",
            "member",
            "file_path",
            "file_name",
        )

        return (station_df, row_to_station_df, df)

    def transform(self):
        logging.info("Reading raw delta table")
        df = self.read_raw_delta()
        logging.info("Creating file name column")
        df = self.create_file_name_column(df)

        df = self.set_timestamp_datatype(df)

        station_df, row_to_station_df, df = self.split_station_and_time(df)
        self.write_delta(station_df, self.config.station_data_path)
        self.write_delta(row_to_station_df, self.config.row_to_station_data_path)

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
