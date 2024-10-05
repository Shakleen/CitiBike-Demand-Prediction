import os
from dataclasses import dataclass
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    OneHotEncoder,
    StandardScaler,
    VectorAssembler,
)

from src.utils import read_delta, write_delta
from src.components.abstract_transformer import AbstractTransformer


@dataclass
class SilverToGoldTransformerConfig:
    root_delta_path: str = os.path.join("Data", "delta")
    silver_delta_path: str = os.path.join(root_delta_path, "silver")
    gold_delta_path: str = os.path.join(root_delta_path, "gold")
    pipeline_artifact_path: str = os.path.join(
        "artifacts", "pipelines", "gold_pipeline"
    )
    numerical_columns = [
        "latitude",
        "longitude",
        "year",
        "dayofmonth_sin",
        "dayofmonth_cos",
        "weekofyear_sin",
        "weekofyear_cos",
        "dayofyear_sin",
        "dayofyear_cos",
        "hour_sin",
        "hour_cos",
    ]
    categorical_columns = ["month", "weekday", "is_holiday"]
    label_columns = ["bike_demand", "dock_deman"]


class SilverToGoldTransformer(AbstractTransformer):
    def __init__(self, spark: SparkSession):
        super().__init__(spark, SilverToGoldTransformerConfig())

    def cyclic_encode(self, df):
        return (
            df.withColumn(
                "dayofmonth_sin", F.sin((2 * F.pi() * F.col("dayofmonth")) / 31)
            )
            .withColumn(
                "dayofmonth_cos", F.cos((2 * F.pi() * F.col("dayofmonth")) / 31)
            )
            .withColumn(
                "weekofyear_sin", F.sin((2 * F.pi() * F.col("weekofyear")) / 52)
            )
            .withColumn(
                "weekofyear_cos", F.cos((2 * F.pi() * F.col("weekofyear")) / 52)
            )
            .withColumn("dayofyear_sin", F.sin((2 * F.pi() * F.col("dayofyear")) / 365))
            .withColumn("dayofyear_cos", F.cos((2 * F.pi() * F.col("dayofyear")) / 365))
            .withColumn("hour_sin", F.sin((2 * F.pi() * F.col("hour")) / 24))
            .withColumn("hour_cos", F.cos((2 * F.pi() * F.col("hour")) / 24))
            .drop("dayofmonth", "weekofyear", "dayofyear", "hour")
        )

    def get_pipeline(self) -> Pipeline:
        numerical_assembler = VectorAssembler(
            inputCols=self.config.numerical_columns,
            outputCol="numerical_features",
        )
        standard_scaler = StandardScaler(
            inputCol="numerical_features",
            outputCol="scaled_features",
            withMean=True,
            withStd=True,
        )
        month_encoder = OneHotEncoder(inputCol="month", outputCol="encoded_month")
        weekday_encoder = OneHotEncoder(inputCol="weekday", outputCol="encoded_weekday")
        concat_assembler = VectorAssembler(
            inputCols=[
                "scaled_features",
                "encoded_month",
                "encoded_weekday",
                "is_holiday",
            ],
            outputCol="final_features",
        )
        return Pipeline(
            stages=[
                numerical_assembler,
                standard_scaler,
                month_encoder,
                weekday_encoder,
                concat_assembler,
            ]
        )

    def transform(self):
        df = read_delta(self.spark, self.config.silver_delta_path)
        df = self.cyclic_encode(df)

        pipeline = self.get_pipeline()
        model = pipeline.fit(df)
        model.save(self.config.pipeline_artifact_path)

        processed_df = model.transform(df)
        processed_df = processed_df.select(
            "final_features", "bike_demand", "dock_demand"
        )
        write_delta(processed_df, self.config.gold_delta_path)


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
    transformer = SilverToGoldTransformer(spark)
    transformer.transform()
