import os
from dataclasses import dataclass
import pyspark
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import List


@dataclass
class DataPreparationPipelineXGBoostConfig:
    root_delta_path: str = os.path.join("Data", "delta")
    gold_delta_path: str = os.path.join(root_delta_path, "gold")
    data_artifact_path: str = os.path.join("artifacts", "data")
    time_feature_columns = [
        "year",
        "month",
        "dayofmonth",
        "weekday",
        "weekofyear",
        "dayofyear",
        "hour",
    ]
    boolean_feature_columns = ["is_holiday"]
    place_feature_columns = ["latitude", "longitude"]
    label_columns = ["demand_bike", "demand_dock"]


class DataPreparationPipelineXGboost:
    def __init__(self, spark: SparkSession):
        self.spark = spark
