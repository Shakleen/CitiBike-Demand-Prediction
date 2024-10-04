import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
import pyspark
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import List

from src.utils import read_delta
from src.pipelines.cyclic_encoder import CyclicEncoder


@dataclass
class DataPreparationPipelineXGBoostConfig:
    root_delta_path: str = os.path.join("Data", "delta")
    gold_delta_path: str = os.path.join(root_delta_path, "gold")
    data_artifact_path: str = os.path.join("artifacts", "data")
    categorical_column_names = ["weekday", "month", "is_holiday"]
    numerical_column_names = ["latitude", "longitude", "year"]
    cyclic_column_periods = {
        "hourofday": 24,
        "dayofmonth": 31,
        "weekofyear": 52,
        "dayofyear": 366,
    }


class DataPreparationPipelineXGboost:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.config = DataPreparationPipelineXGBoostConfig()

    def get_column_transformer(self) -> ColumnTransformer:
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ],
        )

        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False)),
            ]
        )

        cyclic_transformers = [
            (
                f"{name}_pipeline",
                Pipeline(
                    steps=[
                        ("cyclic_encoder", CyclicEncoder(period)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                [name],
            )
            for name, period in self.config.cyclic_column_periods.items()
        ]

        preprocessor = ColumnTransformer(
            [
                ("num_pipeline", num_pipeline, self.config.numerical_column_names),
                ("cat_pipeline", cat_pipeline, self.config.categorical_column_names),
            ]
            + cyclic_transformers
        )

        return preprocessor
    

if __name__ == "__main__":
    pipeline = DataPreparationPipelineXGboost(None)
