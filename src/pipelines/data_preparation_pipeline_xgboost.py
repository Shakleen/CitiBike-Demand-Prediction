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
from sklearn.model_selection import TimeSeriesSplit

from src.utils import read_delta, save_as_pickle
from src.pipelines.cyclic_encoder import CyclicEncoder


@dataclass
class DataPreparationPipelineXGBoostConfig:
    root_delta_path: str = os.path.join("Data", "delta")
    gold_delta_path: str = os.path.join(root_delta_path, "silver")
    data_artifact_path: str = os.path.join("artifacts", "data")
    pipeline_artifact_path: str = os.path.join(
        "artifacts", "pipelines", "xgboost_data_pipeline"
    )
    label_column_names = ["bike_demand", "dock_demand"]
    categorical_column_names = ["weekday", "month", "is_holiday"]
    numerical_column_names = ["latitude", "longitude", "year"]
    cyclic_column_periods = {
        "hour": 24,
        "dayofmonth": 31,
        "weekofyear": 52,
        "dayofyear": 366,
    }
    cv_folds: int = 10
    max_train_size: int = 1e7


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

    def transform(self):
        dataframe = read_delta(self.spark, self.config.gold_delta_path)
        features_df, label_df = self.split_features_and_labels(dataframe)

        features_df = self.transform_features(features_df)
        data = np.c_[features_df, label_df.to_numpy()]

        tscv = TimeSeriesSplit(
            n_splits=self.config.cv_folds,
            max_train_size=self.config.max_train_size,
        )
        # for i, (train_index, test_index) in enumerate(tscv.split(data)):

    def transform_features(self, features_df):
        preprocessor = self.get_column_transformer()
        features_df = preprocessor.fit_transform(features_df)
        save_as_pickle(preprocessor, self.config.pipeline_artifact_path)
        return features_df

    def split_features_and_labels(self, df):
        features_df = df.drop(columns=self.config.label_column_names)
        label_df = df.loc[:, self.config.label_column_names]
        return features_df, label_df


if __name__ == "__main__":
    pipeline = DataPreparationPipelineXGboost(None)
