import os
from pyspark.sql import DataFrame
from typing import Tuple
from pyspark.ml import PipelineModel
from pyspark.ml.regression import RandomForestRegressionModel

from src.date_pipeline.silver_to_gold_transformer import cyclic_encode

model_artifacts = {
    "random_forest": os.path.join("artifacts", "model", "random_forest"),
    "gbt": os.path.join("artifacts", "model", "gbt"),
}

model_postfix = {"random_forest": "rf", "gbt": "gbt"}


def predict(
    df: DataFrame,
    model_name: str,
    data_pipeline_artifact_path: str,
) -> Tuple[int, int]:
    assert model_name in model_artifacts.keys()

    data_pipeline_model = PipelineModel.load(data_pipeline_artifact_path)
    artifact_path = model_artifacts.get(model_name)
    postfix = model_postfix.get(model_name)

    bike_model = RandomForestRegressionModel.load(
        os.path.join(artifact_path, f"bike_model_{postfix}")
    )

    df = cyclic_encode(df)
    transformed_df = data_pipeline_model.transform(df).select("final_features")
    bike_demand = bike_model.predict(transformed_df)

    dock_model = RandomForestRegressionModel.load(
        os.path.join(artifact_path, f"dock_model_{postfix}")
    )
    dock_demand = dock_model.predict(transformed_df)

    return (bike_demand, dock_demand)
