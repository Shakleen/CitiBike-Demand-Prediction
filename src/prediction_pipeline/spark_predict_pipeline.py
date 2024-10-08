import os
from pyspark.sql import DataFrame
from typing import Tuple
from pyspark.ml import PipelineModel
from pyspark.ml.regression import RandomForestRegressionModel

from src.data_pipeline.silver_to_gold_transformer import cyclic_encode

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
    transformed_df = data_pipeline_model.transform(df)
    bike_demand: DataFrame = bike_model.transform(transformed_df)
    bike_demand = int(
        bike_demand.select("predicted_bike_demand").toPandas().iloc[0]
    )

    dock_model = RandomForestRegressionModel.load(
        os.path.join(artifact_path, f"dock_model_{postfix}")
    )
    dock_demand = dock_model.transform(transformed_df)
    dock_demand = int(
        dock_demand.select("predicted_dock_demand").toPandas().iloc[0]
    )

    return (bike_demand, dock_demand)
