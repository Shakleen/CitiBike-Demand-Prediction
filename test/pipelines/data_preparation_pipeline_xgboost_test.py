import pytest

from src.pipelines.data_preparation_pipeline_xgboost import (
    DataPreparationPipelineXGBoostConfig,
)


def test_config():
    config = DataPreparationPipelineXGBoostConfig()

    assert hasattr(config, "root_delta_path")
    assert hasattr(config, "gold_delta_path")
    assert hasattr(config, "data_artifact_path")
    assert hasattr(config, "time_feature_columns")
    assert hasattr(config, "boolean_feature_columns")
    assert hasattr(config, "place_feature_columns")
    assert hasattr(config, "label_columns")
