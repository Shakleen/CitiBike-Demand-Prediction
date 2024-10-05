import pytest

from src.train_pipeline.xgboost_pipeline import XGBoostPipelineConfig

def test_config():
    config = XGBoostPipelineConfig()

    assert hasattr(config, "root_delta_path")
    assert hasattr(config, "gold_delta_path")
    assert hasattr(config, "model_artifact_path")