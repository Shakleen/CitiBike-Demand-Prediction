import os
from dataclasses import dataclass

@dataclass
class XGBoostPipelineConfig:
    root_delta_path: str = os.path.join("Data", "delta")
    gold_delta_path: str = os.path.join(root_delta_path, "gold")
    model_artifact_path: str = os.path.join("artifacts", "model", "xgboost")