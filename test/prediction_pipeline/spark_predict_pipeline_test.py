import pytest
import os
from unittest.mock import Mock, patch
from src.prediction_pipeline.spark_predict_pipeline import predict, model_artifacts, model_postfix


@pytest.mark.parametrize("model", ("random_forest", "gbt"))
def test_predict(model: str):
    data_pipeline_path = "./"
    mock_df = Mock()
    mock_data_pipeline = Mock()
    mock_regresiion_model = Mock()
    postfix = model_postfix.get(model)

    with (
        patch(
            "src.prediction_pipeline.spark_predict_pipeline.PipelineModel.load"
        ) as mocked_pload,
        patch(
            "src.prediction_pipeline.spark_predict_pipeline.RandomForestRegressionModel.load"
        ) as mocked_mload,
        patch(
            "src.prediction_pipeline.spark_predict_pipeline.cyclic_encode"
        ) as mocked_cyclic_encode,
    ):
        mocked_pload.return_value = mock_data_pipeline
        mocked_mload.return_value = mock_regresiion_model
        mocked_cyclic_encode.return_value = mock_df
        mock_regresiion_model.predict.return_value = 1

        output = predict(mock_df, model, data_pipeline_path)

        mocked_pload.assert_called_once_with(data_pipeline_path)
        mocked_mload.assert_called_with(
            os.path.join(model_artifacts.get(model), f"dock_model_{postfix}")
        )

        assert output == (1, 1)
