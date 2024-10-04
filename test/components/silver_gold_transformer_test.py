import pytest

from src.components.silver_to_gold_transformer import SilverToGoldTransformerConfig


def test_config():
    config = SilverToGoldTransformerConfig()

    assert hasattr(config, "root_delta_path")
    assert hasattr(config, "silver_delta_path")
    assert hasattr(config, "gold_delta_path")
