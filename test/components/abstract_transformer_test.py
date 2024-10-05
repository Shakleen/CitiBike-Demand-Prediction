import pytest
from unittest.mock import Mock, patch
from pyspark.sql import SparkSession

from src.components.abstract_transformer import AbstractTransformer


def test_init():
    with pytest.raises(
        TypeError,
        match="Can't instantiate abstract class AbstractTransformer with abstract method transform",
    ):
        mock_spark = Mock(SparkSession)
        mock_config = Mock()
        transformer = AbstractTransformer(mock_spark, mock_config)

def test_transform():
    assert getattr(AbstractTransformer.__dict__['transform'], '__isabstractmethod__', False)