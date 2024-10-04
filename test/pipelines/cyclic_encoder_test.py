import pytest
import numpy as np
from src.pipelines.cyclic_encoder import CyclicEncoder


@pytest.mark.parametrize("period", [24, 30, 365])
def test_init(period: int):
    encoder = CyclicEncoder(period)
    assert encoder.period == period


def test_fit():
    encoder = CyclicEncoder(24)
    assert encoder.fit(None, None) is encoder


@pytest.mark.parametrize("period", [24, 30, 365])
def test_transform(period: int):
    encoder = CyclicEncoder(period)
    X = list(range(period))
    expected = np.concatenate(
        [
            np.sin(2 * np.pi * np.array(X) / period),
            np.cose(2 * np.pi * np.array(X) / period),
        ],
        axis=1,
    )

    assert encoder.transform(X) == expected
