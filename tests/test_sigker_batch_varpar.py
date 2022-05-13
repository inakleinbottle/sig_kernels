
import math

import pytest
import numpy as np

from sigker_backends import sig_kernel_batch_varpar


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def make_brownian_path(rng: np.random.Generator, length: int, dim: int, time: float):
    mu = 0.0
    sigma = math.sqrt(time / length)
    return rng.normal(mu, sigma, size=(length, dim))


@pytest.fixture
def npaths():
    return 3


@pytest.fixture
def brownian_paths(rng, npaths):
    return np.array([make_brownian_path(rng, 50, 2, 1.0) for _ in range(npaths)])


def test_sig_kernel_cpp_backend(brownian_paths, npaths):

    kernel = sig_kernel_batch_varpar(brownian_paths)

    assert kernel.shape == (npaths, 50 + 1, 2 + 1)
