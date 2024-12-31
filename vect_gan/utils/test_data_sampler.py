"""This file contains unit tests for the DataSampler class."""

from typing import NamedTuple

import numpy as np

from vect_gan.utils.data_sampler import DataSampler


class SpanInfo(NamedTuple):
    """Named tuple mimicking the SpanInfo used by DataTransformer."""

    dim: int
    activation_fn: str


def test_data_sampler_init_single_discrete_column() -> None:
    """Test initialisation with a single discrete column."""
    output_info = [[SpanInfo(3, "softmax")]]
    data = np.zeros((100, 3))
    for i in range(100):
        cat = np.random.randint(0, 3)
        data[i, cat] = 1

    sampler = DataSampler(data, output_info, log_frequency=True)
    assert sampler._n_discrete_columns == 1, "Should have one discrete column."
    assert sampler.dim_cond_vec() == 3, "dim_cond_vec should match the total number of categories."


def test_data_sampler_init_multiple_discrete_columns() -> None:
    """Test initialisation with multiple discrete columns."""
    output_info = [[SpanInfo(3, "softmax")], [SpanInfo(2, "softmax")]]

    data = np.zeros((200, 5))
    for i in range(200):
        cat1 = np.random.randint(0, 3)
        cat2 = np.random.randint(0, 2)
        data[i, cat1] = 1
        data[i, 3 + cat2] = 1

    sampler = DataSampler(data, output_info, log_frequency=True)
    assert sampler._n_discrete_columns == 2, "Should have two discrete columns."
    assert sampler.dim_cond_vec() == 5, "Total categories = 3 + 2 = 5."


def test_sample_condvec() -> None:
    """Test sample_condvec with multiple discrete columns."""
    output_info = [[SpanInfo(3, "softmax")], [SpanInfo(2, "softmax")]]

    data = np.zeros((10, 5))
    for i in range(10):
        cat1 = np.random.randint(0, 3)
        cat2 = np.random.randint(0, 2)
        data[i, cat1] = 1
        data[i, 3 + cat2] = 1

    sampler = DataSampler(data, output_info, log_frequency=True)
    batch = 4
    res = sampler.sample_condvec(batch)
    assert res is not None, "sample_condvec should return a tuple."
    cond, mask, discrete_column_id, category_id_in_col = res
    assert cond.shape == (batch, 5), "cond should be batch x total_categories"
    assert mask.shape == (batch, 2), "mask should be batch x n_discrete_columns"
    assert discrete_column_id.shape == (batch,), "discrete_column_id should be a 1D array"
    assert category_id_in_col.shape == (batch,), "category_id_in_col should be a 1D array"

    assert np.all(cond.sum(axis=1) == 1), "Each cond vector row should have exactly one 1."
    assert np.all(mask.sum(axis=1) == 1), "Each mask row should have exactly one 1."


def test_sample_original_condvec() -> None:
    """Test sample_original_condvec."""
    output_info = [[SpanInfo(3, "softmax")]]

    data = np.zeros((100, 3))
    for i in range(100):
        cat = 0 if i < 70 else np.random.randint(0, 3)
        data[i, cat] = 1

    sampler = DataSampler(data, output_info, log_frequency=False)
    batch = 10
    cond = sampler.sample_original_condvec(batch)
    assert cond.shape == (batch, 3), "cond vector must have shape batch x total_categories"
    assert np.all(cond.sum(axis=1) == 1), "Each row should have exactly one category set."


def test_dim_cond_vec() -> None:
    """Test dim_cond_vec just returns sum of categories across discrete columns."""
    output_info = [[SpanInfo(3, "softmax")], [SpanInfo(2, "softmax")]]
    data = np.zeros((10, 5))
    for i in range(10):
        cat1 = np.random.randint(0, 3)
        cat2 = np.random.randint(0, 2)
        data[i, cat1] = 1
        data[i, 3 + cat2] = 1

    sampler = DataSampler(data, output_info, log_frequency=False)
    assert sampler.dim_cond_vec() == 5, "3 categories + 2 categories = 5"


def test_generate_cond_from_condition_column_info() -> None:
    """Test generate_cond_from_condition_column_info."""
    output_info = [[SpanInfo(3, "softmax")]]
    data = np.zeros((20, 3))
    for i in range(20):
        cat = np.random.randint(0, 3)
        data[i, cat] = 1

    sampler = DataSampler(data, output_info, log_frequency=False)

    condition_info = {"discrete_column_id": 0, "column_id": 0, "value_id": 1}

    batch = 5
    cond_vec = sampler.generate_cond_from_condition_column_info(condition_info, batch)
    assert cond_vec.shape == (batch, 3), "Shape must match total categories in column."
    assert np.all(
        cond_vec == np.array([0, 1, 0], dtype="float32")
    ), "All rows must have category 1 set."
