"""This file contains unit tests for the DataTransformer class."""

import numpy as np
import pandas as pd
import pytest
from rdt.transformers import ClusterBasedNormalizer, OneHotEncoder

from vect_gan.utils.data_transformer import DataTransformer


def test_fit_continuous() -> None:
    """Test fitting the DataTransformer on continuous data."""
    df = pd.DataFrame(
        {
            "cont_col": np.random.randn(100) * 10 + 50  # Some random continuous data
        }
    )

    transformer = DataTransformer()
    transformer.fit(df, discrete_columns=())
    assert (
        len(transformer._column_transform_info_list) == 1
    ), "Unexpected number of columns fitted."
    col_info = transformer._column_transform_info_list[0]
    assert col_info.column_type == "continuous", "Column type should be continuous."
    assert isinstance(
        col_info.transform, ClusterBasedNormalizer
    ), "Continuous column should be handled by ClusterBasedNormalizer."
    assert len(col_info.output_info) == 2, "Continuous columns should produce two output spans."


def test_fit_discrete() -> None:
    """Test fitting the DataTransformer on discrete data.

    This checks whether the transformer can identify discrete columns and generate the expected
    metadata.
    """
    df = pd.DataFrame({"disc_col": np.random.choice(["A", "B", "C"], 100)})

    transformer = DataTransformer()
    transformer.fit(df, discrete_columns=("disc_col",))
    assert (
        len(transformer._column_transform_info_list) == 1
    ), "Unexpected number of columns fitted."
    col_info = transformer._column_transform_info_list[0]
    assert col_info.column_type == "discrete", "Column type should be discrete."
    assert isinstance(
        col_info.transform, OneHotEncoder
    ), "Discrete column should be handled by OneHotEncoder."
    assert len(col_info.output_info) == 1, "Discrete columns should produce one output span."
    assert (
        col_info.output_info[0].activation_fn == "softmax"
    ), "Discrete output should use softmax."


def test_transform_continuous() -> None:
    """Test transforming continuous data after fitting."""
    df = pd.DataFrame({"cont_col": np.linspace(0, 100, 50)})

    transformer = DataTransformer()
    transformer.fit(df)

    transformed = transformer.transform(df)
    assert transformed.shape[0] == 50, "Unexpected number of rows after transform."
    assert (
        transformed.shape[1] == transformer.output_dimensions
    ), "Unexpected number of columns after transform."
    assert np.all(
        (transformed[:, 0] >= -1) & (transformed[:, 0] <= 1)
    ), "First continuous dimension not in [-1, 1]."


def test_transform_discrete() -> None:
    """Test transforming discrete data after fitting.

    We check that the transformed output is a valid one-hot encoding.
    """
    df = pd.DataFrame({"disc_col": np.random.choice(["A", "B", "C"], 100)})

    transformer = DataTransformer()
    transformer.fit(df, discrete_columns=("disc_col",))

    transformed = transformer.transform(df)
    assert transformed.shape[0] == 100, "Unexpected number of rows after transform."
    assert transformed.shape[1] == 3, "Unexpected number of columns in discrete transform."
    row_sums = transformed.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), "One-hot rows do not sum to 1."


def test_inverse_transform_continuous() -> None:
    """Test inverse transforming continuous data.

    We check that inverse_transform returns data close to the original range.
    """
    df = pd.DataFrame({"cont_col": np.random.randn(100) * 10 + 50})

    transformer = DataTransformer()
    transformer.fit(df)
    transformed = transformer.transform(df)
    recovered = transformer.inverse_transform(transformed)

    assert recovered.shape == df.shape, "Inverse transformed shape mismatch."
    assert (
        recovered["cont_col"].min() >= df["cont_col"].min() - 5
    ), "Inverse transformed values too low compared to original."
    assert (
        recovered["cont_col"].max() <= df["cont_col"].max() + 5
    ), "Inverse transformed values too high compared to original."


def test_inverse_transform_discrete() -> None:
    """Test inverse transforming discrete data.

    We check that the data is faithfully recovered as the original categories.
    """
    original = np.random.choice(["A", "B", "C"], 100)
    df = pd.DataFrame({"disc_col": original})

    transformer = DataTransformer()
    transformer.fit(df, discrete_columns=("disc_col",))
    transformed = transformer.transform(df)
    recovered = transformer.inverse_transform(transformed)

    assert np.array_equal(
        recovered["disc_col"].values, original
    ), "Discrete inverse transform mismatch."


def test_transform_numpy_input() -> None:
    """Test that the transformer works when given a numpy array as input."""
    data = np.random.randn(50, 2)  # Two continuous columns
    transformer = DataTransformer()
    transformer.fit(data)  # Fit on numpy input
    transformed = transformer.transform(data)
    assert transformed.shape[0] == 50, "Rows mismatch when transforming numpy input."
    assert (
        transformed.shape[1] >= 2
    ), "Continuous transform should yield at least 2 columns (tanh + component)."


def test_inverse_transform_numpy_output() -> None:
    """Test that inverse_transform returns a numpy array when the original input is not valid."""
    data = np.random.randn(50, 2)
    transformer = DataTransformer()
    transformer.fit(data)
    transformed = transformer.transform(data)
    recovered = transformer.inverse_transform(transformed)
    assert isinstance(
        recovered, np.ndarray
    ), "Inverse transform should return numpy array if input was numpy array."
    assert recovered.shape == data.shape, "Inverse transformed shape mismatch for numpy input."


def test_min_max_enforcement() -> None:
    """Test that the min/max enforcement works if enabled."""
    df = pd.DataFrame({"cont_col": np.random.randn(100)})

    transformer = DataTransformer(enforce_min_max_values=True)
    transformer.fit(df)
    transformed = transformer.transform(df)

    transformed[:, 0] = np.clip(transformed[:, 0], -10, 10)

    recovered = transformer.inverse_transform(transformed)

    col_name = transformer._column_transform_info_list[0].column_name
    min_val = transformer._min_max_values[col_name]["min"]
    max_val = transformer._min_max_values[col_name]["max"]

    assert recovered.iloc[:, 0].min() >= min_val, "Values below enforced min."
    assert recovered.iloc[:, 0].max() <= max_val, "Values above enforced max."


def test_convert_column_name_value_to_id_discrete() -> None:
    """Test convert_column_name_value_to_id for discrete columns."""
    df = pd.DataFrame({"disc_col": np.random.choice(["A", "B", "C"], 100)})

    transformer = DataTransformer()
    transformer.fit(df, discrete_columns=("disc_col",))
    res = transformer.convert_column_name_value_to_id("disc_col", "A")
    assert "discrete_column_id" in res, "Missing 'discrete_column_id' in result."
    assert "column_id" in res, "Missing 'column_id' in result."
    assert "value_id" in res, "Missing 'value_id' in result."


def test_convert_column_name_value_to_id_error() -> None:
    """Test that convert_column_name_value_to_id raises a ValueError correctly."""
    df = pd.DataFrame({"disc_col": ["A", "A", "B", "C"]})

    transformer = DataTransformer()
    transformer.fit(df, discrete_columns=("disc_col",))

    with pytest.raises(ValueError):
        transformer.convert_column_name_value_to_id("non_existent_col", "A")

    with pytest.raises(ValueError):
        transformer.convert_column_name_value_to_id("disc_col", "D")
