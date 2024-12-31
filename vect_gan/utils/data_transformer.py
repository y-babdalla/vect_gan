"""DataTransformer module.

Contains the DataTransformer class, which models continuous columns with a Bayesian GMM
and normalises them to a scalar between [-1, 1] and a vector. Discrete columns are encoded
using a OneHotEncoder.
"""

from collections import namedtuple
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdt.transformers import ClusterBasedNormalizer, OneHotEncoder
from tqdm import tqdm

SpanInfo = namedtuple("SpanInfo", ["dim", "activation_fn"])  # noqa: PYI024
ColumnTransformInfo = namedtuple(  # noqa: PYI024
    "ColumnTransformInfo",
    ["column_name", "column_type", "transform", "output_info", "output_dimensions"],
)


class DataTransformer:
    """Data Transformer.

    Models continuous columns with a BayesianGMM and normalises them to a scalar
    between [-1, 1] and a vector. Discrete columns are encoded using a OneHotEncoder.
    """

    def __init__(
        self,
        max_clusters: int = 10,
        weight_threshold: float = 0.005,
        enforce_min_max_values: bool = True,
    ) -> None:
        """Create a data transformer.

        Args:
            max_clusters: Maximum number of Gaussian distributions in Bayesian GMM.
            weight_threshold: Weight threshold for a Gaussian distribution to be kept.
            enforce_min_max_values: Whether to enforce min/max values from training data.
        """
        self._max_clusters: int = max_clusters
        self._weight_threshold: float = weight_threshold
        self._enforce_min_max_values: bool = enforce_min_max_values
        self._min_max_values: dict[str, dict[str, float]] = {}
        self.output_info_list: list[list[SpanInfo]] = []
        self.output_dimensions: int = 0
        self.dataframe: bool = False
        self._column_raw_dtypes: pd.Series | None = None
        self._column_transform_info_list: list[ColumnTransformInfo] = []

    def _fit_continuous(self, data: pd.DataFrame) -> ColumnTransformInfo:
        """Train a Bayesian GMM for continuous columns.

        Args:
            data: A DataFrame containing a single continuous column.

        Returns:
            A ColumnTransformInfo object with transformer details.
        """
        column_name = data.columns[0]

        if self._enforce_min_max_values:
            self._min_max_values[column_name] = {
                "min": float(data[column_name].min()),
                "max": float(data[column_name].max()),
            }

        gm = ClusterBasedNormalizer(
            missing_value_generation="from_column",
            max_clusters=min(len(data), self._max_clusters),
            weight_threshold=self._weight_threshold,
        )
        gm.fit(data, column_name)
        num_components = sum(gm.valid_component_indicator)

        return ColumnTransformInfo(
            column_name=column_name,
            column_type="continuous",
            transform=gm,
            output_info=[SpanInfo(1, "tanh"), SpanInfo(num_components, "softmax")],
            output_dimensions=1 + num_components,
        )

    @staticmethod
    def _fit_discrete(data: pd.DataFrame) -> ColumnTransformInfo:
        """Fit a one-hot encoder for a discrete column.

        Args:
            data: A DataFrame containing a single discrete column.

        Returns:
            A ColumnTransformInfo object with transformer details.
        """
        column_name = data.columns[0]
        ohe = OneHotEncoder()
        ohe.fit(data, column_name)
        num_categories = len(ohe.dummies)

        return ColumnTransformInfo(
            column_name=column_name,
            column_type="discrete",
            transform=ohe,
            output_info=[SpanInfo(num_categories, "softmax")],
            output_dimensions=num_categories,
        )

    def fit(
        self, raw_data: pd.DataFrame | np.ndarray, discrete_columns: tuple[str | int, ...] = ()
    ) -> None:
        """Fit the DataTransformer.

        Args:
            raw_data: The input raw data, either a DataFrame or NumPy array.
            discrete_columns: Columns to treat as discrete (by name or index).
        """
        self.output_info_list = []
        self.output_dimensions = 0
        self.dataframe = True

        if not isinstance(raw_data, pd.DataFrame):
            self.dataframe = False
            discrete_columns = [str(column) for column in discrete_columns]
            column_names = [str(num) for num in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=column_names)

        self._column_raw_dtypes = raw_data.infer_objects().dtypes
        self._column_transform_info_list = []

        for column_name in tqdm(raw_data.columns, desc="Fitting data transformer"):
            if column_name in discrete_columns:
                column_transform_info = self._fit_discrete(raw_data[[column_name]])
            else:
                column_transform_info = self._fit_continuous(raw_data[[column_name]])
            self.output_info_list.append(column_transform_info.output_info)
            self.output_dimensions += column_transform_info.output_dimensions
            self._column_transform_info_list.append(column_transform_info)

    @staticmethod
    def _transform_continuous(
        column_transform_info: ColumnTransformInfo, data: pd.DataFrame
    ) -> np.ndarray:
        """Transform a continuous column using the fitted Bayesian GMM.

        Args:
            column_transform_info: Details about the column's transformer.
            data: Data containing a single column to transform.

        Returns:
            A NumPy array of transformed data for the continuous column.
        """
        column_name = data.columns[0]
        flattened_column = data[column_name].to_numpy().flatten()
        data = data.assign(**{column_name: flattened_column})
        gm: ClusterBasedNormalizer = column_transform_info.transform
        transformed = gm.transform(data)

        # Converts the transformed data to the appropriate output format.
        # The first column (ending in '.normalized') stays the same,
        # but the label encoded column (ending in '.component') is one hot encoded.
        output = np.zeros((len(transformed), column_transform_info.output_dimensions))
        output[:, 0] = transformed[f"{column_name}.normalized"].to_numpy()
        index = transformed[f"{column_name}.component"].to_numpy().astype(int)
        output[np.arange(index.size), index + 1] = 1.0

        return output

    @staticmethod
    def _transform_discrete(
        column_transform_info: ColumnTransformInfo, data: pd.DataFrame
    ) -> np.ndarray:
        """Transform a discrete column using a one-hot encoder.

        Args:
            column_transform_info: Details about the column's transformer.
            data: Data containing a single column to transform.

        Returns:
            A NumPy array of transformed data for the discrete column.
        """
        ohe = column_transform_info.transform
        return ohe.transform(data).to_numpy()

    def _synchronous_transform(
        self, raw_data: pd.DataFrame, column_transform_info_list: list[ColumnTransformInfo]
    ) -> list[np.ndarray]:
        """Transform columns in a synchronous (sequential) manner.

        Args:
            raw_data: The raw data containing all columns.
            column_transform_info_list: List of ColumnTransformInfo objects.

        Returns:
            A list of NumPy arrays with transformed columns.
        """
        column_data_list: list[np.ndarray] = []
        for column_transform_info in column_transform_info_list:
            column_name = column_transform_info.column_name
            data = raw_data[[column_name]]

            if column_transform_info.column_type == "continuous":
                column_data_list.append(self._transform_continuous(column_transform_info, data))
            else:
                column_data_list.append(self._transform_discrete(column_transform_info, data))

        return column_data_list

    def _parallel_transform(
        self, raw_data: pd.DataFrame, column_transform_info_list: list[ColumnTransformInfo]
    ) -> list[np.ndarray]:
        """Transform columns in parallel using joblib.

        Args:
            raw_data: The raw data containing all columns.
            column_transform_info_list: List of ColumnTransformInfo objects.

        Returns:
            A list of NumPy arrays with transformed columns.
        """
        processes = []
        for column_transform_info in column_transform_info_list:
            column_name = column_transform_info.column_name
            data = raw_data[[column_name]]

            if column_transform_info.column_type == "continuous":
                process = delayed(self._transform_continuous)(column_transform_info, data)
            else:
                process = delayed(self._transform_discrete)(column_transform_info, data)
            processes.append(process)

        return Parallel(n_jobs=-1)(processes)

    def transform(self, raw_data: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Take raw data and output transformed matrix data.

        Args:
            raw_data: The raw data to be transformed (DataFrame or NumPy array).

        Returns:
            A NumPy array containing the transformed data.
        """
        if not isinstance(raw_data, pd.DataFrame):
            column_names = [str(num) for num in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=column_names)

        # Only use parallelization with larger data sizes.
        if raw_data.shape[0] < 500:
            column_data_list = self._synchronous_transform(
                raw_data, self._column_transform_info_list
            )
        else:
            column_data_list = self._parallel_transform(raw_data, self._column_transform_info_list)

        return np.concatenate(column_data_list, axis=1).astype(float)

    def _inverse_transform_continuous(
        self,
        column_transform_info: ColumnTransformInfo,
        column_data: np.ndarray,
        sigmas: np.ndarray | None,
        st: int,
    ) -> pd.Series:
        """Inverse transform for continuous data.

        Args:
            column_transform_info: Details about the column's transformer.
            column_data: Transformed continuous data (with normalised and component columns).
            sigmas: Optional noise levels for adding Gaussian noise.
            st: Index offset for sigmas in multi-column transformations.

        Returns:
            A pandas Series with the recovered continuous column.
        """
        gm = column_transform_info.transform
        data = pd.DataFrame(column_data[:, :2], columns=list(gm.get_output_sdtypes())).astype(
            float
        )
        data[data.columns[1]] = np.argmax(column_data[:, 1:], axis=1)

        if sigmas is not None:
            selected_normalized_value = np.random.normal(data.iloc[:, 0], sigmas[st])
            data.iloc[:, 0] = selected_normalized_value

        recovered = gm.reverse_transform(data)

        if "_enforce_min_max_values" in dir(self) and self._enforce_min_max_values:
            column_name = column_transform_info.column_name
            if column_name in self._min_max_values:
                bounds = self._min_max_values[column_name]
                recovered = recovered.clip(lower=bounds["min"], upper=bounds["max"])

        return recovered

    @staticmethod
    def _inverse_transform_discrete(
        column_transform_info: ColumnTransformInfo, column_data: np.ndarray
    ) -> pd.Series:
        """Inverse transform for discrete data.

        Args:
            column_transform_info: Details about the column's transformer.
            column_data: Transformed discrete data (one-hot encoded).

        Returns:
            A pandas Series with the recovered discrete column.
        """
        ohe = column_transform_info.transform
        data = pd.DataFrame(column_data, columns=list(ohe.get_output_sdtypes()))
        return ohe.reverse_transform(data)[column_transform_info.column_name]

    def inverse_transform(
        self, data: np.ndarray, sigmas: np.ndarray | None = None
    ) -> pd.DataFrame | np.ndarray:
        """Take transformed matrix data and output raw data.

        Args:
            data: The transformed data as a NumPy array.
            sigmas: Optional array of noise levels for each continuous column.

        Returns:
            A pandas DataFrame or a NumPy array (depending on the original input type)
            containing the recovered data.
        """
        st = 0
        recovered_column_data_list: list[pd.Series] = []
        column_names: list[str] = []

        for column_transform_info in self._column_transform_info_list:
            dim = column_transform_info.output_dimensions
            column_data = data[:, st : st + dim]

            if column_transform_info.column_type == "continuous":
                recovered_column_data = self._inverse_transform_continuous(
                    column_transform_info, column_data, sigmas, st
                )
            else:
                recovered_column_data = self._inverse_transform_discrete(
                    column_transform_info, column_data
                )

            recovered_column_data_list.append(recovered_column_data)
            column_names.append(column_transform_info.column_name)
            st += dim

        recovered_data = np.column_stack(recovered_column_data_list)
        recovered_data = pd.DataFrame(recovered_data, columns=column_names).astype(
            self._column_raw_dtypes
        )

        if "_enforce_min_max_values" in dir(self) and self._enforce_min_max_values:
            for column_name, bounds in self._min_max_values.items():
                if column_name in recovered_data.columns:
                    recovered_data[column_name] = recovered_data[column_name].clip(
                        lower=bounds["min"], upper=bounds["max"]
                    )

        if not self.dataframe:
            recovered_data = recovered_data.to_numpy()

        return recovered_data

    def convert_column_name_value_to_id(self, column_name: str, value: Any) -> dict[str, int]:
        """Get the IDs of the given column_name.

        Args:
            column_name: Name of the column to look up.
            value: The value in the column for which to retrieve IDs.

        Returns:
            A dictionary containing discrete_column_id, column_id, and value_id.

        Raises:
            ValueError: If the column does not exist or the value is not in the column.
        """
        discrete_counter = 0
        column_id = 0
        for column_transform_info in self._column_transform_info_list:
            if column_transform_info.column_name == column_name:
                break
            if column_transform_info.column_type == "discrete":
                discrete_counter += 1

            column_id += 1

        else:
            raise ValueError(f"The column_name `{column_name}` doesn't exist in the data.")

        ohe = column_transform_info.transform
        data = pd.DataFrame([value], columns=[column_transform_info.column_name])
        one_hot = ohe.transform(data).to_numpy()[0]
        if sum(one_hot) == 0:
            raise ValueError(f"The value `{value}` doesn't exist in the column `{column_name}`.")

        return {
            "discrete_column_id": discrete_counter,
            "column_id": column_id,
            "value_id": int(np.argmax(one_hot)),
        }
