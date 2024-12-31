"""DataSampler module.

Contains the DataSampler class, which samples conditional vectors
and corresponding data for CTGAN.
"""

from typing import Any

import numpy as np


class DataSampler:
    """DataSampler samples the conditional vector and corresponding data for CTGAN.

    This class is responsible for generating conditional vectors based on
    discrete columns, sampling data that satisfy specific conditions, and
    preparing information related to the discrete columns' distribution
    for CTGAN training.
    """

    def __init__(
        self, data: np.ndarray, output_info: list[list[Any]], log_frequency: bool
    ) -> None:
        """Initialise the DataSampler.

        Args:
            data: The training data, expected to be a NumPy array where each
                row corresponds to a sample and columns correspond to features.
            output_info: A nested list describing each column and its
                corresponding activation function (e.g. "softmax" or "tanh").
            log_frequency: Whether to apply log-scaling to frequencies of
                discrete values before normalising to probabilities.
        """
        self._data_length: int = len(data)

        def is_discrete_column(column_info: list[Any]) -> bool:
            return len(column_info) == 1 and column_info[0].activation_fn == "softmax"

        n_discrete_columns = sum(
            1 for column_info in output_info if is_discrete_column(column_info)
        )

        self._discrete_column_matrix_st = np.zeros(n_discrete_columns, dtype="int32")

        # Store the row id for each category in each discrete column.
        # For example _rid_by_cat_cols[a][b] is a list of all rows with the
        # a-th discrete column equal value b.
        self._rid_by_cat_cols = []

        # Compute _rid_by_cat_cols
        st = 0
        for column_info in output_info:
            if is_discrete_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim

                rid_by_cat = []
                for j in range(span_info.dim):
                    rid_by_cat.append(np.nonzero(data[:, st + j])[0])
                self._rid_by_cat_cols.append(rid_by_cat)
                st = ed
            else:
                st += sum([span_info.dim for span_info in column_info])
        assert st == data.shape[1]

        # Prepare an interval matrix for efficiently sample conditional vector
        max_category = max(
            [column_info[0].dim for column_info in output_info if is_discrete_column(column_info)],
            default=0,
        )

        self._discrete_column_cond_st = np.zeros(n_discrete_columns, dtype="int32")
        self._discrete_column_n_category = np.zeros(n_discrete_columns, dtype="int32")
        self._discrete_column_category_prob = np.zeros((n_discrete_columns, max_category))
        self._n_discrete_columns = n_discrete_columns
        self._n_categories = sum(
            [column_info[0].dim for column_info in output_info if is_discrete_column(column_info)]
        )

        st = 0
        current_id = 0
        current_cond_st = 0
        for column_info in output_info:
            if is_discrete_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim
                category_freq = np.sum(data[:, st:ed], axis=0)
                if log_frequency:
                    category_freq = np.log(category_freq + 1)
                sum_category_freq = np.sum(category_freq)
                if sum_category_freq == 0:
                    category_prob = np.zeros_like(category_freq)
                else:
                    category_prob = category_freq / sum_category_freq
                self._discrete_column_category_prob[current_id, : span_info.dim] = category_prob
                self._discrete_column_cond_st[current_id] = current_cond_st
                self._discrete_column_n_category[current_id] = span_info.dim
                current_cond_st += span_info.dim
                current_id += 1
                st = ed
            else:
                st += sum([span_info.dim for span_info in column_info])

    def _random_choice_prob_index(self, discrete_column_id: int) -> np.ndarray:
        """Return random category indices for a given discrete column using column probabilities.

        Args:
            discrete_column_id: The index of the discrete column.

        Returns:
            A 1-D array of randomly chosen categories for each row.
        """
        probs = self._discrete_column_category_prob[discrete_column_id]
        r = np.expand_dims(np.random.rand(probs.shape[0]), axis=1)
        return (probs.cumsum(axis=1) > r).argmax(axis=1)

    def sample_condvec(
        self, batch: int
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Generate the conditional vector for training.

        Args:
            batch: Number of samples to create a conditional vector for.

        Returns:
            A tuple containing:
            - cond: The conditional vector of shape (batch, #categories).
            - mask: A one-hot vector of shape (batch, #discrete_columns)
              indicating the selected discrete column.
            - discrete_column_id: A 1-D array of shape (batch,) with the
              selected column indices.
            - category_id_in_col: A 1-D array of shape (batch,) with the
              selected category within the chosen column.

            If there are no discrete columns, returns None.
        """
        if self._n_discrete_columns == 0:
            return None

        discrete_column_id = np.random.choice(np.arange(self._n_discrete_columns), batch)

        cond = np.zeros((batch, self._n_categories), dtype="float32")
        mask = np.zeros((batch, self._n_discrete_columns), dtype="float32")
        mask[np.arange(batch), discrete_column_id] = 1

        category_id_in_col = self._random_choice_prob_index(discrete_column_id)
        category_id = self._discrete_column_cond_st[discrete_column_id] + category_id_in_col

        cond[np.arange(batch), category_id] = 1

        return cond, mask, discrete_column_id, category_id_in_col

    def sample_original_condvec(self, batch: int) -> np.ndarray | None:
        """Generate the conditional vector using original frequency distribution.

        Args:
            batch: Number of samples to create a conditional vector for.

        Returns:
            The conditional vector if there are discrete columns, otherwise None.
        """
        if self._n_discrete_columns == 0:
            return None

        category_freq = self._discrete_column_category_prob.flatten()
        category_freq = category_freq[category_freq != 0]
        category_freq = category_freq / np.sum(category_freq)

        col_idxs = np.random.choice(np.arange(len(category_freq)), batch, p=category_freq)
        cond = np.zeros((batch, self._n_categories), dtype="float32")
        cond[np.arange(batch), col_idxs] = 1
        return cond

    def sample_data(
        self, data: np.ndarray, n: int, col: list[int] | None, opt: list[int] | None
    ) -> np.ndarray:
        """Sample data from the original training data satisfying a sampled conditional vector.

        Args:
            data: The training data.
            n: Number of rows to sample.
            col: List of column indices to condition on, or None if no conditioning.
            opt: List of category indices for each column in `col`, or None if no conditioning.

        Returns:
            A NumPy array of shape (n, data.shape[1]) with sampled rows.
        """
        if col is None:
            idx = np.random.randint(len(data), size=n)
            return data[idx]

        idx_list = []
        for c, o in zip(col, opt, strict=False):
            idx_list.append(np.random.choice(self._rid_by_cat_cols[c][o]))

        return data[idx_list]

    def dim_cond_vec(self) -> int:
        """Return the total number of categories.

        Returns:
            The total number of categories across all discrete columns.
        """
        return self._n_categories

    def generate_cond_from_condition_column_info(
        self, condition_info: dict[str, int], batch: int
    ) -> np.ndarray:
        """Generate a condition vector based on specified column info for discrete columns.

        Args:
            condition_info: Dictionary containing:
                - "discrete_column_id": The integer ID of the discrete column.
                - "value_id": The category ID within that column.
            batch: Number of samples for which to generate identical condition vectors.

        Returns:
            A NumPy array of shape (batch, self._n_categories) with a one-hot
            vector for the specified column/category.
        """
        vec = np.zeros((batch, self._n_categories), dtype="float32")
        id_ = self._discrete_column_matrix_st[condition_info["discrete_column_id"]]
        id_ += condition_info["value_id"]
        vec[:, id_] = 1
        return vec
