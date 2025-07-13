import re
from cProfile import label
from typing import Tuple

import esig
import numpy as np
import pandas as pd
import tqdm
from sklearn.cluster import HDBSCAN  # type: ignore
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import SparseRandomProjection

from autoanalyst.core import string_maps
from autoanalyst.core.base_classes import BaseAggregatorEnhancer, BaseCategoryAggregator


def _name_col(idx: int) -> str:
    """
    Generate a column name based on the index.
    """
    return f"col_{idx}"


def _contrib_weights(width: int, depth: int) -> np.ndarray:
    # str is of the form " 1 2 3 [1,2] [1,3]..."
    column_contrib_str: str = esig.logsigkeys(width, depth)
    chunks = column_contrib_str.strip().split()
    return np.array([len(chunk.split(",")) for idx, chunk in enumerate(chunks)])


def _pivoted_cholesky_linear(
    X: np.ndarray,
    max_rank: int = 30,
    tol: float = 1e-10,
) -> Tuple[list[int], np.ndarray]:
    """
    Low-rank approximation of the linear-kernel matrix K = X·Xᵀ using
    greedy (pivoted) Cholesky without ever forming K in memory.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Projected feature matrix (e.g. JL-compressed log-signatures).
    max_rank : int, default=30
        Target rank k of the approximation.
    tol : float, default=1e-10
        Stopping threshold on the residual diagonal.

    Returns
    -------
    pivots : List[int]
        Indices of the selected pivot rows (basis paths).
    L : np.ndarray, shape (n_samples, k)
        Cholesky factor such that  K ≈ L · Lᵀ.
        Each column is a normalised kernel column for a pivot.
    """
    n_samples = X.shape[0]

    # Diagonal of K for a linear kernel is just squared ℓ₂-norms of rows
    diag_residual = np.einsum("ij,ij->i", X, X).copy()

    pivot_indices: list[int] = []
    L_columns: list[np.ndarray] = []

    for _ in range(min(max_rank, n_samples)):
        pivot = int(np.argmax(diag_residual))
        pivot_indices.append(pivot)

        pivot_residual = diag_residual[pivot]
        if pivot_residual < tol:
            break  # remaining residuals are negligible

        # Full kernel column k(:, pivot) = X · X[pivot]ᵀ
        kernel_col = X @ X[pivot]

        # Orthogonalise against previous columns
        if L_columns:
            previous_cols = np.vstack(L_columns).T  # (n_samples, current_rank)
            correction = previous_cols @ previous_cols[pivot]
            kernel_col -= correction

        # Normalise new column
        new_col = kernel_col / np.sqrt(pivot_residual)
        L_columns.append(new_col)

        # Update residual diagonal:  diag_residual ← diag_residual − new_col²
        diag_residual -= new_col**2

    L = np.vstack(L_columns).T  # shape (n_samples, k)
    return pivot_indices, L


def signature_transform(
    data: pd.DataFrame,
    id_col: str,
    depth: int,
    projection_size: int = 1000,
    show_progress: bool = True,
) -> pd.DataFrame:
    weights = _contrib_weights(data.shape[1], depth)

    logsig_size = esig.logsigdim(data.shape[1], depth)
    use_projection = projection_size < logsig_size
    projector = SparseRandomProjection(
        n_components=projection_size,
        dense_output=True,
        random_state=42,
    )
    if use_projection:
        projector.fit(np.zeros((1, len(weights)), dtype=np.float32))
    else:
        projection_size = logsig_size

    ids = []
    outputs = []
    base_zeros = np.zeros((depth, data.shape[1]), dtype=np.float32)  # type: ignore

    iterator = (
        tqdm.tqdm(data.sort_index().groupby(id_col))
        if show_progress
        else data.sort_index().groupby(id_col)
    )
    for id_val, chunk in iterator:
        ids.append(id_val)
        stream = np.concatenate([base_zeros, chunk.values], axis=0)
        transformed = esig.stream2logsig(stream, depth=depth).astype(np.float32)
        transformed = np.sign(transformed) * (np.abs(transformed) ** (1 / weights))

        # Use random projection to prevent the very real risk of memory issues
        if use_projection:
            transformed = projector.transform(transformed.reshape(1, -1))
            transformed = transformed.reshape(-1)  # type: ignore

        outputs.append(transformed.astype(np.float32))

    id_index = pd.Index(ids, name=id_col)
    columns = [_name_col(i) for i in range(projection_size)]
    outputs = pd.DataFrame(outputs, index=id_index, columns=columns)
    return outputs


def _normalise_kernel_density(
    data, target_dist: float = 1.0, k_nearest=5, quantile: float = 0.5
) -> pd.DataFrame:
    nn = NearestNeighbors(n_neighbors=k_nearest).fit(data)
    d_k = nn.kneighbors(data, return_distance=True)[0][:, -1]
    median_d = np.quantile(d_k, q=quantile) + 1e-7
    return data * (target_dist / median_d)


def get_kernel_basis(
    data: pd.DataFrame, kernel_basis_size: int, eps: float = 1e-10
) -> pd.DataFrame:
    pivots, _ = _pivoted_cholesky_linear(
        data.values.astype(np.float32), max_rank=kernel_basis_size, tol=eps
    )
    basis_df = data.iloc[pivots].copy()
    X_basis = basis_df.values.astype(np.float32).T  # shape (k, n_features)
    X_users = data.values.astype(np.float32)
    kernel_basis = X_users @ X_basis  # shape (n_users, k)

    return pd.DataFrame(
        _normalise_kernel_density(kernel_basis),
        index=data.index,
        columns=[_name_col(i) for i in range(kernel_basis.shape[1])],
    )


class SignatureKernelAggregator(BaseCategoryAggregator):
    def __init__(
        self,
        preprocessors: list[BaseAggregatorEnhancer],
        signature_depth: int = 2,
        kernel_basis_size: int = 30,
        projection_size: int = 1000,
        membership_threshold: float = 0.66,
        id_col: str = string_maps.ID_COL,
        date_col: str = string_maps.DATE_COL,
        min_cluster_size: int | None = None,
        min_samples: int | None = None,
        show_progress: bool = True,
    ):
        super().__init__(id_col, date_col, preprocessors)
        self.signature_depth = signature_depth
        self.kernel_basis_size = kernel_basis_size
        self.projection_size = projection_size
        self.membership_threshold = membership_threshold
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.show_progress = show_progress

    def _map_to_density(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data by applying the preprocessors and transforming to signatures.
        """
        x_tx = signature_transform(
            data,
            self.id_col,
            self.signature_depth,
            self.projection_size,
            self.show_progress,
        )
        return get_kernel_basis(x_tx, self.kernel_basis_size)

    def _cluster_obs(self, data: pd.DataFrame):
        min_cluster_size = (
            int(np.sqrt(data.shape[0]))
            if self.min_cluster_size is None
            else self.min_cluster_size
        )
        min_samples = (
            int(np.log10(data.shape[0]) / 2)
            if self.min_samples is None
            else self.min_samples
        )

        clusterer = HDBSCAN(
            min_cluster_size=max(10, min_cluster_size),
            min_samples=max(1, min_samples),
            cluster_selection_epsilon=0.5,
            metric="euclidean",
            cluster_selection_method="eom",
            n_jobs=-1,
        )
        return pd.Series(
            clusterer.fit_predict(data),
            index=data.index,
            name="cluster_label",
        )

    def _agg_categories(
        self, categories: pd.DataFrame, labels: pd.Series
    ) -> dict[str, list[list[str]]]:
        """
        Aggregate the categories based on the clustering results.
        """
        most_recent = categories.sort_index().groupby(self.id_col).last()

        remapping: dict[str, list[list[str]]] = {}
        for column in most_recent.columns:
            counts = most_recent[column].value_counts()

            concentration = most_recent[column].groupby(labels).value_counts()
            concentration = concentration / concentration.index.get_level_values(1).map(
                counts
            )

            # Only keep categories that have a significant presence in the cluster
            concentration = (
                concentration[concentration > self.membership_threshold]
                .to_frame()
                .reset_index()
            )
            concentration.columns = ["cluster", "col", "count"]
            concentration = concentration[concentration.cluster != -1]

            # We only care about clusters with more than one category
            gt_one = concentration.cluster.value_counts()
            gt_one = gt_one[gt_one > 1].index

            concentration = concentration[concentration.cluster.isin(gt_one)]
            if len(concentration) == 0:
                # nothing to do, bail out
                continue

            mapping = (
                concentration.groupby("cluster")["col"]
                .apply(lambda x: x.tolist())
                .tolist()
            )
            remapping[column] = mapping

        return remapping

    def _coarsegrain_categories(
        self,
        categories: pd.DataFrame,
        category_group_mappings: dict[str, list[list[str]]],
    ) -> pd.DataFrame:
        """
        Coarse-grain the categories based on the provided mappings.
        """
        categories = categories.copy()
        for key, value in category_group_mappings.items():
            remapper = {}
            for graining in value:
                new_cat = f"{'--'.join(graining)}"
                remapper.update({col: new_cat for col in graining})
            categories[key] = categories[key].replace(to_replace=remapper)

        return categories

    def aggregate_raw(
        self,
        X: pd.DataFrame,
        categories: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series, dict[str, list[list[str]]]]:
        """
        Aggregate the data based on signatures.

        Parameters:
            X (pd.DataFrame): Features DataFrame.
            categories (pd.DataFrame): Categories DataFrame.

        Returns:
            dict[str, list[list[str]]]: Per-column categories that can be merged
        """
        x_tx = self.preprocess(X, categories)
        x_tx = self._map_to_density(x_tx.astype(np.float32))
        labels = self._cluster_obs(x_tx)
        mappings = self._agg_categories(categories, labels)

        return x_tx, labels, mappings

    def aggregate(
        self,
        X: pd.DataFrame,
        categories: pd.DataFrame,
    ) -> pd.DataFrame:
        _, _, mappings = self.aggregate_raw(X, categories)

        return self._coarsegrain_categories(categories, mappings)
