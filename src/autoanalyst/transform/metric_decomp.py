import itertools

import pandas as pd

from autoanalyst.core import string_maps
from autoanalyst.core.base_classes import BaseTransformer

# _DIFF_TAG = "_diff"
# _LAG_TAG = "_lag"


def _map_col(col: str, is_diff: bool) -> str:
    return col + (string_maps.DIFF_SUFFIX if is_diff else string_maps.LAG_SUFFIX)


def decompose_funnel_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Decompose arbitrary funnel metrics into top-level KPI-valued contributions.

    Yields an application of the metric decomposition formula:
        $$
        KPI_t =
            KPI_{t-1} + \\sum_{i=1}^{n} contrib(metric_t)
        $$
    where `contrib(metric_t)` is the contribution of the metric at time `t` to the KPI.

    The return value of this function represents the contributions.

    Args:
        df: DataFrame with columns [date_col, group_cols, metrics]

    Returns:
        DataFrame with columns [date_col, group_cols, metrics, kpi, kpi_contribution]
    """
    group_col = [string_maps.ID_COL]
    funnel_cols = df.columns.tolist()
    df = pd.concat(
        [
            df.groupby(group_col).shift(1).add_suffix(string_maps.LAG_SUFFIX),
            df.groupby(group_col).diff(1).add_suffix(string_maps.DIFF_SUFFIX),
        ],
        axis=1,
    ).fillna(0)

    # Get the cartesian product of all possible combinations of True/False values
    # For our non-target columns below, these will forn the "other" contributions
    # of our n-dimensional prism. The target columm's contribution is always
    # True, as we only consider the diff component and subtract off the lagged
    # value.
    combinations = list(itertools.product(*[(True, False)] * (len(funnel_cols) - 1)))

    # Build each column's total contribution one by one
    output = []
    for target_col in funnel_cols:
        # Mark the first column as the fixed diff, as the element of all lags
        # ends up subtracted off we just need to pin the values we use
        other_cols = [col for col in funnel_cols if col != target_col]
        accumulator = []

        # This loop handles the explosion of terms
        for combo in combinations:
            # Build the contribution of a single direction of the prism
            contrib_cols = [_map_col(target_col, True)] + [
                _map_col(*item) for item in zip(other_cols, combo)
            ]

            # Each subprism is shared amoung all subprisms that contribute delta;
            # rather than make heuristic arguments about assingment priority we
            # just spread the shared delta across all contributing metrics

            # The sharing factor is the reciprocol of the number of diff cols in
            # the equation
            sharing_factor = 1 / (sum(combo) + 1)
            contribution = (
                df[contrib_cols].prod(axis=1) * sharing_factor  # type: ignore
            )

            accumulator.append(contribution)
        # Sum together the contributions
        output.append(pd.concat(accumulator, axis=1).sum(axis=1).rename(target_col))

    return pd.concat(output, axis=1).fillna(0)


class MetricDecompTransform(BaseTransformer):
    """
    Transformer for decomposing metrics data.
    """

    def __init__(
        self, id_col: str = string_maps.ID_COL, date_col: str = string_maps.DATE_COL
    ):
        """
        Initialize the MetricDecompTransform.
        """
        super().__init__(id_col=id_col, date_col=date_col)

    def fit(self, X, y):
        """
        Fit the transformer to the data.
        """
        # No fitting required for decomposition
        return super().fit(X, y)

    def transform(self, X, y):
        """
        Transform the data by decomposing metrics.
        """
        transformed = decompose_funnel_metrics(
            df=X,
        )

        transformed[string_maps.RESIDUAL_COL] = y - transformed.sum(axis=1)
        return transformed
