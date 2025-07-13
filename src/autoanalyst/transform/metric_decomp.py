import itertools

import numpy as np
import pandas as pd

from autoanalyst.core import string_maps
from autoanalyst.core.base_classes import BaseTransformer, BaseUnitConversionStrategy
from autoanalyst.transform.unit_conversion_strategies import (
    SumOfWholeStrategy,
)


def _map_col(col: str, is_diff: bool) -> str:
    return col + (string_maps.DIFF_SUFFIX if is_diff else string_maps.LAG_SUFFIX)


def _padded_diff(inner):

    return pd.DataFrame(
        np.diff(inner, prepend=0, axis=0),
        index=inner.index,
        columns=inner.columns,
    )


def _prep_cols(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:

    df = pd.concat(
        [
            df.groupby(group_cols).shift(1).add_suffix(string_maps.LAG_SUFFIX),
            df.groupby(group_cols)
            .apply(_padded_diff)
            .add_suffix(string_maps.DIFF_SUFFIX)
            .droplevel(0),
        ],
        axis=1,
    ).fillna(0)

    return df


def decompose_funnel_metrics(
    df: pd.DataFrame,
    funnel_cols: list[str],
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
        self,
        id_col: str = string_maps.ID_COL,
        date_col: str = string_maps.DATE_COL,
        unit_conversion_strategy: BaseUnitConversionStrategy | None = None,
    ):
        """
        Initialize the MetricDecompTransform.
        """

        unit_converter = (
            SumOfWholeStrategy(id_col=id_col, date_col=date_col)
            if unit_conversion_strategy is None
            else unit_conversion_strategy
        )
        super().__init__(
            id_col=id_col, date_col=date_col, unit_conversion_strategy=unit_converter
        )

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

        #
        group_col = [self.id_col]
        funnel_cols = X.columns.tolist()
        # return _prep_cols(X, group_col, self.multiplier_cols)

        # The metric decomp is a stateless transformation, so we can just
        # apply it directly to the input DataFrame.
        transformed = decompose_funnel_metrics(
            df=_prep_cols(X, group_col),
            funnel_cols=funnel_cols,
        )
        # The output of the decomp is a set of deltas, all in the top-level KPI
        # column's units. We need to recover the origial scale (i.e. not diffs)
        # by cumulatively summing the deltas then adding the value at t=0
        # Rounding is done to avoid floating point precision issues
        transformed = transformed.groupby(self.id_col).cumsum().round(9)

        # Finally, we need to add the residual column, which is the difference
        # between the original metric and the sum of the contributions.
        transformed[string_maps.RESIDUAL_COL] = y - transformed.sum(axis=1)
        return transformed

    def map_units(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target: pd.Series,
        is_id_compatible: bool = False,
    ) -> pd.DataFrame:
        """
        Map units of the data. This is a no-op by default.
        """
        return self.unit_conversion(
            X=X,
            y=y,
            target=target,
            is_id_compatible=is_id_compatible,
        )
