import numpy as np
import pandas as pd

from autoanalyst.core.base_classes import BaseUnitConversionStrategy


class SumOfWholeStrategy(BaseUnitConversionStrategy):
    """
    Strategy to convert units by summing to the whole.

    This strategy makes sense when fixing zero denominators makes sense
    by filling with zeroes, to avoid null-space issues.
    """

    def id_compatible_transform(
        self, X: pd.DataFrame, y: pd.Series, target: pd.Series
    ) -> pd.DataFrame:
        """
        Convert units for ID-compatible data, assigning equal values of the target
        column to each ID
        """
        # A basic strategy is to break the child xs into contributions towards y, which
        # should be near zero, and then sum them up to get the target.
        # By construction, X (with its resids) sums to y, so we can use y in the
        # denominator to normalise the contributions. We then multiply by Y
        # TODO: Figure out a more robust way to handle zeros
        print("REMOVE ME")
        print(X.div(y, axis=0).mul(target, axis=0).head())
        return X.div(y.replace(0, np.nan), axis=0).mul(target, axis=0).fillna(0)

    def grain_change_transform(
        self, X: pd.DataFrame, y: pd.Series, target: pd.Series
    ) -> pd.DataFrame:
        """
        Convert units for grain change data by summing the whole.
        """
        # We can't use IDs here, so we use time to aggregate the data.
        # So; we normalise Xs by the MONTHLY Y, so that multiplying by the monthly
        # target gives us the correct value.
        agg_ys = y.groupby(self.date_col).sum()
        agg_targets = target.groupby(self.date_col).sum()

        # Now we map to dates
        date_ys = (
            y.index.get_level_values(self.date_col)
            .to_series()
            .map(agg_ys)
            .set_axis(y.index)
        )
        date_targets = (
            y.index.get_level_values(self.date_col)
            .to_series()
            .map(agg_targets)
            .set_axis(y.index)
        )

        # Now the fun bit. We need to normalise and then multiply
        return (
            X.div(date_ys.replace(0, np.nan), axis=0)
            .mul(date_targets, axis=0)
            .fillna(0)
        )
