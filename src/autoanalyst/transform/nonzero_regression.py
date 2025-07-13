import numpy as np
import pandas as pd
from sklearn import linear_model

from autoanalyst.core import string_maps
from autoanalyst.core.base_classes import BaseTransformer, BaseUnitConversionStrategy
from autoanalyst.core.exceptions import TransformNotFittedError
from autoanalyst.transform.unit_conversion_strategies.sum_of_whole import (
    SumOfWholeStrategy,
)


def _get_monthly_coefs(old_coefs, y, x) -> pd.Series:
    count = len(y)
    if count < 2:
        # Can only pass forward the old coefficients
        return old_coefs
    if count == 2:
        # "Refit" the bias term only by manually calculating the residual
        # This is a special case where we have only two observations
        # and we want to keep the old coefficients for the other columns
        new_coefs = old_coefs.copy()
        short_cols = [col for col in x.columns if col != string_maps.CONSTANT_COL]
        pred = x[short_cols].mul(old_coefs[short_cols]).sum(axis=1)
        resid = y - pred
        new_coefs[string_maps.CONSTANT_COL] = resid.mean()
        return new_coefs

    model = linear_model.Ridge(
        alpha=1e-6, fit_intercept=True, solver="auto", max_iter=1000
    ).fit(x, y)

    new_coefs = pd.Series(model.coef_, index=x.columns)

    new_coefs[string_maps.CONSTANT_COL] = model.intercept_
    return new_coefs
    # print(type(coefficients))


class NonzeroRegression(BaseTransformer):
    """
    Transformer for approximating metrics that consist of non-zero contributions
    from other metrics, such as weights or discounts.
    """

    params: pd.DataFrame | None = None

    def __init__(
        self,
        id_col: str = string_maps.ID_COL,
        date_col: str = string_maps.DATE_COL,
        unit_conversion_strategy: BaseUnitConversionStrategy | None = None,
        force_refit: bool = False,
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

        self._prefit = False
        self.force_refit = force_refit

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the transformer to the data.
        """
        super().fit(X, y)

        # Filter to only nonzero values for the fit
        null_labels = y.copy()
        if (null_labels < 0).any():
            raise ValueError(
                "The target series must not contain negative values for "
                "NonzeroRegression."
            )
        null_labels[null_labels == 0] = np.nan

        # We want all months with at least 2 observations; lets count them
        periods = y.index.to_frame()[self.date_col].unique()

        cols = X.columns
        # Start with zero - this is a manual Ffill
        coefs = pd.Series([0.0] * len(cols), index=cols)
        params = []
        for per in sorted(periods):
            m_y: pd.Series = null_labels.loc[per].copy()  # type: ignore
            m_x: pd.DataFrame = X.loc[per][m_y.notnull()].copy()  # type: ignore
            m_y = m_y[m_y.notnull()]

            coefs = _get_monthly_coefs(coefs, m_y, m_x)
            params.append({**coefs.to_dict(), self.date_col: per})

        self._params = pd.DataFrame(params)
        self._prefit = True
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series):
        """
        Transform the data by decomposing metrics.
        """
        if not self._prefit:
            raise TransformNotFittedError(
                "Transform params must be fit before transforming"
            )
        if self._params is None:
            raise TransformNotFittedError(
                "Transform params must be fit before transforming"
            )
        transformed = X.copy()
        transformed[string_maps.CONSTANT_COL] = 1.0

        # Wipe the zeroes
        transformed[y <= 0] = np.nan

        weights = self._params.set_index(self.date_col).reindex(
            transformed.index.get_level_values(self.date_col)
        )[transformed.columns]

        if weights.isnull().values.any():
            raise TransformNotFittedError(
                "Transform params are not available for all dates in the data"
            )
        transformed = (transformed * weights.values).fillna(0.0)
        transformed[string_maps.RESIDUAL_COL] = y - transformed.sum(axis=1)
        return transformed

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        if self.force_refit or not self._prefit:
            self.fit(X, y)
        return self.transform(X, y)

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
