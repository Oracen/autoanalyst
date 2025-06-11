import pandas as pd

from autoanalyst.core import string_maps
from autoanalyst.core.base_classes import BaseTransformer, BaseUnitConversionStrategy
from autoanalyst.transform.unit_conversion_strategies.sum_of_whole import (
    SumOfWholeStrategy,
)


class SubtotalTransform(BaseTransformer):
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

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the transformer to the data.
        """
        # No fitting required for decomposition
        return super().fit(X, y)

    def transform(self, X: pd.DataFrame, y: pd.Series):
        """
        Transform the data by decomposing metrics.
        """
        transformed = X.copy()

        transformed[string_maps.RESIDUAL_COL] = y - X.sum(axis=1)
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
