from autoanalyst.core import string_maps
from autoanalyst.core.base_classes import BaseTransformer


class SubtotalTransform(BaseTransformer):
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
        transformed = X.copy()

        transformed[string_maps.RESIDUAL_COL] = y - X.sum(axis=1)
        return transformed
