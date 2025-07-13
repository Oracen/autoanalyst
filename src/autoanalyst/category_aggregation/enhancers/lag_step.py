import pandas as pd

from autoanalyst.core import string_maps
from autoanalyst.core.base_classes import BaseAggregatorEnhancer


class LagStepEnhancer(BaseAggregatorEnhancer):
    """
    Enhancer for aggregators that adds a lag step to the aggregation process.
    This can be used to create a lag-lead feature for signature kernels.
    """

    def __init__(
        self,
        n_steps: int = 1,
        suffix: str = string_maps.LEADLAG_SUFFIX,
        id_col: str = string_maps.ID_COL,
        date_col: str = string_maps.DATE_COL,
    ):
        super().__init__(id_col, date_col)
        self.n_steps = n_steps
        self.suffix = suffix

    def enhance(self, X: pd.DataFrame, categories: pd.DataFrame) -> pd.DataFrame:
        return (
            X.sort_index()
            .groupby(self.id_col)
            .shift(self.n_steps)
            .add_suffix(self.suffix)
            .fillna(0)
        )
