import pandas as pd

from autoanalyst.core import string_maps
from autoanalyst.core.base_classes import BaseDecisionRule


def get_xmr_bounds(data_x: pd.Series, sigma: float, sigma_d2: float) -> pd.DataFrame:
    mean = data_x.mean()
    diff = data_x.diff().astype(float)  # Leaving the first NaN value as is
    sign = diff >= 0.0
    range_x = diff.abs()
    mean_range = range_x.mean()
    upper_limit = mean + sigma * mean_range
    lower_limit = mean - sigma * mean_range

    upper_midpoint = mean + 0.5 * sigma * mean_range
    lower_midpoint = mean - 0.5 * sigma * mean_range

    upper_limit_residual = mean_range * sigma_d2

    return pd.DataFrame(
        {
            "x": data_x,
            "upper_limit": upper_limit,
            "lower_limit": lower_limit,
            "upper_midpoint": upper_midpoint,
            "lower_midpoint": lower_midpoint,
            "mean": mean,
            "residual": range_x,
            "sign": sign,
            "mean_residual": mean_range,
            "upper_limit_residual": upper_limit_residual,
        },
        index=data_x.index,
    )


class XMRRule(BaseDecisionRule):
    """
    XMR decision rule for anomaly detection.
    """

    def __init__(
        self,
        parent_col: str,
        sigma: float = 2.66,  # Taken from Chebychev's inequality
        sigma_d2: float = 3.267,  # Shewhart’s d2 table value for n=2
        id_col: str = string_maps.ID_COL,
        date_col: str = string_maps.DATE_COL,
    ) -> None:
        """
        Initialize the XMR rule with ID and date columns.

        See https://entropicthoughts.com/statistical-process-control-a-practitioners-guide
        for more details on the XMR rule.
        """
        super().__init__(id_col, date_col, parent_col)

        self.sigma = sigma
        self.sigma_d2 = sigma_d2

    def apply(self, X: pd.DataFrame, explained_variance: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the XMR decision rule to the data.
        """
        return pd.DataFrame()
