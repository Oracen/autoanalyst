import pandas as pd

from autoanalyst.core import data_manip
from autoanalyst.core.base_classes import BaseEntity


class DataFrameEntity(BaseEntity):
    """
    A class representing a DataFrame entity.
    This class is used to encapsulate a DataFrame and its associated metadata.
    """

    def __init__(self, name: str, id_col: str, date_col: str, df: pd.DataFrame):

        self.df = data_manip.sequence_index(df, date_col, id_col)
        super().__init__(name=name, id_col=id_col, date_col=date_col)

    def __repr__(self):
        return f"DataFrameEntity(name={self.name}, shape={self.df.shape})"

    def get_valid_columns(self) -> list[str]:
        return list(self.df.reset_index().columns)
