import abc
from turtle import st

import pandas as pd

from .string_maps import DATE_COL, ID_COL


class BaseTransformer(abc.ABC):
    """
    Abstract base class for transformers. Assumes index is (id, date)
    """

    def __init__(self, id_col: str, date_col: str):
        """
        Initialize the transformer with ID and date columns. Defaults are used within
        the AutoAnalyst framework, but can be overridden if necessary to e.g. link into
        SKLearn.
        """
        self.id_col = id_col
        self.date_col = date_col

    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the transformer to the data.
        """
        assert X.index == y.index
        return self

    @abc.abstractmethod
    def transform(self, X: pd.DataFrame, y: pd.Series):
        """
        Transform the data.
        """
        pass

    def fit_transform(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit and transform the data.
        """
        self.fit(X, y)
        return self.transform(X, y)


class BaseLoader(abc.ABC):
    """
    Abstract base class for loaders.
    """

    @abc.abstractmethod
    def load_metrics(
        self,
        head_col: str,
        children_cols: list[str],
    ):
        """
        Load metrics data for a given entity.
        """
        pass

    @abc.abstractmethod
    def load_dims(self):
        """
        Load dimensions data for a given entity.
        """
        pass


class BaseEntity(abc.ABC):
    """
    Abstract base class for entities.
    """

    def __init__(self, name: str, id_col: str, date_col: str):
        """
        Initialize the entity with a name and ID/date columns.
        """
        self.name = name
        self.id_col = id_col
        self.date_col = date_col

        self.valid_columns = self.get_valid_columns()

        self._check_key_cols()

    @abc.abstractmethod
    def get_valid_columns(self) -> list[str]:
        """
        Get the columns of the entity.
        """
        pass

    def _check_key_cols(self):
        missing = self.check_column_membership([self.id_col, self.date_col])
        if missing:
            raise ValueError(
                f"Entity {self.name} is missing key columns: {missing}. "
                f"Expected ID column: {self.id_col}, Date column: {self.date_col}."
            )

    def check_column_membership(self, candidate_columns: list[str]):
        return [col for col in candidate_columns if col not in self.valid_columns]
