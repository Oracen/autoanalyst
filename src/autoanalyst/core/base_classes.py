import abc
from ast import Dict, Tuple
from typing import Any

import pandas as pd


class BaseTransformer(abc.ABC):
    """
    Abstract base class for transformers. Assumes index is (id, date)
    """

    def __init__(
        self,
        id_col: str,
        date_col: str,
        unit_conversion_strategy: "BaseUnitConversionStrategy",
    ):
        """
        Initialize the transformer with ID and date columns. Defaults are used within
        the AutoAnalyst framework, but can be overridden if necessary to e.g. link into
        SKLearn.
        """
        self.id_col = id_col
        self.date_col = date_col
        self.unit_conversion = unit_conversion_strategy

    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the transformer to the data.
        """
        assert (X.index == y.index).all()
        return self

    @abc.abstractmethod
    def transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Transform the data.
        """
        pass

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit and transform the data.
        """
        self.fit(X, y)
        return self.transform(X, y)

    @abc.abstractmethod
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
        return X


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

    def __init__(
        self,
        name: str,
        id_col: str,
        date_col: str,
        dim_cols: list[str],
        metric_cols: list[str],
        category_aggregator: "BaseCategoryAggregator",
    ):
        """
        Initialize the entity with a name and ID/date columns.
        """
        self.name = name
        self.id_col = id_col
        self.date_col = date_col
        self.dim_cols = dim_cols
        self.metric_cols = metric_cols
        self.category_aggregator = category_aggregator

        self.valid_columns = self.get_valid_columns()

        self._check_key_cols()

    @abc.abstractmethod
    def get_valid_columns(self) -> list[str]:
        """
        Get the columns of the entity.
        """
        pass

    def _check_key_cols(self):
        missing = self.check_column_membership(
            [self.id_col, self.date_col] + self.dim_cols + self.metric_cols
        )
        if missing:
            raise ValueError(
                f"Entity {self.name} is missing key columns: {missing}. "
                f"Expected ID column: {self.id_col}, Date column: {self.date_col}."
            )

    def check_column_membership(self, candidate_columns: list[str]):
        return [col for col in candidate_columns if col not in self.valid_columns]

    @abc.abstractmethod
    def load_columns(self, columns: list[str]) -> pd.DataFrame:
        """
        Load specific columns from the entity.
        """
        pass

    def load_dims(self) -> pd.DataFrame:
        """
        Load dimension columns from the entity.
        """
        return self.load_columns(self.dim_cols)


class BaseStorageModule(abc.ABC):
    """
    Abstract base class for storage modules.
    """

    def __init__(self):
        """
        Initialize the storage module with an option to save intermediate results.
        """

    @abc.abstractmethod
    def save_dataset(self, name: str, entity: pd.DataFrame):
        """
        Save an entity to the storage module.
        """
        pass

    @abc.abstractmethod
    def load_dataset(self, name: str) -> pd.DataFrame:
        """
        Load an entity from the storage module by name.
        """
        pass

    @abc.abstractmethod
    def load_categories(self, head_col: str, children_cols: list[str]) -> pd.DataFrame:
        """
        Load metrics data for a given entity.
        """
        pass

    @abc.abstractmethod
    def save_categories(self, name: str, categories: pd.DataFrame):
        """
        Save categories to the storage module.
        """
        pass


class BaseUnitConversionStrategy(abc.ABC):
    """
    Abstract base class for unit conversion strategies.
    """

    def __init__(self, id_col: str, date_col: str):
        """
        Initialize the unit conversion strategy with ID and date columns.
        """
        self.id_col = id_col
        self.date_col = date_col

    def __call__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target: pd.Series,
        is_id_compatible: bool = False,
    ) -> pd.DataFrame:
        """
        Convert units of the data.
        """
        if is_id_compatible:
            return self.id_compatible_transform(X, y, target)
        else:
            return self.grain_change_transform(X, y, target)

    @abc.abstractmethod
    def id_compatible_transform(
        self, X: pd.DataFrame, y: pd.Series, target: pd.Series
    ) -> pd.DataFrame:
        """
        Check if the ID columns of X and y are compatible.
        """
        pass

    @abc.abstractmethod
    def grain_change_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target: pd.Series,
    ) -> pd.DataFrame:
        """
        Transform the data to a different grain.
        """
        pass


class BaseAggregatorEnhancer(abc.ABC):
    """
    Abstract base class for aggregator enhancers; feature engineering for clustering
    """

    def __init__(self, id_col: str, date_col: str):
        """
        Initialize the aggregator enhancer with ID and date columns.
        """
        self.id_col = id_col
        self.date_col = date_col

    @abc.abstractmethod
    def enhance(self, X: pd.DataFrame, categories: pd.DataFrame) -> pd.DataFrame:
        """
        Enhance the data for aggregation.
        """
        pass


class BaseCategoryAggregator(abc.ABC):
    """
    Abstract base class for category aggregators; clustering algos for category
    reduction.
    """

    def __init__(
        self, id_col: str, date_col: str, preprocessors: list[BaseAggregatorEnhancer]
    ):
        """
        Initialize the category aggregator with ID and date columns.
        """
        self.id_col = id_col
        self.date_col = date_col
        self.preprocessors = preprocessors

    @abc.abstractmethod
    def aggregate(self, X: pd.DataFrame, categories: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate the data by categories.
        """
        pass

    @abc.abstractmethod
    def aggregate_raw(self, X: pd.DataFrame, categories: pd.DataFrame) -> Any:
        """Allow debugging of algorithm internals"""
        pass

    def preprocess(self, X: pd.DataFrame, categories: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data using the registered preprocessors.
        """

        return pd.concat(
            [X]
            + [
                preprocessor.enhance(X, categories)
                for preprocessor in self.preprocessors
            ],
            axis=1,
        )


class BaseDecisionRule(abc.ABC):
    """
    Abstract base class for decision rules.
    """

    def __init__(self, id_col: str, date_col: str, parent_col: str):
        """
        Initialize the decision rule with ID and date columns.
        """
        self.id_col = id_col
        self.date_col = date_col
        self.parent_col = parent_col

    @abc.abstractmethod
    def evaluate(
        self, X: pd.DataFrame, explained_variance: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Evaluate the decision rule on the data.
        """
        pass
