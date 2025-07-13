import pandas as pd

from autoanalyst.core.base_classes import BaseStorageModule


class DataFrameStorageModule(BaseStorageModule):
    """
    A storage module for handling DataFrame entities.
    This class is used to encapsulate the logic for loading and managing DataFrame entities.
    """

    def __init__(self):
        """
        Initialize the DataFrame storage module with an option to save intermediate results.
        """
        super().__init__()
        self.processed_datasets: dict[str, pd.DataFrame] = {}
        self.processed_categories: dict[str, pd.DataFrame] = {}

    def save_dataset(self, name: str, entity: pd.DataFrame):
        self.processed_datasets[name] = entity

    def load_dataset(self, name: str) -> pd.DataFrame:
        return self.processed_datasets[name].copy()

    def save_categories(self, name: str, categories: pd.DataFrame):
        self.processed_categories[name] = categories

    def load_categories(self, name: str) -> pd.DataFrame:
        return self.processed_categories[name].copy()
