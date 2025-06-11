from typing import List

import pandas as pd

from .string_maps import DATE_COL, ID_COL


def standardise_index(df: pd.DataFrame, date_col: str, id_col: str) -> pd.DataFrame:

    return df.rename_axis(index={date_col: DATE_COL, id_col: ID_COL})


def remap_index(df: pd.DataFrame, date_col: str, id_col: str) -> pd.DataFrame:

    return df.rename_axis(index={DATE_COL: date_col, ID_COL: id_col})


def sequence_index(df: pd.DataFrame, date_col: str, id_col: str) -> pd.DataFrame:
    """
    Set the index of the DataFrame to a multi-index based on date and ID columns.

    Args:
        df (pd.DataFrame): The DataFrame to index.
        date_col (str): The name of the date column.
        id_col (str): The name of the ID column.

    Returns:
        pd.DataFrame: DataFrame with a multi-index set to (date_col, id_col).
    """
    return df.set_index([date_col, id_col]).sort_index()


def prep_columns(
    df: pd.DataFrame, date_col: str, id_col: str, variable_cols: List[str]
) -> pd.DataFrame:
    """
    Prepare the DataFrame by resetting the index and ensuring the correct columns are set.

    Args:
        df (pd.DataFrame): The DataFrame to prepare.
        date_col (str): The name of the date column.
        id_col (str): The name of the ID column.
        funnel_cols (List[str]): List of funnel metric columns.

    Returns:
        pd.DataFrame: Prepared DataFrame with specified columns.
    """
    all_cols = [date_col, id_col] + variable_cols

    # Quick preflight check
    df_cols = set(df.columns)
    msg = f"Required columns {all_cols} not found, got {df_cols}"
    assert set(all_cols).issubset(df_cols), msg

    # Build the raw components of the decomposition equation from lagged values (i.e.
    # the metric_{t-1} component) and the diffs (i.e. the del metric_t component)
    sequenced = sequence_index(df[all_cols], date_col, id_col)
    return standardise_index(sequenced, date_col, id_col)
