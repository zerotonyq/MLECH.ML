from typing import List, Dict
import pandas as pd


class DataTransform:
    """
    A class for performing grouping and aggregation on a DataFrame using specified columns and aggregation functions.

    Attributes:
    ----------
    group_columns : List[str]
        A list of column names to group by.
    aggregate_functions : Dict[str, str]
        A dictionary where keys are column names to be aggregated, and values are aggregation functions
        (e.g., 'mean', 'sum').

    Methods:
    -------
    transform(df: pd.DataFrame, used_columns: List[str]=None) -> pd.DataFrame
        Transforms the provided DataFrame by grouping and aggregating based on specified columns and functions.
    """

    def __init__(self, df: pd.DataFrame, group_cols: List[str]=None, aggregate_functions: Dict[str, List[str]]=None):
        """
        Initializes the DataTransform object with specified grouping columns and aggregation functions.

        Parameters:
        ----------
        df : pd.DataFrame
            The DataFrame to be transformed.
        group_cols : List[str], optional
            A list of column names to group by (default is an empty list).
        aggregate_functions : Dict[str, str], optional
            A dictionary of aggregation functions where the key is the column name, and the value is the aggregation
            function (default is an empty dictionary).
        """

        if group_cols is None:
            group_cols = []

        if aggregate_functions is None:
            aggregate_functions = {}

        self.group_columns = group_cols
        self.aggregate_functions = aggregate_functions
        self.df = df

    def transform(self, used_columns: List[str]=None) -> pd.DataFrame:
        """
        Performs grouping and aggregation based on specified columns and functions.

        Parameters:
        ----------
        used_columns : List[str], optional
            A list of columns to use for grouping and aggregation (default is all columns in the DataFrame `df`).

        Returns:
        ----------
        pd.DataFrame
            A new DataFrame with the results of grouping and aggregation. Column names will be combined
            with the aggregation function used.

        Example:
        -------
        # >>> import pandas as pd
        # >>> df = pd.DataFrame({
        # ...     'category': ['A', 'A', 'B', 'B'],
        # ...     'value1': [10, 20, 30, 40],
        # ...     'value2': [100, 200, 300, 400]
        # ... })
        # >>> transform = DataTransform(df, group_cols=['category'], aggregate_functions={'value1': 'sum', 'value2': 'mean'})
        # >>> result = transform.transform()
        # >>> print(result)

        Output:
            category  value1_sum  value2_mean
            A         30         150.0
            B         70         350.0
        """

        if used_columns is None:
            used_columns = self.df.columns.tolist()

        grouped_df = self.df[used_columns].groupby(self.group_columns).agg(self.aggregate_functions)
        grouped_df.columns = ['_'.join(col).strip() for col in grouped_df.columns]
        return grouped_df
