import os
import shutil
import pandas as pd
import numpy as np
from typing import Tuple, List
from pandas.api.types import is_numeric_dtype, is_string_dtype
from embedder.network_category import *

TITLE_FORMAT = 'Weights for %s'
SCATTER_EMBEDDINGS_FORMAT = '%s_embedding.%s'
PLOT_LOSS_FORMAT = 'loss_epochs.%s'
    
def get_embedding_size(unique_values: int) -> int:
    """
    Return the embedding size to be used on the Embedding layer
    :param unique_values: the number of unique values in the given category
    :return: the size to be used on the embedding layer
    """
    size = int(min(np.ceil(unique_values / 2), 50))
    if size < 2:
        return 2
    else:
        return size

def get_categorial_cols(df: pd.DataFrame, target_name: str) -> List:
    """
    Returns a list of the categories from a given pandas DataFrame, with the exception of the provided target name
    :param df: the DataFrame
    :param target_name: the name of the target column to not be included
    :return: a List of Category with the df columns except the provided one
    """
    cat_list = []
    for category in df:
        if not category == target_name and is_string_dtype(df[category]): # Only encode object columns
            cat_list.append(NetworkCategory(category, df[category].nunique()))
    return cat_list

def get_numerical_cols(df: pd.DataFrame, target_name:str) -> List:
    """
    Generates a list of numerial categories from a dataframe
    """
    num_list = []
    for category in df:
        if not category == target_name and is_numeric_dtype(df[category]): # Only encode numerical columns
            num_list.append(category)
    return num_list

def check_weights_output(weights_output: str) -> None:
    if not weights_output:
        raise ValueError("You should provide a output file for the embeddings weights")
    
def series_to_list(series: pd.Series) -> List:
    """
    This method is used to convert a given pd.Series object into a list
    :param series: the list to be converted
    :return: the list containing all the elements from the Series object
    """
    list_cols = []
    for item in series:
        list_cols.append(item)

    return list_cols

def sample(X: np.ndarray, y: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    num_row = X.shape[0]
    indices = np.random.randint(num_row, size=n)
    return X[indices, :], y[indices]


def get_X_y(df: pd.DataFrame, name_target: str) -> Tuple[List, List]:
    """
    This method is used to gather the X (features) and y (targets) from a given dataframe based on a given
    target name
    :param df: the dataframe to be used as source
    :param name_target: the name of the target variable
    :return: the list of features and targets
    """
    X_list = []
    y_list = []

    for index, record in df.iterrows():
        fl = series_to_list(record.drop(name_target))
        X_list.append(fl)
        y_list.append(int(record[name_target]))

    return X_list, y_list

def transpose_to_list(X: np.ndarray) -> List[np.ndarray]:
    """
    :param X: the ndarray to be used as source
    :return: a list of nd.array containing the elements from the numpy array
    """
    features_list = []
    for index in range(X.shape[1]):
        features_list.append(X[..., [index]])

    return features_list

def get_all_columns_except(df: pd.DataFrame, columns_to_skip: List[str]) -> pd.DataFrame:
    return df.loc[:, list(filter(lambda x: x not in columns_to_skip, df.columns))]

def create_random_dataframe(rows: int = 4, cols: int = 4,
                            columns: str = 'ABCD') -> pd.DataFrame:
    return pd.DataFrame(np.random.randint(0, 10, size=(rows, cols)), columns=list(columns))

def create_random_csv(path, filename, rows: int = 4, cols: int = 4, columns: str = 'ABCD') -> str:
    df = create_random_dataframe(rows, cols, columns)
    full_path = os.path.join(path, filename)
    os.makedirs(path, exist_ok=True)
    df.to_csv(full_path, index=False)
    return full_path

def remove_random_csv(path) -> None:
    shutil.rmtree(path)