import pandas as pd
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

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

def sample(X: np.ndarray, y: np.ndarray, n: int) -> Tuple[np.ndarray,
                                                          np.ndarray]:
    """
    This method is used to sample a random number of N rows betwen [0, X.shape[0]]
    :param X: the X array to sample from
    :param y: the y array to sample from
    :param n: the number of samples
    :return: the tuple containing a subset of samples in X and y
    """
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

def get_class_weights(neg: int, pos: int) -> dict:
    total = neg + pos
    weight_for_0 = (1 / neg)*(total)/2.0
    weight_for_1 = (1 / pos)*(total)/2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}
    return class_weight

def prepare_network_data(df: pd.DataFrame,
                         target_name: str,
                         n_numerical_cols: int,
                         train_ratio: float) -> Tuple[np.ndarray, np.ndarray,
                                                      np.ndarray, np.ndarray,
                                                      List[LabelEncoder]]:
    # Get X and y
    X, y = get_X_y(df, target_name)
    X, labels = encode_vector_label(X, n_numerical_cols)
    y = np.array(y)
    
    # Use a utility from sklearn to split and shuffle our dataset
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
                                                          train_size=train_ratio,
                                                          random_state=0)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train[:,:n_numerical_cols])
    X_valid_sc = scaler.transform(X_valid[:,:n_numerical_cols])
    X_train = np.concatenate([X_train_sc, X_train[:,n_numerical_cols:]], axis=1)
    X_valid = np.concatenate([X_valid_sc, X_valid[:,n_numerical_cols:]], axis=1)

    return X_train, X_valid, y_train, y_valid, labels, scaler

def encode_vector_label(data: List[np.ndarray],
                        n_numerical_cols: int) -> Tuple[List[np.ndarray],
                                                        List[LabelEncoder]]:
    #TODO: Handle the case where cat columns are before num columns
    labels_encoded = []
    data_encoded = np.array(data)
    for i in range(n_numerical_cols, data_encoded.shape[1]):
        le = preprocessing.LabelEncoder()
        le.fit(data_encoded[:, i])
        labels_encoded.append(le)
        data_encoded[:, i] = le.transform(data_encoded[:, i])

    data_encoded = data_encoded.astype(float) # TODO: Determine the right type
    return data_encoded, labels_encoded