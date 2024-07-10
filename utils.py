import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import warnings


def delete_from_substring(string):
    index = string.find('(')
    if index != -1:
        return string[:index]
    return string


def split_df(data: pd.DataFrame, n: int, n_col: int):
    dataframes = {}
    for i in range(n):
        col_first = i * n_col
        col_last = col_first + n_col

        df = data.iloc[:, col_first:col_last]
        dataframes[f'X_{i+1}'] = df

    return dataframes


def columns_df(df: pd.DataFrame):
    int_columns = list(df.select_dtypes(include='int').columns)
    float_columns = list(df.select_dtypes(include='float').columns)
    object_columns = list(df.select_dtypes(include='object').columns)
    numerical_columns = int_columns + float_columns

    print('Shape:', df.shape)
    print('Numerical features: ', len(numerical_columns))
    print('Categorical features: ', len(object_columns))

    return int_columns, float_columns, object_columns, numerical_columns


def most_frequent(row):
    values, counts = np.unique(row, return_counts=True)
    max_count_index = np.argmax(counts)
    return values[max_count_index]


warnings.filterwarnings("ignore", message="X does not have valid feature names, but NearestNeighbors was fitted with "
                                          "feature names")


def relief(X, y, n_neighbors=10):
    """
    Relief algorithm for feature ranking.

    Parameters:
    - X: Feature matrix (pandas DataFrame of shape (n_samples, n_features))
    - y: Target variable (pandas Series of shape (n_samples,))
    - n_neighbors: Number of neighbors to consider (default is 10)

    Returns:
    - feature_weights: Pandas Series of feature weights
    """

    n_samples, n_features = X.shape
    feature_weights = np.zeros(n_features)

    # Initialize NearestNeighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)  # +1 because the point itself is included
    nn.fit(X)

    for i in range(n_samples):
        distances, indices = nn.kneighbors(X.iloc[i].values.reshape(1, -1))

        hit_indices = indices[0][y.iloc[indices[0]] == y.iloc[i]][1:]  # Exclude the instance itself
        miss_indices = indices[0][y.iloc[indices[0]] != y.iloc[i]][:n_neighbors]

        for feature in range(n_features):
            # Calculate differences for hits and misses
            hit_diff = np.sum((X.iloc[i, feature] - X.iloc[hit_indices, feature]) ** 2)
            miss_diff = np.sum((X.iloc[i, feature] - X.iloc[miss_indices, feature]) ** 2)

            # Update feature weights
            feature_weights[feature] -= hit_diff / n_neighbors
            feature_weights[feature] += miss_diff / n_neighbors

    # Normalize feature weights
    feature_weights = (feature_weights - np.min(feature_weights)) / (np.max(feature_weights) - np.min(feature_weights))

    return pd.Series(feature_weights, index=X.columns).to_frame(name='Weights').reset_index()
