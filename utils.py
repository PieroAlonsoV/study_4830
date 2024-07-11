import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin, clone


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


class CustomAdaBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, base_learners=None, n_estimators=50):
        self.base_learners = base_learners
        self.n_estimators = n_estimators

    def fit(self, X, y):
        self.learners_ = []
        self.alphas_ = []
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape

        # Initialize weights
        sample_weights = np.ones(n_samples) / n_samples

        for i in range(self.n_estimators):
            learner = clone(self.base_learners[i % len(self.base_learners)])

            if hasattr(learner, 'fit') and 'sample_weight' in learner.fit.__code__.co_varnames:
                # If learner supports sample_weight
                learner.fit(X, y, sample_weight=sample_weights)
            else:
                # Resample the dataset based on sample_weights
                sample_indices = np.random.choice(np.arange(n_samples), size=n_samples, p=sample_weights)
                X_resampled = X[sample_indices]
                y_resampled = y[sample_indices]
                learner.fit(X_resampled, y_resampled)

            predictions = learner.predict(X)

            incorrect = (predictions != y)
            error = np.dot(sample_weights, incorrect) / np.sum(sample_weights)

            if error > 0.5:
                continue

            alpha = 0.5 * np.log((1 - error) / error)
            self.learners_.append(learner)
            self.alphas_.append(alpha)

            # Update sample weights
            sample_weights *= np.exp(-alpha * y * predictions)
            sample_weights /= np.sum(sample_weights)

        return self

    def predict(self, X):
        learner_preds = np.array([alpha * learner.predict(X) for alpha, learner in zip(self.alphas_, self.learners_)])
        final_predictions = np.sign(np.sum(learner_preds, axis=0))
        return final_predictions
