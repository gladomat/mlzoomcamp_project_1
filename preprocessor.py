import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction import DictVectorizer
from sklearn_features.transformers import DataFrameSelector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import SGDRegressor

import mlflow
from hyperopt import hp, space_eval, fmin, tpe, STATUS_OK, Trials
from hyperopt.pyll import scope
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient


def dump_pickle(obj, filename):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


def preprocess(df: pd.DataFrame, model_path: str=None):
    """Preprocess the dataframe.
    
    :param df: The dataframe to preprocess.
    :return: The preprocessed dataframe and the pipeline.
    """

    # Compute volume of diamonds.
    df['volume'] = df['x'] * df['y'] * df['z']


    # Drop unnecessary columns
    df.drop(['x', 'y', 'z', 'depth', 'table', 'Unnamed: 0'], axis=1, inplace=True)

    x = df.drop('price', axis=1)
    y = df['price']

    return x, y

def load_data(data_path: str):
    """Load the data from the data path.
    
    :param data_path: The path to the data.
    :return: The dataframe.
    """

    df = pd.read_csv(data_path)
    return df

def data_splitter(X, y, test_size: float=0.2, dest_path: str=None):
    """
    Split the data into train and test.
    :param X: The data.
    :param y: The target.
    :param df: The dataframe to split.
    :param test_size: The size of the test set.
    """
    os.makedirs(dest_path, exist_ok=True)

    # Create train, validation, and test data.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42)

    # Make dataframes from splits and add column names.
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_val = pd.DataFrame(X_val, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)

    # Save datasets.
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "X_val.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "X_test.pkl"))


def preprocessor(raw_data_path: str='./data/Diamonds_Prices2022.csv', 
        dest_path: str='./processed_data', model_path: str='./models'):
    """
    Main function.
    :param raw_data_path: Path to raw data.
    :param dest_path: Path to store preprocessed data.
    """
    df = load_data(raw_data_path)
    X, y = preprocess(df, model_path)
    data_splitter(X, y, dest_path=dest_path)


if __name__ == "__main__":
    preprocessor()