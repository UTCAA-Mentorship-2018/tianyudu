"""
Data processing methods as placed in this script.
"""

import numpy as np
import pandas as pd
import keras
from typing import Dict, List
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# Pre-defined constants
# For quick debuging.
FILE_DIR = "/Users/tianyudu/Documents/Activities/UTCAA-Mentorship-2018/data/application_train.csv"
DROP_THRESHOLD = 0.1
DROP_COLUMNS = []


def load_data(
    file_dir: str,
    drop_columns: List[str],
    drop_threshold: float
) -> pd.DataFrame:
    """
    This method loads data from csv to a pandas data frame.

    Args:
        file_dir:
            the directory of target csv file.
        drop_columns: 
            A list of string representing column names (feature). Column names in this list would 
            be dropped from the raw data.
        drop_threshold:
            (between 0 and 1)
            If the percentage of Nan (missing/unverfied) values of one certain feature is more than the 
            threshold value, this feature will be dropped from the raw data.
            !Note! this is a very general dropping rule. In later models, I would specify a 
            list of columns in DROP_COLUMNS and turn this off by setting its value to 1.

    Returns:
        A pre-processed data frame containing samples.
    """

    # Load raw data from csv.
    print("Loading dataset from local file...")
    df = pd.read_csv(file_dir, sep=",", header="infer")
    print(f"Raw data shape: {df.shape}")

    # Drop columns with too many Nan values.
    df = drop_by_na_percent(df, drop_threshold)

    # Drop specific columns.
    if drop_columns != []:
        df.drop(columns=[drop_columns], inplace=True)
        print(f"Data shape after dropping specified columns: {df.shape}")

    # Drop observations with Nan attributes.
    df = drop_na_obs(df)

    assert "TARGET" in df.columns, "Oops, target not found in dataset."
    return df


def drop_by_na_percent(
    raw: pd.DataFrame,
    threshold: float
) -> pd.DataFrame:
    assert 0 <= threshold <= 1, "threshold should be between 0 and 1."
    df = raw.copy(deep=True)

    for col in df.columns:
        nan_array = pd.isna(df[col])
        nan_array = nan_array.astype(np.int).values
        nan_ratio = sum(nan_array) / len(nan_array)
        if nan_ratio > threshold:
            df.drop(columns=[col], inplace=True)
    print(
        f"Data shape after column drop(threshold: {threshold}): {df.shape}\n\
        num of features left {df.shape[1]}")
    return df


def drop_na_obs(
    raw: pd.DataFrame
) -> pd.DataFrame:
    df = raw.copy(deep=True)

    original_num_obs = len(df)
    df.dropna(inplace=True)
    num_obs_lost = original_num_obs - len(df)

    print(
        f"Observation lost after ignoring obs w/ nan attributes: {num_obs_lost / original_num_obs * 100: .3f} %")
    print(f"Data shape after ignoring obs w/ nan attributes: {df.shape}")

    return df


def split_data(
    df: pd.DataFrame,
    taget_col: str="TARGET",
    ratio: dict={"train": 0.6, "test": 0.2, "validation": 0.2},
    shuffle=True
) -> Dict[str, pd.DataFrame]:
    """
    Spliting the entire dataset into training, testing and validation sets by given ratios.

    Args:
        df: 
            the dataset (including both X and y, y is labelled as "TARGET")
        target_col:
            column name of target/label in the dataset.
        ratio: 
            the ratio to split dataset into training, testing and validation sets.
        shuffle: 
            if shuffle the dataset before spliting.

    Returns:
        A dictionary of DataFrames with keys
        X_train, y_train, X_test, y_test, X_val, y_val
        and values are the corresponding DataFrame.
    """
    assert np.sum(list(ratio.values())
                  ) == 1, "Spliting ratios should sum up to 1"
    if shuffle:
        df = df.sample(frac=1)

    y = df[taget_col]
    X = df.drop(columns=[taget_col])
    print(f"Raw dataset shape: X={X.shape}, y={y.shape}")
    assert len(X) == len(y)

    num_obs = len(X)
    num_train = int(num_obs * ratio["train"])
    num_test = int(num_obs * ratio["test"])
    num_val = int(num_obs * ratio["validation"])

    X_train = X[:num_train]
    y_train = y[:num_train]

    X_test = X[num_train: num_train + num_test]
    y_test = y[num_train: num_train + num_test]

    X_val = X[num_train + num_test:]
    y_val = y[num_train + num_test:]

    assert len(X_train) + len(X_test) + len(X_val) == num_obs
    assert len(y_train) + len(y_test) + len(y_val) == num_obs

    print(f"""Train/Test/Validation Spliting Summary
    Shuffle: {shuffle}
    Training set:
        X_train: {X_train.shape}
        y_train: {y_train.shape}
    Testing set:
        X_test: {X_test.shape}
        y_test: {y_test.shape}
    Validation set:
        X_val: {X_val.shape}
        y_val: {y_val.shape}
    """)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "X_val": X_val, "y_val": y_val
    }


def int_encode_data(
    src: pd.DataFrame
):
    """
    Integer Encoding.
    Encode categorical data in the given data frame. 
    Note that in for the application_train.csv file, categorical features
    would be loaded as "object" type while other features would be in
    int64 or float 64.
    Args:
        src:
            the source data frame to be encoded.

    Returns:
        df:
            the dataframe with categorical features encoded.
        encoders:
            A dictionary with column names as keys and encoders as values
            Only columns processed in this method would be included in
    """
    df = src.copy()
    print(f"Types in dataframe received:\
    {set([str(df[col].dtypes) for col in df.columns])}")

    total = sum(np.array([str(df[col].dtypes)
                          for col in df.columns]) == "object")
    print(f"Feature with dtype: object ({total}) will be encoded.")

    encoders = dict()
    for col in df.columns:
        if str(df[col].dtypes) == "object":
            print(f"\tEncoding {col}")
            target = df[col]
            # Fit and transform.
            encoder = LabelEncoder()
            encoder.fit(target)
            encoded_target = encoder.transform(target)
            # Apply the change to data frame and save the encoder.
            df[col] = encoded_target
            encoders[col] = encoder

    return df, encoders


def standardize_data(
    splited: Dict[str, pd.DataFrame]
) -> (dict, StandardScaler, StandardScaler):
    """
    Standardize dataset.

    Args:

    Returns:

    """
    scaled_set = dict()

    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaler.fit(splited["X_train"].values)
    y_scaler.fit(splited["y_train"].values.reshape(-1, 1))

    scaled_set["X_train"] = X_scaler.transform(splited["X_train"].values)
    scaled_set["X_test"] = X_scaler.transform(splited["X_test"].values)
    scaled_set["X_val"] = X_scaler.transform(splited["X_val"].values)

    scaled_set["y_train"] = y_scaler.transform(
        splited["y_train"].values.reshape(-1, 1))
    scaled_set["y_test"] = y_scaler.transform(
        splited["y_test"].values.reshape(-1, 1))
    scaled_set["y_val"] = y_scaler.transform(
        splited["y_val"].values.reshape(-1, 1))

    return scaled_set, X_scaler, y_scaler
