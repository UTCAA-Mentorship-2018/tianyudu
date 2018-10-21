"""
Data processing methods
"""
import numpy as np
import pandas as pd
import keras
from typing import Dict


# Pre-defined constants
FILE_DIR = "/Users/tianyudu/Documents/Activities/UTCAA-Mentorship-2018/data/application_train.csv"
DROP_THRESHOLD = 0.1
DROP_COLUMNS = []

def load_data(
    file_dir: str,
    drop_columns: list,
    drop_threshold: float
) -> pd.DataFrame:
    """
    file_dir:
        the directory of application_train.csv
    drop_columns: 
        the columns would be dropped from the raw data
    drop_threshold:
        the percentage threshold of Nan value of dropping the column.
        Note this is a very general dropping rule. In later models, I would specify a 
        list of columns in DROP_COLUMNS.
    """
    assert 0 <= drop_threshold <= 1, "drop_threshold should be between 0 and 1."

    # Load raw data from csv.
    print("Loading dataset from local file...")
    df = pd.read_csv(file_dir, sep=",", header="infer")
    print(f"Raw data shape: {df.shape}")

    # Drop columns with too many Nan values.
    for col in df.columns:
        nan_array = pd.isna(df[col])
        nan_array = nan_array.astype(np.int).values
        nan_ratio = sum(nan_array) / len(nan_array)
        if nan_ratio > drop_threshold:
            df.drop(columns=[col], inplace=True)
    print(
        f"Data shape after column drop(threshold: {drop_threshold}): {df.shape}")
    
    # Drop specific columns.
    if drop_columns != []:
        df.drop(columns=[drop_columns], inplace=True)
        print(f"Data shape after dropping specified columns: {df.shape}")

    # Drop observations with Nan attributes.
    raw_num_obs = len(df)
    df.dropna(inplace=True)
    num_obs_lost = raw_num_obs - len(df)
    print(f"Observation lost after ignoring obs w/ nan attributes: {num_obs_lost / raw_num_obs * 100: .3f} %")
    print(f"Data shape after ignoring obs w/ nan attributes: {df.shape}")

    assert "TARGET" in df.columns, "Oops, target not found in dataset."
    return df


def split_data(
    df: pd.DataFrame,
    ratio: dict={"train": 0.6, "test": 0.2, "validation": 0.2},
    shuffle=True
) -> Dict[str, pd.DataFrame]:
    """
    Spliting the entire dataset into training, testing and validation sets by given ratios.

    df: the dataset (including both X and y, y is labelled as "TARGET")
    ratio: the ratio to split dataset into training, testing and validation sets.
    shuffle: if shuffle the dataset before spliting.
    

    """
    assert np.sum(list(ratio.values())) == 1, "Spliting ratios should sum up to 1"
    if shuffle:
        df = df.sample(frac=1)
    
    y = df["TARGET"]
    X = df.drop(columns=["TARGET"])
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

