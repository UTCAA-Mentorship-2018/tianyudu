"""
Data processing methods
"""
import numpy as np
import pandas as pd
import keras


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
    # TODO: add section

    # Drop observations with Nan attributes.
    raw_num_obs = len(df)
    df.dropna(inplace=True)
    num_obs_lost = raw_num_obs - len(df)
    print(f"Observation lost after ignoring obs w/ nan attributes: {num_obs_lost / raw_num_obs * 100: .3f} %")
    print(f"Data shape after ignoring obs w/ nan attributes: {df.shape}")
    return df


