"""
debug file.
"""
from data_proc import *

df = load_data(
    file_dir=FILE_DIR,
    drop_threshold=DROP_THRESHOLD,
    drop_columns=DROP_COLUMNS)

values = df.values

e, encoders = int_encode_data(df)

splited = split_data(df)
for item in splited.keys():
    exec(f"{item} = splited['{item}']")