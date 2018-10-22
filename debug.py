"""
debug file.
"""
from data_proc import *
from baseline_nn import BaselineNN

df = load_data(
    file_dir=FILE_DIR,
    drop_threshold=DROP_THRESHOLD,
    drop_columns=DROP_COLUMNS)

values = df.values

e, encoders = int_encode_data(df)

num_fea = df.shape[1] - 1

splited = split_data(e)
for item in splited.keys():
    exec(f"{item} = splited['{item}']")
    exec(f"print({item}.shape)")

model = BaselineNN(input_dim=num_fea)

keras.utils.print_summary(model.model)

model.fit(
    X_train.values,
    y_train.values,
    X_val.values,
    y_val.values,
    epochs=100
)

pred = model.model.predict(
    x=X_test
)
