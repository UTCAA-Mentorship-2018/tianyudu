"""
debug file.
"""
from data_proc import *
from baseline_nn import BaselineNN
import matplotlib
import matplotlib.pyplot as plt

df = load_data(
    file_dir=FILE_DIR,
    drop_threshold=DROP_THRESHOLD,
    drop_columns=DROP_COLUMNS)

values = df.values

e, encoders = int_encode_data(df)

num_fea = df.shape[1] - 1

splited = split_data(e)
scaled_splited, X_scaler, y_scaler = standardize_data(splited)

for item in splited.keys():
    exec(f"{item} = scaled_splited['{item}']")
    exec(f"print({item}.shape)")

model = BaselineNN(input_dim=num_fea)

keras.utils.print_summary(model.model)

# Pass the scaled dataset into the model.
model.fit(
    X_train,
    y_scaler.inverse_transform(y_train),
    X_val,
    y_scaler.inverse_transform(y_val),
    epochs=25
)

pred = model.model.predict(
    x=X_test,
    verbose=1
)

actual = y_scaler.inverse_transform(y_test)