"""
debug file.
"""
from data_proc import *
from baseline_nn import BaselineNN
import matplotlib
import matplotlib.pyplot as plt


# ======== CONSTANTS ========
FILE_DIR = "/Users/tianyudu/Documents/Activities/UTCAA-Mentorship-2018/data/application_train.csv"
DROP_THRESHOLD = 0.1
DROP_COLUMNS = []

df = load_data(
    file_dir=FILE_DIR,
    drop_threshold=DROP_THRESHOLD,
    drop_columns=DROP_COLUMNS)

values = df.values

e, encoders = int_encode_data(df)

num_fea = df.shape[1] - 1

splited = split_data(e, target_col="TARGET")
scaled_splited, X_scaler, y_scaler = standardize_data(splited)

for item in splited.keys():
    exec(f"{item} = scaled_splited['{item}']")
    exec(f"print({item}.shape)")

# ======== Neural Net ========
model = BaselineNN(input_dim=num_fea)

keras.utils.print_summary(model.model)

# Pass the scaled dataset into the model.
model.fit(
    scaled_splited["X_train"],
    splited["y_train"],
    scaled_splited["X_val"],
    splited["y_val"],
    epochs=100
)

pred = model.core.predict(
    x=scaled_splited["X_test"],
    verbose=1
)

actual = y_scaler.inverse_transform(y_test)
