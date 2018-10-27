"""
debug file.
"""
from core.data.data_proc import *
from core.models.baseline_nn import BaselineNN
import matplotlib
import matplotlib.pyplot as plt
from core.tools.roc_visualize import visualize_roc
import keras

# ======== CONSTANTS ========
FILE_DIR_1 = "/Users/tianyudu/Documents/Activities/UTCAA-Mentorship-2018/data/application_train.csv"
FILE_DIR_2 = "/Volumes/Intel/Data/UTCAA-Mentorship-2018/application_train.csv"

DROP_THRESHOLD = 0.1
DROP_COLUMNS = []

df = load_data(
    file_dir=FILE_DIR_1,
    drop_threshold=DROP_THRESHOLD,
    drop_columns=DROP_COLUMNS)

values = df.values

e, encoders = int_encode_data(df)

num_fea = df.shape[1] - 1

splited = split_data(e, target_col="TARGET")
scaled_splited, X_scaler, y_scaler = standardize_data(splited)

# ======== Neural Net ========
model = BaselineNN(input_dim=num_fea)

keras.utils.print_summary(model.core)

# Pass the scaled dataset into the model.
model.fit(
    scaled_splited["X_train"],
    splited["y_train"],
    scaled_splited["X_val"],
    splited["y_val"],
    epochs=3
)

pred = model.core.predict(
    x=scaled_splited["X_test"],
    verbose=1
)

visualize_roc(
    actual=splited["y_test"],
    pred_prob=pred,
    save_dir="./sample.html",
    show=True
)
