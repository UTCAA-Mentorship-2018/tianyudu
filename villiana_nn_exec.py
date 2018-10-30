"""
debug file.
"""
from core.data.data_proc import *
from core.models.baseline_nn import BaselineNN
from core.tools.roc_visualize import visualize_roc
import keras
from constants import *
from pprint import pprint
from ui_control import *

# ======== CONSTANTS ========

DROP_THRESHOLD = 0.1
DROP_COLUMNS = []

EXPERIMENT_NAME = input("experiment name >>> ")

FILE_DIR = choose_dataset(SAVED_FILE_DIRS)

EPOCHS = int(input("Model training epochs >>> "))

# ======== END ========

df = load_data(
    file_dir=FILE_DIR,
    drop_threshold=DROP_THRESHOLD,
    drop_columns=DROP_COLUMNS)

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
    epochs=EPOCHS
)

pred = model.core.predict(
    x=scaled_splited["X_test"],
    verbose=1
)

model.save_model(file_dir=EXPERIMENT_NAME)

visualize_roc(
    actual=splited["y_test"],
    pred_prob=pred,
    save_dir=f"./saved_models/{EXPERIMENT_NAME}/roc.html",
    show=False
)
