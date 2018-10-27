"""
GBM classifier.
"""
from data_proc import *
from baseline_nn import BaselineNN
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import GradientBoostingClassifier

FILE_DIR = "/Volumes/Intel/Data/UTCAA-Mentorship-2018/application_train.csv"
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


classifier = GradientBoostingClassifier(
    verbose=1
)

classifier.fit(
    X=scaled_splited["X_train"],
    y=splited["y_train"]
)

pred_test = classifier.predict(scaled_splited["X_test"])
pred_prob = classifier.predict_proba(scaled_splited["X_test"])