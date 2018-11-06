"""
GBM classifier.
"""
import os
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from constants import *
from core.data.data_proc import *
from core.tools.roc_visualize import *
from ui_control import *


DROP_THRESHOLD = 0.1
DROP_COLUMNS = []

FILE_DIR = choose_dataset(SAVED_FILE_DIRS)

df = load_data(
    file_dir=FILE_DIR,
    drop_threshold=DROP_THRESHOLD,
    drop_columns=DROP_COLUMNS)

df.drop(columns=["SK_ID_CURR"], inplace=True)

X, y = df.drop(columns=["TARGET"]), df["TARGET"]

e, encoders = int_encode_data(X)

num_fea = df.shape[1] - 1

X_train, X_test, y_train, y_test = train_test_split(
    e,
    y,
    test_size=0.2,
    shuffle=False
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.25,
    shuffle=True
)

X_scaler = StandardScaler()
X_scaler.fit(X_train)

print(f"Training set: {X_train.shape}, {y_train.shape}\
\nTesting set: {X_test.shape}, {y_test.shape}\
\nValidation set: {X_val.shape}, {y_val.shape}")

# ======== GBM Setup ========
# ROUNDS = int(input("Number of boosting rounds >>> "))

clf = GradientBoostingClassifier(
    n_estimators=300,
    verbose=1,
    validation_fraction=0.1,
    n_iter_no_change=10
)

clf.fit(
    X_scaler.transform(X_train),
    y_train
)

y_pred = clf.predict_proba(
    X_scaler.transform(X_test)
)

# ======== Present Result ========

matplotlib_roc(
    actual=y_test,
    pred_prob=y_pred[:, 1],
    show=True,
    file_dir=None
)

