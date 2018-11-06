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

X_scaler = StandardScaler()
X_scaler.fit(X_train)

# X_train, X_val, y_train, y_val = train_test_split(
#     X_train,
#     y_train,
#     test_size=0.25,
#     shuffle=True
# )

# print(f"Training set: {X_train.shape}, {y_train.shape}\
# \nTesting set: {X_test.shape}, {y_test.shape}\
# \nValidation set: {X_val.shape}, {y_val.shape}")

print(f"Training set: {X_train.shape}, {y_train.shape}\
\nTesting set: {X_test.shape}, {y_test.shape}")

# ======== GBM Setup ========
ROUNDS = int(input("Number of boosting rounds >>> "))

clf = GradientBoostingClassifier(
    n_estimators=100,
    verbose=1
)

clf.fit(
    X_train,
    y_train.values
)

y_pred = clf.predict_proba(
    X_test
)

# fea_imp = np.stack(
#     [classifier.feature_name(), classifier.feature_importance()], axis=1)

# srt_fea_imp = np.array(
#     sorted([x for x in fea_imp], key=lambda x: -float(x[1]))
# )

# ======== SAVE MODEL ========
record_name = input("Record Name >>> ")
model_dir = f"./saved_models/{record_name}"
os.system(f"mkdir {model_dir}")

print("Saving Feature Importances...")
np.savetxt(f"{model_dir}/importance.csv", srt_fea_imp, fmt="%s,%s")

print("Saving ROC plot...")
matplotlib_roc(
    actual=y_test,
    pred_prob=y_pred[:, 1],
    show=True,
    file_dir=None
)

print("Saving AUC training history...")
lgb.plot_metric(booster=evals_result, metric="auc")
plt.savefig(f"{model_dir}/auc_history.svg")
plt.close()

print("Saving loss history...")
lgb.plot_metric(booster=evals_result, metric="binary_logloss")
plt.savefig(f"{model_dir}/loss_history.svg")
plt.close()

print("Saving importance plot...")
lgb.plot_importance(classifier)
plt.savefig(f"{model_dir}/importance.svg")
plt.close()

print("Saving model...")
classifier.save_model(f"{model_dir}/bgm.txt")

# lgb.Booster(model_file='model.txt')
