"""
GBM classifier.
"""
import os
import lightgbm as lgb
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

X_train, X_test, y_train, y_test = train_test_split(e, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)

print(f"Training set: {X_train.shape}, {y_train.shape}\
\nTesting set: {X_test.shape}, {y_test.shape}\
\nValidation set: {X_val.shape}, {y_val.shape}"
)

scaler = StandardScaler().fit(X_train)

scaled_X_train = scaler.transform(X_train)
scaled_X_val = scaler.transform(X_val)
scaled_X_test = scaler.transform(X_test)

# ======== GBM Setup ========

train_data = lgb.Dataset(
    scaled_X_train,
    label=y_train,
    feature_name=list(X.columns.astype(str))
)

validation_data = lgb.Dataset(
    scaled_X_val,
    label=y_val,
    reference=train_data,
    feature_name=list(X.columns.astype(str))
)

params = {
    "learning_rate": 0.03,
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "sub_feature": 0.5,
    "num_leaves": 64,
    "min_data": 50,
    "max_depth": 25,
    "max_bin": 512
}

evals_result = dict()

nbr = int(input("Number of boosting rounds >>> "))
classifier = lgb.train(
    train_set=train_data,
    params=params,
    num_boost_round=nbr,
    valid_sets=[train_data, validation_data],
    evals_result=evals_result,
    verbose_eval=10
)

y_pred = classifier.predict(scaled_X_test)

fea_imp = np.stack(
    [classifier.feature_name(), classifier.feature_importance()], axis=1)

srt_fea_imp = np.array(
    sorted([x for x in fea_imp], key=lambda x: -float(x[1]))
)

# ======== SAVE MODEL ========
record_name = input("Record Name >>> ")
model_dir = f"./saved_models/{record_name}"
os.system(f"mkdir {model_dir}")

print("Saving Feature Importances...")
np.savetxt(f"{model_dir}/importance.csv", srt_fea_imp, fmt="%s,%s")

print("Saving ROC plot...")
matplotlib_roc(
    actual=y_test,
    pred_prob=y_pred,
    show=False,
    file_dir=f"{model_dir}/roc.svg"
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
