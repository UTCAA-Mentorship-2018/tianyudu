"""
GBM classifier.
"""
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

from core.data.data_proc import *
from constants import *
from ui_control import *

import lightgbm as lgb

DROP_THRESHOLD = 0.1
DROP_COLUMNS = []

FILE_DIR = choose_dataset(SAVED_FILE_DIRS)

df = load_data(
    file_dir=FILE_DIR,
    drop_threshold=DROP_THRESHOLD,
    drop_columns=DROP_COLUMNS)

values = df.values

e, encoders = int_encode_data(df)

num_fea = df.shape[1] - 1

splited = split_data(e, target_col="TARGET")
scaled_splited, X_scaler, y_scaler = standardize_data(splited)

d_train = lgb.Dataset(scaled_splited["X_train"], label=splited["y_train"])

params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 32
params['min_data'] = 50
params['max_depth'] = 25

clf = lgb.train(params, d_train, 30)

y_pred = clf.predict(scaled_splited["X_test"])

plot_roc(
    actual=splited["y_test"],
    pred_prob=y_pred,
    save_dir="./test.html",
    show=True
)

fpr, tpr, thresholds = metrics.roc_curve(
    splited["y_test"],
    y_pred
)


roc_auc = metrics.auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(
    fpr, tpr, color="darkorange", lw=lw, label=f"ROC Curve (area = {roc_auc: 0.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
