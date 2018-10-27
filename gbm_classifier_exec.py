"""
GBM classifier.
"""
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

from core.data.data_proc import *
from core.models.baseline_nn import BaselineNN
from constants import *

DROP_THRESHOLD = 0.1
DROP_COLUMNS = []

df = load_data(
    file_dir=FILE_DIR_MAC,
    drop_threshold=DROP_THRESHOLD,
    drop_columns=DROP_COLUMNS)

values = df.values

e, encoders = int_encode_data(df)

num_fea = df.shape[1] - 1

splited = split_data(e, target_col="TARGET")
scaled_splited, X_scaler, y_scaler = standardize_data(splited)

classifier = GradientBoostingClassifier(
    learning_rate=0.3,
    n_estimators=300,
    verbose=1
)

classifier.fit(
    X=scaled_splited["X_train"],
    y=splited["y_train"]
)

pred_prob = classifier.predict_proba(scaled_splited["X_test"])

# ==== Remove ====

fpr, tpr, thresholds = metrics.roc_curve(
    splited["y_test"],
    pred_prob[:, 0]
)

# fpr, tpr, thresholds = metrics.roc_curve(
#     splited["y_test"],
#     pred
# )

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
