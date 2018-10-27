"""
Methods in this file visualize the metrics to evaluate model
using AUC of ROC.
"""
from data_proc import *
import numpy as np
import sklearn
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt
import bokeh

fpr, tpr, thresholds = metrics.roc_curve()

def visualize_roc(
    actual: np.ndarray,
    pred_prob: np.ndarray,
    save_dir: str=None
) -> None:
    fpr, tpr, thresholds = metrics.roc_curve(
        y_true=actual,
        y_score=pred_prob
    )

    roc_auc = metrics.auc(fpr, tpr)

    plot = bokeh.plotting.figure(
        title="Receiver operating characteristic",
        x_axis_label="False Positive Rate",
        y_axis_label="True Positive Rate",
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0)
    )

    plot.line(
        fpr,tpr,
        color="red",
        alpha=0.7,
        legend=f"ROC Curve (AUC={roc_auc: 0.2f})"
    )

    plt.plot(
        [0, 1], [0, 1],
        color="navy", alpha=0.7
    )

    bokeh.io.show(figure)

