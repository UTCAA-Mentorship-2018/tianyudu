"""
Methods in this file visualize the metrics to evaluate model
using AUC of ROC.
"""
import matplotlib
import matplotlib.pyplot as plt
from core.data.data_proc import *
import numpy as np
import sklearn
from sklearn import metrics
import bokeh
import bokeh.plotting


def bokeh_roc(
    actual: np.ndarray,
    pred_prob: np.ndarray,
    save_dir: str="./temp_roc.html",
    show: bool=False
) -> None:
    fpr, tpr, thresholds = metrics.roc_curve(
        y_true=actual,
        y_score=pred_prob
    )

    roc_auc = metrics.auc(fpr, tpr)

    p = bokeh.plotting.figure(
        title="Receiver operating characteristic",
        x_axis_label="False Positive Rate",
        y_axis_label="True Positive Rate",
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0)
    )

    bokeh.io.output_file(
        filename=save_dir,
        title="Receiver operating characteristic"
    )

    p.line(
        fpr, tpr,
        color="red",
        alpha=0.7,
        legend=f"ROC Curve (AUC={roc_auc: 0.2f})"
    )

    p.line(
        [0, 1], [0, 1],
        color="navy", alpha=0.7
    )

    p.legend.location = "bottom_right"

    if save_dir is not None:
        bokeh.io.save(p, filename=save_dir)

    if show:
        bokeh.io.show(p)


def matplotlib_roc(
    actual: np.ndarray,
    pred_prob: np.ndarray,
    file_dir: str=None,
    show: bool=False
) -> None:
    assert file_dir is not None or show
    
    fpr, tpr, thresholds = metrics.roc_curve(
        actual,
        pred_prob
    )

    roc_auc = metrics.auc(fpr, tpr)

    plt.figure()
    lw = 0.5
    plt.plot(
        fpr, tpr, color="darkorange", lw=lw, label=f"ROC Curve (area = {roc_auc: 0.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    
    if show:
        plt.show()
    else:
        plt.savefig(file_dir)
    plt.close()