"""
Methods in this file visualize the metrics to evaluate model
using AUC of ROC.
"""
import numpy as np
import sklearn
from sklearn import metrics

fpr, tpr, thresholds = metrics.roc_curve()
