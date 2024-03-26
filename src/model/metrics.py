import numpy as np
import torch
import torchmetrics
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def get_metrics(config: DictConfig):
    metrics = torchmetrics.MetricCollection(
        [
            torchmetrics.Accuracy(
                task=config.metrics.task,
                num_classes=config.num_classes,
                average=config.metrics.average,
                top_k=config.metrics.top_k,
            ),
            torchmetrics.Precision(
                task=config.metrics.task,
                num_classes=config.num_classes,
                average=config.metrics.average,
                top_k=config.metrics.top_k,
            ),
            torchmetrics.Recall(
                task=config.metrics.task,
                num_classes=config.num_classes,
                average=config.metrics.average,
                top_k=config.metrics.top_k,
            ),
            torchmetrics.Specificity(
                task=config.metrics.task,
                num_classes=config.num_classes,
                average=config.metrics.average,
                top_k=config.metrics.top_k,
            ),
            torchmetrics.F1Score(
                task=config.metrics.task,
                num_classes=config.num_classes,
                average=config.metrics.average,
                top_k=config.metrics.top_k,
            ),
            torchmetrics.FBetaScore(
                task=config.metrics.task,
                num_classes=config.num_classes,
                average=config.metrics.average,
                top_k=config.metrics.top_k,
            ),
        ]
    )

    return metrics


def get_classification_metrics(true, y_proba, y_pred, t_onehot, config: DictConfig):
    accuracy = accuracy_score(true, y_pred)
    precision = precision_score(true, y_pred, average=config.metrics.average)
    recall = recall_score(true, y_pred, average=config.metrics.average)
    f1 = f1_score(true, y_pred, average=config.metrics.average)
    specificity = torchmetrics.functional.specificity(
        torch.tensor(y_pred),
        torch.tensor(true),
        task=config.metrics.task,
        num_classes=config.num_classes,
        average=config.metrics.average,
    )
    kappa = cohen_kappa_score(true, y_pred)
    auc = roc_auc_score(t_onehot, y_proba, average="macro")

    results = (accuracy, precision, recall, f1, specificity, kappa, auc)

    return results


def get_classification_report(true, y_pred, int_to_label):
    true = [int_to_label[true_] for true_ in true]
    y_pred = [int_to_label[y_pred_] for y_pred_ in y_pred]
    return classification_report(true, y_pred)


def get_confusion_matrix(true, y_pred, labels):
    cm = confusion_matrix(true, y_pred, labels=labels)
    return cm


def get_roc_curve(t_onehot, y_proba, config: DictConfig):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(config.num_classes):
        fpr[i], tpr[i], _ = roc_curve(t_onehot[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(t_onehot.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(config.num_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(config.num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= config.num_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc
