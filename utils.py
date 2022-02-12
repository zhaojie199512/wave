import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics, preprocessing


def load_data(data_dir, labels, length, stride):
    X, y = [], []
    for i, label in enumerate(labels):
        for file in os.listdir(f"{data_dir}/{label}/target"):
            if file.endswith("wave.csv"):
                mat = np.loadtxt(f"{data_dir}/{label}/target/{file}", delimiter=",")
                ind = np.arange(0, len(mat) - length + 1, stride)
                X += [mat[s: s + length] for s in ind]
                y += [i] * len(ind)
    X, y = np.array(X), np.array(y)
    Y = preprocessing.label_binarize(y, classes=np.arange(len(labels)))  # onehot概率化
    return X, y, Y


def plot_roc(Y_test, Y_score, labels, title=None, out_file=None):
    plt.figure(figsize=[8, 8], dpi=100)
    if title is not None:
        plt.title(title)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    # 所有类别的 FPR、TPR
    fpr_all, tpr_all = [], []
    for i, label in enumerate(labels):
        fpr, tpr, _ = metrics.roc_curve(Y_test[:, i], Y_score[:, i])
        fpr_all += [fpr]
        tpr_all += [tpr]
        auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label=f"ROC curve of class {label} (AUC = {auc:0.2f})")
    # micro auc直接展平
    micro_fpr, micro_tpr, _ = metrics.roc_curve(Y_test.ravel(), Y_score.ravel())
    micro_auc = metrics.auc(micro_fpr, micro_tpr)
    plt.plot(micro_fpr, micro_tpr, "--", lw=1, label=f"micro-average ROC curve (AUC = {micro_auc:0.2f})")
    # macro auc 插值取平均
    macro_fpr = np.unique(np.concatenate(fpr_all))
    macro_tpr = sum([np.interp(macro_fpr, fpr, tpr) for fpr, tpr in zip(fpr_all, tpr_all)]) / len(labels)
    macro_auc = metrics.auc(macro_fpr, macro_tpr)
    plt.plot(macro_fpr, macro_tpr, "--", lw=1, label=f"macro-average ROC curve (AUC = {macro_auc:0.2f})")
    # 画出斜线
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    #
    plt.legend()
    if out_file is not None:
        plt.savefig(out_file)
    plt.show()


def evaluate(y_true, y_pred):
    scores = {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "macro_precision": metrics.precision_score(y_true, y_pred, average="macro"),
        "micro_precision": metrics.precision_score(y_true, y_pred, average="micro"),
        "weighted_precision": metrics.precision_score(y_true, y_pred, average="weighted"),
        "macro_recall": metrics.recall_score(y_true, y_pred, average="macro"),
        "micro_recall": metrics.recall_score(y_true, y_pred, average="micro"),
        "weighted_recall": metrics.recall_score(y_true, y_pred, average="weighted"),
        "macro_f1": metrics.f1_score(y_true, y_pred, average="macro"),
        "micro_f1": metrics.f1_score(y_true, y_pred, average="micro"),
        "weighted_f1": metrics.f1_score(y_true, y_pred, average="weighted")
    }
    for k, score in scores.items():
        print(f'{k:>20}: {score:.3f}')
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    print(f'    confusion_matrix:\n{confusion_matrix}')
