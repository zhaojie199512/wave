import os
import random
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics, preprocessing


def load_data(data_dir, classes, *, seq_len=300, seq_stride=30):
    data = [{"X": [], "y": [], "Y": []} for _ in range(3)]
    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)
    for i, c in enumerate(classes):
        files = [f for f in (data_dir / c / "target").iterdir() if f.name.endswith("wave.csv")]
        ranges = [0, int(0.6 * len(files)), int(0.8 * len(files)), len(files)]
        random.shuffle(files)
        for item, start, end in zip(data, ranges[:-1], ranges[1:]):
            for f in files[start:end]:
                matrix = np.loadtxt(f, delimiter=",")
                indices = np.arange(0, len(matrix) - seq_len + 1, seq_stride)
                item["X"] += [matrix[s: s + seq_len] for s in indices]
                item["y"] += [i] * len(files)
                print(f'{f}: {len(matrix)} rows, {len(indices)} sequences')
    for item in data:
        item["X"] = np.array(item["X"])
        item["y"] = np.array(item["y"])
        item["Y"] = preprocessing.label_binarize(item["y"], classes=range(len(classes)))
    return data


def plot_roc(Y_test, Y_score, classes, title, out_file=None):
    plt.figure(figsize=[10, 10], dpi=100)
    plt.title(title)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    # 所有类别的 FPR、TPR
    fpr_all, tpr_all = [], []
    for i, c in enumerate(classes):
        fpr, tpr, _ = metrics.roc_curve(Y_test[:, i], Y_score[:, i])
        fpr_all += [fpr]
        tpr_all += [tpr]
        auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label=f"ROC curve of class {c} (AUC = {auc:0.2f})")
    # micro auc直接展平
    micro_fpr, micro_tpr, _ = metrics.roc_curve(Y_test.ravel(), Y_score.ravel())
    micro_auc = metrics.auc(micro_fpr, micro_tpr)
    plt.plot(micro_fpr, micro_tpr, "--", lw=1, label=f"micro-average ROC curve (AUC = {micro_auc:0.2f})")
    # macro auc 插值取平均
    macro_fpr = np.unique(np.concatenate(fpr_all))
    macro_tpr = sum([np.interp(macro_fpr, fpr, tpr) for fpr, tpr in zip(fpr_all, tpr_all)]) / len(classes)
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
    print(f"● Accuracy            = {metrics.accuracy_score(y_true, y_pred):.3f}")
    #
    print(f"● Precision(macro)    = {metrics.precision_score(y_true, y_pred, average='macro'):.3f}")
    print(f"● Precision(micro)    = {metrics.precision_score(y_true, y_pred, average='micro'):.3f}")
    print(f"● Precision(weighted) = {metrics.precision_score(y_true, y_pred, average='weighted'):.3f}")
    #
    print(f"● Recall(macro)       = {metrics.precision_score(y_true, y_pred, average='macro'):.3f}")
    print(f"● Recall(micro)       = {metrics.precision_score(y_true, y_pred, average='micro'):.3f}")
    print(f"● Recall(weighted)    = {metrics.precision_score(y_true, y_pred, average='weighted'):.3f}")
    #
    print(f"● F1(macro)           = {metrics.precision_score(y_true, y_pred, average='macro'):.3f}")
    print(f"● F1(micro)           = {metrics.precision_score(y_true, y_pred, average='micro'):.3f}")
    print(f"● F1(weighted)        = {metrics.precision_score(y_true, y_pred, average='weighted'):.3f}")
    #
    print(f"● Confusion Matrix    =\n{metrics.confusion_matrix(y_true, y_pred)}")


if __name__ == "__main__":
    data = load_data("data/human_count/run_free", [f"{i}_free" for i in range(6)])
    print(data[0]["X"].shape)
    print(data[1]["X"].shape)
    print(data[2]["X"].shape)
