import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics, model_selection, preprocessing
from sklearn.decomposition import PCA
from sklearn.svm import SVC


def main():
    n_classes = 5
    data_dir = "data/500hz_csi_data/human_count/run_circle"
    #
    X, y = load_data(data_dir, length=300, stride=10)
    Y = preprocessing.label_binarize(y, classes=np.arange(n_classes))
    #
    X_train, X_test = model_selection.train_test_split(X, test_size=0.4, random_state=0)
    y_train, y_test = model_selection.train_test_split(y, test_size=0.4, random_state=0)
    Y_train, Y_test = model_selection.train_test_split(Y, test_size=0.4, random_state=0)
    #
    pca = PCA(n_components=32)
    pca.fit(X_train)
    X_train, X_test = pca.transform(X_train), pca.transform(X_test)
    #
    classifier = SVC(probability=True)
    classifier.fit(X_train, y_train)
    acc = classifier.score(X_test, y_test)
    print(f"Accuracy = {acc:.2f}")
    #
    Y_score = classifier.predict_proba(X_test)
    plot_roc(Y_test, Y_score, n_classes, out_file="out/svm.png")


def load_data(data_dir, length, stride):
    X, y = [], []
    for i, category in enumerate(os.listdir(data_dir)):
        for file in os.listdir(f"{data_dir}/{category}/target"):
            if file.endswith("wave.csv"):
                mat = np.loadtxt(f"{data_dir}/{category}/target/{file}", delimiter=",")
                ind = np.arange(0, len(mat) - length + 1, stride)
                X += [mat[s: s + length].flatten() for s in ind]
                y += [i] * len(ind)
    return np.array(X), np.array(y)


def plot_roc(Y_test, Y_score, n_classes, out_file=None):
    # 计算每一类的ROC
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(Y_test[:, i], Y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(Y_test.ravel(), Y_score.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    fpr_all = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    tpr_ave = np.zeros_like(fpr_all)
    for i in range(n_classes):
        tpr_ave += np.interp(fpr_all, fpr[i], tpr[i])
    tpr_ave /= n_classes

    fpr["macro"], tpr["macro"] = fpr_all, tpr_ave
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=[8, 8], dpi=80)
    plt.plot(fpr["micro"], tpr["micro"], "--", lw=1, label=f"micro-average ROC curve (AUC = {roc_auc['micro']:0.2f})")
    plt.plot(fpr["macro"], tpr["macro"], "--", lw=1, label=f"macro-average ROC curve (AUC = {roc_auc['macro']:0.2f})")
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=1, label=f"ROC curve of class {i} (AUC = {roc_auc[i]:0.2f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(loc="lower right")
    if out_file is not None:
        plt.savefig(out_file)
    plt.show()


if __name__ == "__main__":
    main()
