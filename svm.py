import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics, model_selection, preprocessing
from sklearn.decomposition import PCA
from sklearn.svm import SVC

import os

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from utils import load_data, plot_roc, evaluate

data_dir = "data/500hz_csi_data/human_count/run_circle"
labels = os.listdir(data_dir)
random_state = 0

if __name__ == "__main__":
    # 加载数据
    X, y, Y = load_data(data_dir, labels, length=300, stride=30)
    X = X.reshape(X.shape[0], -1)
    # 切分数据
    X_train, X_test = train_test_split(X, test_size=0.4, random_state=random_state)
    y_train, y_test = train_test_split(y, test_size=0.4, random_state=random_state)
    Y_train, Y_test = train_test_split(Y, test_size=0.4, random_state=random_state)
    # 数据降维
    pca = PCA(n_components=100)
    pca.fit(X_train)
    X_train, X_test = pca.transform(X_train), pca.transform(X_test)
    # 训练分类器
    classifier = SVC(probability=True)
    classifier.fit(X_train, y_train)
    # 评估模型
    y_pred = classifier.predict(X_test)
    evaluate(y_test, y_pred)
    # 画出ROC曲线
    Y_score = classifier.predict_proba(X_test)
    plot_roc(Y_test, Y_score, labels, title="SVM - human_count/run_circle", out_file="out/svm.roc.png")
