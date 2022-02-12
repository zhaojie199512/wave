import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV

from utils import load_data, plot_roc, evaluate

if __name__ == "__main__":
    n_classes = 5
    data_dir = "data/500hz_csi_data/human_count/run_circle"
    # 加载数据
    X, y = load_data(data_dir, length=300, stride=30)
    Y = preprocessing.label_binarize(y, classes=np.arange(n_classes))  # onehot概率化
    # 切分数据
    X_train, X_test = model_selection.train_test_split(X, test_size=0.4, random_state=0)
    y_train, y_test = model_selection.train_test_split(y, test_size=0.4, random_state=0)
    Y_train, Y_test = model_selection.train_test_split(Y, test_size=0.4, random_state=0)
    # 数据降维
    pca = PCA(n_components=100)
    pca.fit(X_train)
    X_train, X_test = pca.transform(X_train), pca.transform(X_test)
    # 训练分类器
    classifier = LogisticRegressionCV(multi_class="multinomial", max_iter=2000)
    classifier.fit(X_train, y_train)
    # 评估模型
    y_pred = classifier.predict(X_test)
    evaluate(y_test, y_pred)
    # 画出ROC曲线
    Y_score = classifier.predict_proba(X_test)
    plot_roc(Y_test, Y_score, n_classes, title="Logistic Regression", out_file="out/logistic_regression.roc.png")
