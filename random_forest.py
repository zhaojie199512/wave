import os

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from utils import evaluate, load_data, plot_roc

#
random_state = 0
data_name = "human_count/run_free"


if __name__ == "__main__":
    # 加载数据
    classes = [f"{i}_free" for i in range(6)]
    print(classes)
    data = load_data(f"data/{data_name}", classes, seq_len=200, seq_step=200)
    X_train, y_train = data["X_train"].reshape(len(data["X_train"]), -1), data["y_train"]
    X_validate, y_validate, Y_validate = (
        data["X_validate"].reshape(len(data["X_validate"]), -1),
        data["y_validate"],
        data["Y_validate"],
    )
    # 数据降维
    pca = PCA(n_components=100)
    pca.fit(X_train)
    X_train, X_validate = pca.transform(X_train), pca.transform(X_validate)
    # 训练分类器
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    # 评估模型
    y_pred = classifier.predict(X_validate)
    evaluate(y_validate, y_pred)
    # 画出ROC曲线
    Y_score = classifier.predict_proba(X_validate)
    plot_roc(Y_validate, Y_score, classes, title=f"Random Forest({data_name})", out_file="out/random_forest.roc.png")
