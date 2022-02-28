import joblib

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from utils import evaluate, load_data, plot_roc

#
random_state = 0
data_name = "human_count/run_free"


if __name__ == "__main__":
    # 加载数据
    classes = [f"{i}_free" for i in range(6)]
    print(classes)
    data = load_data(f"data/{data_name}", classes, seq_len=200, seq_step=20)
    data["X_train"] = data["X_train"].reshape(len(data["X_train"]), -1)
    data["X_validate"] = data["X_validate"].reshape(len(data["X_validate"]), -1)
    # 数据降维
    pca = PCA(n_components=100)
    pca.fit(data["X_train"])
    data["X_train"], data["X_validate"] = pca.transform(data["X_train"]), pca.transform(data["X_validate"])
    joblib.dump(pca, "out/pca.model")
    # 训练分类器
    classifier = RandomForestClassifier()
    classifier.fit(data["X_train"], data["y_train"])
    # 保存模型
    joblib.dump(classifier, "out/rf.model")
    # 评估模型
    y_pred = classifier.predict(data["X_validate"])
    evaluate(data["y_validate"], y_pred)
    # 画出ROC曲线
    Y_score = classifier.predict_proba(data["X_validate"])
    plot_roc(
        data["Y_validate"], Y_score, classes, title=f"Random Forest({data_name})", out_file="out/random_forest.roc.png"
    )
