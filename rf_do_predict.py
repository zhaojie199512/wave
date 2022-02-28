import joblib
import numpy as np

#
file_path = "data/human_count/run_free/0_free/target/0ren_500hz_3m_interval0.002s_20_1wave.csv"


if __name__ == "__main__":
    # 加载数据
    classes = [f"{i}_free" for i in range(6)]
    length, stride = 200, 200
    arr = np.loadtxt(file_path, delimiter=",")
    X = np.array([arr[s : s + length] for s in np.arange(0, len(arr) - length + 1, stride)])
    X = X.reshape(len(X), -1)
    # 加载模型
    pca = joblib.load("out/pca.model")
    classifier = joblib.load("out/rf.model")
    # 数据降维
    X = pca.transform(X)
    # 预测
    y = classifier.predict(X)
    print(y)
    c = np.bincount(y).argmax()
    print(f"Prediction result: {classes[c]}")
