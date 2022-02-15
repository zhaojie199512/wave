import os

from sklearn.model_selection import train_test_split

from lstm import MyLSTM
from utils import evaluate, load_data, plot_roc

device = "cuda:0"
#
random_state = 0
data_name = "500hz_csi_data/human_count/run_circle"

if __name__ == "__main__":
    # 加载数据
    classes = os.listdir(f"data/{data_name}")
    X, y, Y = load_data(f"data/{data_name}", classes, length=300, stride=10)
    # 切分数据
    X_train, X_test, y_train, y_test, Y_train, Y_test = train_test_split(
        X, y, Y, test_size=0.4, random_state=random_state
    )
    # 训练分类器
    net = MyLSTM(seq_len=300, d_in=30, d_out=5, d_hidden=64).to(device)
    net.fit((X_train, y_train), (X_test, y_test), device=device)
    # 保存模型
    net.save('out/lstm.pth')
    # 评估模型
    y_pred = net.predict(X_test, device=device)
    evaluate(y_test, y_pred)
    # 画出ROC曲线
    Y_score = net.predict_proba(X_test, device=device)
    plot_roc(Y_test, Y_score, classes, title=f"LSTM({data_name})", out_file="out/lstm.roc.png")
