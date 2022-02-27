import os

from sklearn.model_selection import train_test_split

from lstm import MyLSTM
from utils import evaluate, load_data, plot_roc

device = "cuda:0"
#
random_state = 0
data_name = "500hz_csi_data/human_count/run_circle"
#
model_path = "out/lstm.pth"

if __name__ == "__main__":
    # 加载数据
    classes = [f'{i}_free' for i in range(6)]
    print(classes)
    D_train, D_validate, D_evaluate = load_data(f"data/{data_name}", classes, seq_len=300, seq_step=10)
    # 训练分类器
    net = MyLSTM(seq_len=300, d_in=30, d_out=len(classes), d_hidden=64).to(device)
    net.fit(D_train, D_validate, device=device)
    net.evaluate(D_evaluate, device=device)
    # 评估模型
    y_pred = net.predict(D_evaluate[0], device=device)
    evaluate(D_evaluate[0], y_pred)
    # 画出ROC曲线
    Y_score = net.predict_proba(D_evaluate[0], device=device)
    plot_roc(D_evaluate[1], Y_score, classes, title=f"LSTM({data_name})", out_file="out/lstm.roc.png")
