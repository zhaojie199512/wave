import numpy as np
from lstm import MyLSTM


device = "cuda:0"
#
file_path = "data/human_count/run_free/0_free/target/0ren_500hz_3m_interval0.002s_20s_4wave.csv"
ckpt_path = "ckpts/epoch_52-loss_1.489.pth"
seq_len, seq_step = 200, 200

if __name__ == "__main__":
    classes = [f"{i}_free" for i in range(6)]
    print(classes)
    # 加载数据
    matrix = np.loadtxt(file_path, delimiter=",")
    X = np.array([matrix[s : s + seq_len] for s in np.arange(0, len(matrix) - seq_len + 1, seq_step)])
    # 加载分类器
    net = MyLSTM(seq_len=200, d_in=30, d_out=len(classes), d_hidden=64).to(device)
    net.load(ckpt_path)
    # 预测
    y = net.predict(X, device=device)
    print(y)
    c = np.bincount(y).argmax()
    print(f"Prediction result: {classes[c]}")
