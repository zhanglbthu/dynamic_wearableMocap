import matplotlib.pyplot as plt
import numpy as np

# 示例数据，包含模型名称和0到7米的误差数据
# data = {
#     "1": [0, 0.3348,    0.5727,	0.7263,	0.8466,	0.9675,	1.0835,	1.2103],
#     "1+3+9+27": [0, 0.2808,	0.4993,	0.6497,	0.7657,	0.8812,	1.0104,	1.1222],
#     "all+pose": [0, 0.2703, 0.4234, 0.5451, 0.6516, 0.7152, 0.7705, 0.8181],
#     "all": [0, 0.2506, 0.3762, 0.4683, 0.5276, 0.5717, 0.6191, 0.6506]
# }
data = {
    "MobilePoser": [0, 0.37,	0.57,	0.67,	0.75,	0.77,	0.79,	0.80],
    "Ours": [0, 0.31,	0.47,	0.64,	0.71,	0.67,	0.63,	0.55]
}

# x 轴表示真实行进距离
x = np.arange(0, 8, 1)  # 0 到 7 米

# 绘制 TotalCapture 数据集的图表
plt.figure(figsize=(10, 5))

for model, errors in data.items():
    plt.plot(x, errors, label=model)

# 设置标题和标签
plt.title("Real-World Dataset")
plt.xlabel("Real Travelled Distance (m)")
plt.ylabel("Mean Translation Error (m)")
plt.legend()
plt.grid(True)
plt.show()

# 如果需要绘制另一个类似的图表（IMUPoser Dataset），
# 可以重复上面的代码，只需更新数据。