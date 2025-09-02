import torch
import math
import os
import matplotlib.pyplot as plt
import articulate as art
from config import paths

ji_mask = torch.tensor([18, 19, 1, 2, 15, 0])

def rotation_matrix_to_angle(R1, R2):
    """
    计算两个旋转矩阵序列的逐元素旋转角度误差
    R1, R2: [N, imu_num, 3, 3]
    返回: [N, imu_num] 误差角度（单位: 弧度）
    """
    R_err = torch.matmul(R1.transpose(-1, -2), R2)  # [N, imu_num, 3, 3]
    trace = R_err.diagonal(dim1=-2, dim2=-1).sum(-1)  # [N, imu_num]
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)
    return theta  # [N, imu_num]

if __name__ == '__main__':
    data_dir = 'data/dataset'
    dataset_name = 'imuposer_test_25fps.pt'

    dataset_path = os.path.join(data_dir, dataset_name)
    data = torch.load(dataset_path)

    oris = data['ori']   # 假设形状 [N, imu_num, 3, 3]
    poses = data['pose'] # 同上

    all_errs = []
    body_model = art.ParametricModel(paths.smpl_file)
    for ori, pose in zip(oris, poses):
        pose = body_model.forward_kinematics(pose, calc_mesh=False)[0].view(-1, 24, 3, 3)
        ori_real, ori_gt = ori[:, 3], pose[:, ji_mask[3]]  # 取出关节对应的旋转矩阵
        err = rotation_matrix_to_angle(ori_real, ori_gt)  # [N, imu_num]
        err_deg = err * 180 / math.pi
        all_errs.append(err_deg.mean(dim=0))  

    all_errs = torch.stack(all_errs)  # [N, 1]

    # 绘制柱状图
    plt.figure(figsize=(16, 10))
    x = range(all_errs.shape[0])
    plt.bar(x, all_errs.numpy())
    plt.xticks(x, [f"{i}" for i in range(all_errs.shape[0])])
    plt.ylabel("Mean Angle Error (degrees)")
    plt.title("Average Rotation Error per IMU")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

    # ---------- 可视化 3: 箱型图 ----------
    plt.figure(figsize=(8, 10))
    plt.boxplot(all_errs.numpy(), vert=True, labels=["Error"])
    plt.ylabel("Angle Error (degrees)")
    plt.title("Error Distribution (Boxplot)")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.show()