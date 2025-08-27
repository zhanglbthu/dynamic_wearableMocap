import torch
import math
import os
import matplotlib.pyplot as plt
import articulate as art

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

    save_dir = "data/rotation_error"
    os.makedirs(save_dir, exist_ok=True)

    for idx, (ori, pose) in enumerate(zip(oris, poses)):
        ori_real, ori_gt = ori[:, 3], pose[:, ji_mask[3]]

        rR = torch.matmul(ori_real.transpose(-1, -2), ori_gt)
        rR_axis = art.math.rotation_matrix_to_axis_angle(rR).view(-1, 3)
        rR_euler = art.math.rotation_matrix_to_euler_angle(rR, seq='YZX').view(-1, 3)
        rR_euler = rR_euler * 180 / math.pi
        
        thetas = rotation_matrix_to_angle(ori_real, ori_gt) * 180 / math.pi

        plt.figure(figsize=(20, 5))
        plt.plot(thetas.cpu(), label="deg")

        plt.xlabel("Frame index")
        plt.ylabel("Rotation error (°)")
        plt.title(f"Rotation error over time (Sequence {idx})")  # 在标题里标序号
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()