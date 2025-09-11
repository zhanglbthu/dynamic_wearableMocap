import torch
import math
import os
import matplotlib.pyplot as plt
import articulate as art
from articulate.utils.bullet.view_rotation_np import RotationViewer
from articulate.utils.pygame import StreamingDataViewer
from pygame.time import Clock

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
    data_dir = 'data/livedemo_processed'
    dataset_name = 'walk_20250817_193240.pt'

    dataset_path = os.path.join(data_dir, dataset_name)
    data = torch.load(dataset_path)

    oris = data['ori']   # 假设形状 [N, imu_num, 3, 3]
    oris_gt = data['ori_gt'] # 同上

    all_errs = []

    save_path = os.path.join(data_dir, dataset_name.split('.')[0], 'rotation_error.mp4')

    ori_real, ori_gt = oris[:, 1], oris_gt[:, 1]

    rR = torch.matmul(ori_real.transpose(-1, -2), ori_gt)
    rR_axis = art.math.rotation_matrix_to_axis_angle(rR).view(-1, 3)
    rR_euler = art.math.rotation_matrix_to_euler_angle(rR, seq='YZX').view(-1, 3)
    rR_euler = rR_euler * 180 / math.pi
    rR_q = art.math.axis_angle_to_quaternion(rR_axis).view(-1, 4)
    
    # rviewer = RotationViewer(1, order='wxyz'); rviewer.connect()
    sviewer = StreamingDataViewer(3, y_range=(-180, 180), window_length=100, names=['Y', 'Z', 'X']); sviewer.connect()
    clock = Clock()
    
    sviewer.start_recording(save_path, fps=30)
    for i in range(ori_real.shape[0]):
        clock.tick(30)
        rot_list = []
        rR = rR_q[i].cpu().numpy()
        rot_list.append(rR)
        
        # rviewer.update_all(rot_list)
        sviewer.plot(rR_euler[i].cpu().numpy())
    sviewer.stop_recording()


    # thetas = rotation_matrix_to_angle(ori_real, ori_gt) * 180 / math.pi

    # plt.figure(figsize=(20, 5))
    # plt.plot(rR_euler[:, 0].cpu(), label="Y")
    # plt.plot(rR_euler[:, 1].cpu(), label="Z")
    # plt.plot(rR_euler[:, 2].cpu(), label="X")
    # plt.xlabel("Frame index")
    # plt.ylabel("Rotation error (°)")
    # plt.title(f"Rotation error over time")  # 在标题里标序号
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()