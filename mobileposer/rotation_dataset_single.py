import torch
import math
import os
import matplotlib.pyplot as plt
from articulate.utils.bullet.view_rotation_np import RotationViewer
from articulate.utils.pygame import StreamingDataViewer
import articulate as art
from pygame.time import Clock
from config import paths

ji_mask = torch.tensor([18, 19, 1, 2, 15, 0])

def rotation_matrix_to_angle(R1, R2):
    """
    计算两个旋转矩阵序列的逐元素旋转角度误差
    R1, R2: [N, imu_num, 3, 3]
    返回: [N, imu_num] 误差角度（单位: 弧度）
    """
    # R_err = torch.matmul(R1.transpose(-1, -2), R2)  # [N, imu_num, 3, 3]
    R_err = R1.transpose(-1, -2) @ R2
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
    body_model = art.ParametricModel(paths.smpl_file)

    # convert to global
    
    all_errs = []

    save_dir = "data/rotation_error/imuposer/video"
    os.makedirs(save_dir, exist_ok=True)

    idx = 9
    save_path = os.path.join(save_dir, f'rotation_error_{idx}.mp4')
    
    ori, pose = oris[idx], poses[idx]
    pose = body_model.forward_kinematics(pose, calc_mesh=False)[0].view(-1, 24, 3, 3)
    ori_real, ori_gt = ori[:, 3], pose[:, ji_mask[3]]

    # rviewer = RotationViewer(3, order='wxyz'); rviewer.connect()
    sviewer = StreamingDataViewer(3, y_range=(-90, 90), window_length=100, names=['Y', 'Z', 'X']); sviewer.connect()
    clock = Clock()
    
    rR = torch.matmul(ori_real.transpose(-1, -2), ori_gt)
    rR_axis = art.math.rotation_matrix_to_axis_angle(rR).view(-1, 3)
    rR_euler = art.math.rotation_matrix_to_euler_angle(rR, seq='YZX').view(-1, 3)
    rR_euler = rR_euler * 180 / math.pi
    
    R_real_axis = art.math.rotation_matrix_to_axis_angle(ori_real).view(-1, 3)
    R_gt_axis = art.math.rotation_matrix_to_axis_angle(ori_gt).view(-1, 3)
    R_real_q = art.math.axis_angle_to_quaternion(R_real_axis).view(-1, 4)
    R_gt_q = art.math.axis_angle_to_quaternion(R_gt_axis).view(-1, 4)
    rR_q = art.math.axis_angle_to_quaternion(rR_axis).view(-1, 4)
    
    sviewer.start_recording(save_path, fps=30)
    for i in range(ori_real.shape[0]):
        clock.tick(30)
        # rot_list = []
        # R_gt, R_real, rR = R_gt_q[i].cpu().numpy(), R_real_q[i].cpu().numpy(), rR_q[i].cpu().numpy()
        # rot_list.append(R_gt)
        # rot_list.append(R_real)
        # rot_list.append(rR)
        
        # rviewer.update_all(rot_list)
        sviewer.plot(rR_euler[i].cpu().numpy())
    sviewer.stop_recording()