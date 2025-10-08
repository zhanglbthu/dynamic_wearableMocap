import torch
import math
import os
import matplotlib.pyplot as plt
from articulate.utils.bullet.view_rotation_np import RotationViewer
from articulate.utils.pygame import StreamingDataViewer
import articulate as art
from pygame.time import Clock
from config import paths
import numpy as np

ji_mask = torch.tensor([18, 19, 1, 2, 15, 0])

def save_rotation_plot(R_real_euler, R_gt_euler, axis=2, fig_path="rotation_plot.png"):
    """
    保存对齐后的欧拉角分量曲线对比图

    Args:
        R_real_euler (torch.Tensor or np.ndarray): [T, 3] IMU 欧拉角 (deg)
        R_gt_euler   (torch.Tensor or np.ndarray): [T, 3] 视觉欧拉角 (deg)
        axis (int): 选择的欧拉角分量 (0=X, 1=Y, 2=Z)
        fig_path (str): 保存路径
    """
    # 保证是 numpy
    if isinstance(R_real_euler, torch.Tensor):
        x_real = R_real_euler[:, axis].detach().cpu().numpy()
    else:
        x_real = R_real_euler[:, axis]
    if isinstance(R_gt_euler, torch.Tensor):
        x_gt = R_gt_euler[:, axis].detach().cpu().numpy()
    else:
        x_gt = R_gt_euler[:, axis]

    plt.figure(figsize=(10, 5))
    plt.plot(x_real, label='Real (IMU)', alpha=0.7)
    plt.plot(x_gt, label='GT (Vision)', alpha=0.7)
    plt.xlabel("Frame index")
    plt.ylabel("Rotation angle (deg)")
    plt.title(f"Euler axis {axis} comparison (aligned)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"✅ Saved static plot to {fig_path}")

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

def compute_lag(x, y):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    corr = np.correlate(x - np.mean(x), y - np.mean(y), mode='full')
    lag = np.argmax(corr) - (len(x) - 1)
    return lag

def view_axis(ori_real, ori_gt, save_path=None, fig_path=None, axis=2):

    R_real_euler = art.math.rotation_matrix_to_euler_angle(ori_real, seq='YZX').view(-1, 3)
    R_gt_euler = art.math.rotation_matrix_to_euler_angle(ori_gt, seq='YZX').view(-1, 3)
    
    R_real_euler = R_real_euler * 180 / math.pi
    R_gt_euler = R_gt_euler * 180 / math.pi

    # align
    lag = compute_lag(R_real_euler[:, axis], R_gt_euler[:, axis])
    print(f"Lag between real and gt (axis {axis}): {lag}")
    if lag > 0:
        R_real_euler = R_real_euler[lag:]
        R_gt_euler   = R_gt_euler[:-lag]
    elif lag < 0:
        R_real_euler = R_real_euler[:lag]
        R_gt_euler   = R_gt_euler[-lag:]

    if save_path is not None:
        sviewer = StreamingDataViewer(2, y_range=(-90, 90), window_length=60, names=['X_real', 'X_gt']); sviewer.connect()
        clock = Clock()
        sviewer.start_recording(save_path, fps=30)
        for i in range(ori_real.shape[0]):
            clock.tick(30)
            sviewer.plot([R_real_euler[i][axis].cpu().numpy(), R_gt_euler[i][axis].cpu().numpy()])
        sviewer.stop_recording()
    
    if fig_path is not None:
        save_rotation_plot(R_real_euler, R_gt_euler, axis=axis, fig_path=fig_path)

def view_delta(ori_real, ori_gt, save_path=None, lag=None):
    
    if lag is not None:
        if lag > 0:
            ori_real = ori_real[lag:]
            ori_gt   = ori_gt[:-lag]
        elif lag < 0:
            ori_real = ori_real[:lag]
            ori_gt   = ori_gt[-lag:]
    
    rR = torch.matmul(ori_real.transpose(-1, -2), ori_gt)
    rR_euler = art.math.rotation_matrix_to_euler_angle(rR, seq='YZX').view(-1, 3)
    rR_euler = rR_euler * 180 / math.pi
    
    if save_path is not None:
        sviewer = StreamingDataViewer(3, y_range=(-180, 180), window_length=100, names=['Y', 'Z', 'X']); sviewer.connect()
        clock = Clock()
        sviewer.start_recording(save_path, fps=30)
        for i in range(ori_real.shape[0]):
            clock.tick(30)
            sviewer.plot(rR_euler[i].cpu().numpy())
        sviewer.stop_recording()

def view_abs_angle(ori_real, ori_gt, save_path=None, lag=None):
    """
    分别可视化 ori_real 和 ori_gt 的旋转角度大小（相对于单位矩阵I）
    """

    if lag is not None:
        if lag > 0:
            ori_real = ori_real[lag:]
            ori_gt   = ori_gt[:-lag]
        elif lag < 0:
            ori_real = ori_real[:lag]
            ori_gt   = ori_gt[-lag:]

    def rotation_angle(R):
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        return trace
        cos_theta = (trace - 1) / 2
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        return torch.acos(cos_theta) * 180 / math.pi  # 角度制

    real_angles = rotation_angle(ori_real)  # [T]
    gt_angles   = rotation_angle(ori_gt)    # [T]

    if save_path is not None:
        sviewer = StreamingDataViewer(2, y_range=(0, 5), window_length=100, names=['Real_Angle', 'GT_Angle']); sviewer.connect()
        clock = Clock()
        sviewer.start_recording(save_path, fps=30)
        for i in range(real_angles.shape[0]):
            clock.tick(30)
            sviewer.plot([real_angles[i].item(), gt_angles[i].item()])
        sviewer.stop_recording()

    return real_angles, gt_angles

def get_rot_lag(ori_real, ori_gt):
    R_real_euler = art.math.rotation_matrix_to_euler_angle(ori_real, seq='YZX').view(-1, 3)
    R_gt_euler = art.math.rotation_matrix_to_euler_angle(ori_gt, seq='YZX').view(-1, 3)
    
    R_real_euler = R_real_euler * 180 / math.pi
    R_gt_euler = R_gt_euler * 180 / math.pi

    lag = compute_lag(R_real_euler[:, 2], R_gt_euler[:, 2])
    return lag

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

    idx = 23
    
    ori, pose = oris[idx], poses[idx]
    pose = body_model.forward_kinematics(pose, calc_mesh=False)[0].view(-1, 24, 3, 3)
    ori_real, ori_gt = ori[:, 3], pose[:, ji_mask[3]]

    # # view axis
    # axis = 2
    # save_path = os.path.join(save_dir, f'idx{idx}_axis{axis}.mp4')
    # fig_path = os.path.join(save_dir, f'idx{idx}_axis{axis}.png')
    # view_axis(ori_real, ori_gt, fig_path=fig_path, axis=axis)
    
    lag = get_rot_lag(ori_real, ori_gt)
    print(f"Overall lag: {lag}")
    save_path = os.path.join(save_dir, f'idx{idx}_delta_aligned_lag{lag}.mp4')
    fig_path = os.path.join(save_dir, f'idx{idx}_delta.png')
    view_delta(ori_real, ori_gt, lag=lag, save_path=save_path)

    # # view abs angle
    # save_path = os.path.join(save_dir, f'idx{idx}_abs_angle_aligned_lag{lag}.mp4')
    # view_abs_angle(ori_real, ori_gt, save_path=save_path, lag=lag)
    