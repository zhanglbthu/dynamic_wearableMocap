import torch
import math
import articulate as art
from articulate.utils.pygame import StreamingDataViewer
from pygame.time import Clock
import os
import matplotlib.pyplot as plt
        
def rotation_matrix_to_angle(R1, R2):
    """
    计算两个旋转矩阵序列的逐元素旋转角度误差
    R1, R2: [N, imu_num, 3, 3]
    返回: [N, imu_num] 误差角度（单位: 弧度）
    """
    R_err = torch.matmul(R1, R2.transpose(-1, -2))  # [N, imu_num, 3, 3]
    trace = R_err.diagonal(dim1=-2, dim2=-1).sum(-1)  # [N, imu_num]
    # 数值稳定性，保证输入在 [-1, 1]
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)
    return theta

if __name__ == '__main__':
    data_dir = 'data/livedemo/sit_20250811_180140'
    sensor_data_name = 'sit_20250811_180140.pt'
    baseline_name = 'sit_20250811_180140_tic.pt'
    ours_name = 'RealData_cali_pred.pt'
    
    sensor_data_path = os.path.join(data_dir, sensor_data_name)
    baseline_path = os.path.join(data_dir, baseline_name)
    ours_path = os.path.join(data_dir, ours_name)
    
    sensor_data = torch.load(sensor_data_path)
    baseline_data = torch.load(baseline_path)
    ours_data = torch.load(ours_path)
    
    rot, rot_gt = sensor_data['ori'], sensor_data['ori_gt']
    rot_baseline = baseline_data['rot_p']
    rot_ours = ours_data['rot_p']
    print(f'rot shape: {rot.shape}, rot_gt shape: {rot_gt.shape}, rot_baseline shape: {rot_baseline.shape}, rot_ours shape: {rot_ours.shape}')
    # [N, imu_num, 3, 3]
    imu_idx = 1
    
    # 计算误差
    err_rot = rotation_matrix_to_angle(rot[:, imu_idx], rot_gt[:, imu_idx])        # [N]
    err_baseline = rotation_matrix_to_angle(rot_baseline[:, imu_idx], rot_gt[:, imu_idx])
    err_ours = rotation_matrix_to_angle(rot_ours[:, imu_idx], rot_gt[:, imu_idx])

    # 拼接结果
    results = torch.stack([err_rot, err_baseline, err_ours], dim=1)  # [N, 3]
    results = results * 180 / torch.pi  # 转换为度数
    # 去掉前后 10帧
    results = results[400:-100]  # [N-20, 3]
    
    # # Streaming Visualization
    # sviewer = StreamingDataViewer(3, (0, 80), 200, names=['raw', 'baseline', 'ours']); sviewer.connect()
    # clock = Clock()
    # for i in range(results.shape[0]):
    #     sviewer.plot(results[i].cpu().numpy())
    #     clock.tick(30)
    
    # whole visualization
    plt.figure(figsize=(20, 5))
    plt.plot(results[:, 0].cpu(), label="raw")
    plt.plot(results[:, 1].cpu(), label="baseline")
    plt.plot(results[:, 2].cpu(), label="ours")
    plt.xlabel("Frame index")
    plt.ylabel("Rotation error (°)")
    plt.title("Rotation error over time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # mean_errors = results.mean(dim=0).cpu().numpy()  # [3]
    # std_errors = results.std(dim=0).cpu().numpy()    # [3]
    # names = ["raw", "baseline", "ours"]
    # plt.figure(figsize=(6, 5))
    # plt.bar(names, mean_errors, yerr=std_errors, capsize=5)
    # plt.ylabel("Rotation error (°)")
    # plt.title("Average rotation error with std")
    # plt.grid(axis="y")
    # plt.tight_layout()
    # plt.show()