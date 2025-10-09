import articulate as art
from config import *
from my_model import *
from TicOperator import *
from evaluation_functions import *
from argparse import ArgumentParser
from utils.model_utils import load_mobileposer_model, load_heightposer_model
import matplotlib.pyplot as plt
import os

imu_num = config.imu_num
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_tic = TIC(stack=3, n_input=imu_num * (3 + 3 * 3), n_output=imu_num * 6)
model_tic.restore('data/checkpoints/calibrator/TIC_MP/TIC_20.pth')

model_lstm = LSTMIC(n_input=imu_num * (3 + 3 * 3), n_output=imu_num * 6)
model_lstm.restore('data/checkpoints/calibrator/RealData_0915_4pants_12345/RealData_0915_4pants_12345_20.pth')

model_tic = model_tic.to(device).eval()
model_lstm = model_lstm.to(device).eval()

# folders = ['sit_20250811_180140']
eval_dir = "/root/autodl-tmp/data/Real_Dataset/RealData_Raw_0915_types/pants_2"
# folders = os.listdir(eval_dir)
folders = [f for f in os.listdir(eval_dir) if f.endswith('.pt')]
print(f'Found folders: {folders}')
# folders = ['s1']

def smooth_curve(data, window_size=15):
    """对数据进行滑动平均平滑"""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def rotation_matrix_error(R_pred, R_gt):
    """Compute angular error (radians) between two rotation matrices."""
    # convert to numpy
    R_pred = R_pred.cpu().numpy()
    R_gt = R_gt.cpu().numpy()
    # Ensure square 3x3
    assert R_pred.shape == (3, 3)
    assert R_gt.shape == (3, 3)
    R_diff = R_pred @ R_gt.T
    cos_theta = (np.trace(R_diff) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # numerical stability
    return np.arccos(cos_theta)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='data/checkpoints/mobileposer/lw_rp/base_model.pth')
    parser.add_argument('--dataset', type=str, default='dip')
    parser.add_argument('--use_cali', action='store_true')
    args = parser.parse_args()
    
    print('=====Inference Start=====')
    types = ['w/o_cali', 'ours(real)', 'baseline(syn)', 'gt']
    global_errors = {t: {'watch': [], 'phone': []} for t in types if t != 'gt'}

    for f in folders:
        f = f.split('.')[0]
        data_root = os.path.join(eval_dir, f)
        os.makedirs(data_root, exist_ok=True)

        data = torch.load(os.path.join(eval_dir, f + '.pt'))
        imu_acc = data['acc'].to(device)[100:-100]
        imu_rot = data['ori'].to(device)[100:-100]
        gt_acc  = data['acc_gt'].to(device)[100:-100]
        gt_rot  = data['ori_gt'].to(device)[100:-100]

        print(f'Processing {f}')

        # 用于存储该文件不同type的误差曲线
        file_errors = {t: {'watch': None, 'phone': None} for t in types if t != 'gt'}

        for type in types:
            if type == 'w/o_cali':
                rot = imu_rot
                rot = gt_rot.clone()  
                acc = imu_acc
            elif type == 'gt':
                rot = gt_rot
                acc = gt_acc
            elif type == 'ours(real)':
                imu_rot = gt_rot.clone()  
                ts = TicOperator(TIC_network=model_lstm, imu_num=imu_num, data_frame_rate=30)
                rot, acc, pred_offset, use_cali = ts.run_per_frame(imu_rot, imu_acc)
            elif type == 'baseline(syn)':
                imu_rot = gt_rot.clone()
                ts = TicOperator(TIC_network=model_tic, imu_num=imu_num, data_frame_rate=30)
                rot, acc, _, _, _ = ts.run(imu_rot, imu_acc)
            else:
                raise "Unknown type"

            if type != 'gt' and not f.startswith('live'):
                errors = {'watch': [], 'phone': []}
                for R_pred, R_gt in zip(rot, gt_rot):
                    errors['watch'].append(rotation_matrix_error(R_pred[0], R_gt[0]))
                    errors['phone'].append(rotation_matrix_error(R_pred[1], R_gt[1]))

                # 转为 numpy 数组（单位：degree）
                for key in errors:
                    frame_errors = np.array(errors[key]) * 180 / np.pi
                    mean_error = np.mean(frame_errors)
                    print(f"Mean rotation error for {key} ({type}): {mean_error:.2f}°", end='; ')

                    # 保存每帧误差曲线（供绘图使用）
                    file_errors[type][key] = frame_errors

                    # 全局误差累积
                    global_errors[type][key].extend(frame_errors.tolist())
                print()

                # # * mocap module
                # acc = acc / amass.acc_scale
                # x = torch.cat((acc.flatten(1), rot.flatten(1)), dim=1).to(device)
                # # mocap = load_mobileposer_model(model_path=args.model, combo_id=model_config.combo_id)
                # mocap = load_mobileposer_model(model_path=args.model, combo_id=model_config.combo_id)
                # mocap.eval()
                # online_results = [mocap.forward_frame(f) for f in torch.cat((x, x[-1].repeat(model_config.future_frames, 1)))]
                # pose_p = torch.stack(online_results[model_config.future_frames:], dim=0)

                # torch.save({
                #     'pose_p': pose_p,
                #     'rot_p': rot,
                #     'use_cali': use_cali if args.use_cali else None,
                # }, os.path.join(data_root, f'{type}_pred.pt'))
                
            # ===== 绘图部分 =====
            for key in ['watch', 'phone']:
                plt.figure(figsize=(10, 4))
                plt.title(f"{f} — {key} Rotation Error Comparison", fontsize=14)
                plt.xlabel("Frame Index", fontsize=12)
                plt.ylabel("Rotation Error (°)", fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.5)

                for type in ['w/o_cali', 'ours(real)', 'baseline(syn)']:
                    if file_errors[type][key] is not None:
                        smoothed = smooth_curve(file_errors[type][key], window_size=15)
                        plt.plot(smoothed, label=type, linewidth=1.8)

                plt.ylim(0, 90)
                plt.legend()
                plt.tight_layout()

                save_path = os.path.join(data_root, f"{key}_compare_types.png")
                plt.savefig(save_path, dpi=200)
                plt.close()

    # 所有文件完成后，计算类型平均
    print("\n=====Overall Mean Rotation Error Across Files=====")
    for t in global_errors:
        for key in ['watch', 'phone']:
            if global_errors[t][key]:
                avg_err = np.mean(global_errors[t][key])
                print(f"{t} - {key}: {avg_err:.2f}°")