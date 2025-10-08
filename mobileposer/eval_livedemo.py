import articulate as art
from config import *
from my_model import *
from TicOperator import *
from evaluation_functions import *
from argparse import ArgumentParser
from utils.model_utils import load_mobileposer_model, load_heightposer_model

import os

imu_num = config.imu_num
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_tic = TIC(stack=3, n_input=imu_num * (3 + 3 * 3), n_output=imu_num * 6)
model_tic.restore('data/checkpoints/calibrator/TIC_MP/TIC_20.pth')

model_lstm = LSTMIC(n_input=imu_num * (3 + 3 * 3), n_output=imu_num * 6)
model_lstm.restore('data/checkpoints/calibrator/RealData_0914_2pants/RealData_0914_2pants_20.pth')

model_tic = model_tic.to(device).eval()
model_lstm = model_lstm.to(device).eval()

# folders = ['sit_20250811_180140']
eval_dir = "/root/autodl-tmp/data/Real_Dataset/RealData_Raw_0915_types/pants_6"
# folders = os.listdir(eval_dir)
folders = [f for f in os.listdir(eval_dir) if f.endswith('.pt')]
print(f'Found folders: {folders}')
# folders = ['s1']

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

# Inference
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='data/checkpoints/mobileposer/lw_rp/base_model.pth')
    parser.add_argument('--use_cali', action='store_true')
    args = parser.parse_args()
    types = ['gt', 'raw', 'lstm', 'tic']
    print('=====Inference Start=====')
    for f in folders:
        f = f.split('.')[0]
        data_root = os.path.join(eval_dir, f)
        os.makedirs(data_root, exist_ok=True)

        data = torch.load(os.path.join(eval_dir, f +'.pt'))
        
        imu_acc = data['acc'].to(device)[100:-100]
        imu_rot = data['ori'].to(device)[100:-100]
        gt_acc  = data['acc_gt'].to(device)[100:-100]
        gt_rot  = data['ori_gt'].to(device)[100:-100]
        print(f'Processing {f}')
        for type in types:
            if type == 'raw':
                rot = imu_rot
                acc = imu_acc
            elif type == 'gt':
                rot = gt_rot
                acc = gt_acc
            elif type == 'lstm':
                ts = TicOperator(TIC_network=model_lstm, imu_num=imu_num, data_frame_rate=30)
                rot, acc, pred_offset, use_cali = ts.run_per_frame(imu_rot, imu_acc)
            elif type == 'tic':
                ts = TicOperator(TIC_network=model_tic, imu_num=imu_num, data_frame_rate=30)
                rot, acc, _, _, _ = ts.run(imu_rot, imu_acc)
            else:
                raise "Unknown type"
            use_cali = None
            # only evaluate error if not gt
            if type != 'gt' and not f.startswith('live'):
                errors = {'watch': [], 'phone': []}
                for R_pred, R_gt in zip(rot, gt_rot):
                    # errors.append(rotation_matrix_error(R_pred, R_gt))
                    errors['watch'].append(rotation_matrix_error(R_pred[0], R_gt[0]))
                    errors['phone'].append(rotation_matrix_error(R_pred[1], R_gt[1]))
                for key in errors:
                    errors[key] = np.array(errors[key])
                    mean_error = np.mean(errors[key]) * 180/np.pi
                    print(f"Mean rotation error for {key} ({type}): {mean_error:.2f}°", end='; ')
                print()
            
            acc = acc / amass.acc_scale
            
            x = torch.cat((acc.flatten(1), rot.flatten(1)), dim=1).to(device)
            # mocap = load_mobileposer_model(model_path=args.model, combo_id=model_config.combo_id)
            mocap = load_mobileposer_model(model_path=args.model, combo_id=model_config.combo_id)
            mocap.eval()
            online_results = [mocap.forward_frame(f) for f in torch.cat((x, x[-1].repeat(model_config.future_frames, 1)))]
            pose_p = torch.stack(online_results[model_config.future_frames:], dim=0)

            torch.save({
                'pose_p': pose_p,
                'rot_p': rot,
                'use_cali': use_cali if args.use_cali else None,
            }, os.path.join(data_root, f'{type}_pred.pt'))