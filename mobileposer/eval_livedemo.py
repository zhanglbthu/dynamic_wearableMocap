import articulate as art
from config import *
from my_model import *
from TicOperator import *
from evaluation_functions import *
from argparse import ArgumentParser
from utils.model_utils import load_mobileposer_model

import os

imu_num = config.imu_num
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = TIC(stack=3, n_input=imu_num * (3 + 3 * 3), n_output=imu_num * 6)
# model.restore('data/checkpoints/calibrator/TIC_MP/TIC_20.pth')
model = LSTMIC(n_input=imu_num * (3 + 3 * 3), n_output=imu_num * 6)
model.restore('data/checkpoints/calibrator/LSTMIC_realdata/LSTMIC_realdata_20.pth')
model = model.to(device).eval()

tag = 'RealData'
folders = ['sit_20250811_180140']
# folders = ['s1']

# Inference
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='data/checkpoints/mobileposer/lw_rp/base_model.pth')
    parser.add_argument('--use_cali', action='store_true')
    args = parser.parse_args()
    types = ['raw', 'gt', 'cali']
    print('=====Inference Start=====')
    for f in folders:
        data_root = os.path.join(config.paths.livedemo_dataset_dir, f)
        os.makedirs(data_root, exist_ok=True)
        print(f'processing {f}')

        data = torch.load(os.path.join(config.paths.livedemo_dataset_dir, f +'.pt'))
        
        imu_acc = data['acc'].to(device)
        imu_rot = data['ori'].to(device)
        gt_acc  = data['acc_gt'].to(device)
        gt_rot  = data['ori_gt'].to(device)
        
        for type in types:

            # ts = TicOperator(TIC_network=model, imu_num=imu_num, data_frame_rate=30)
            # rot, acc, pred_offset, use_cali = ts.run_per_frame(imu_rot, imu_acc)
            if type == 'raw':
                rot = imu_rot
                acc = imu_acc
            elif type == 'gt':
                rot = gt_rot
                acc = gt_acc
            elif type == 'cali':
                ts = TicOperator(TIC_network=model, imu_num=imu_num, data_frame_rate=30)
                rot, acc, pred_offset, use_cali = ts.run_per_frame(imu_rot, imu_acc)
            else:
                raise "Unknown type"
            use_cali = None

            acc = acc / amass.acc_scale
            
            x = torch.cat((acc.flatten(1), rot.flatten(1)), dim=1).to(device)
            mocap = load_mobileposer_model(model_path=args.model, combo_id=model_config.combo_id)
            mocap.eval()
            online_results = [mocap.forward_frame(f) for f in torch.cat((x, x[-1].repeat(model_config.future_frames, 1)))]
            pose_p = torch.stack(online_results[model_config.future_frames:], dim=0)

            torch.save({
                'pose_p': pose_p,
                'use_cali': use_cali if args.use_cali else None,
            }, os.path.join(data_root, f'{tag}_{type}_pred.pt'))