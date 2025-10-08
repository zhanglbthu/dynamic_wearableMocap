import os
import numpy as np
import torch
from argparse import ArgumentParser
from tqdm import tqdm

from config import *
import articulate as art
from utils.model_utils import load_imuposer_model, load_imuposer_glb_model, load_mobileposer_model, load_heightposer_model
from data import PoseDataset
from pathlib import Path
from utils.file_utils import (
    get_best_checkpoint
)
from config import model_config
from articulate.model import ParametricModel

from TicOperator import *
from my_model import *

body_model = ParametricModel(paths.smpl_file, device=model_config.device)

class PoseEvaluator:
    def __init__(self):
        self._eval_fn = art.FullMotionEvaluator(paths.smpl_file, joint_mask=torch.tensor([1, 2, 16, 17]), fps=datasets.fps)

    def eval(self, pose_p, pose_t, joint_p=None, tran_p=None, tran_t=None):
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        if tran_p is not None and tran_t is not None:
            tran_p = tran_p.clone().view(-1, 3)
            tran_t = tran_t.clone().view(-1, 3)
        pose_p[:, joint_set.ignored] = torch.eye(3, device=pose_p.device)
        pose_t[:, joint_set.ignored] = torch.eye(3, device=pose_t.device)

        errs = self._eval_fn(pose_p, pose_t, tran_p=tran_p, tran_t=tran_t)
        return torch.stack([errs[9], errs[3], errs[9], errs[0]*100, errs[7]*100, errs[1]*100, errs[4] / 100, errs[6]])

    @staticmethod
    def print(errors):
        for i, name in enumerate(['SIP Error (deg)', 'Angular Error (deg)', 'Masked Angular Error (deg)',
                                  'Positional Error (cm)', 'Masked Positional Error (cm)', 'Mesh Error (cm)', 
                                  'Jitter Error (100m/s^3)', 'Distance Error (cm)']):
            print('%s: %.2f (+/- %.2f)' % (name, errors[i, 0], errors[i, 1]))
            
    @staticmethod
    def print_single(errors, file=None):
        metric_names = [
            'SIP Error (deg)', 
            'Angular Error (deg)', 
            'Masked Angular Error (deg)',
            'Positional Error (cm)', 
            'Masked Positional Error (cm)', 
            'Mesh Error (cm)', 
            'Jitter Error (100m/s^3)', 
            'Distance Error (cm)'
        ]
        
        # 找出最长的指标名，以便统一对齐
        max_len = max(len(name) for name in metric_names)

        # 将每个指标的输出字符串保存到列表中，最后 join 成一行输出
        output_parts = []
        for i, name in enumerate(metric_names):
            if name in ['Angular Error (deg)', 'Mesh Error (cm)']:
                # 对这类指标使用“均值 ± 方差”的格式
                output_str = f"{name:<{max_len}}: {errors[i,0]:.2f}"
            else:
                continue
            
            output_parts.append(output_str)

        # 最终打印为一行
        print(" | ".join(output_parts), file=file)
        # 如果需要在末尾换行，print 本身就会换行，无需额外操作

@torch.no_grad()
def evaluate_pose(model, dataset, calibrator:TicOperator, save_dir=None, use_cali=False):
    # specify device
    device = model_config.device

    # load data
    xs, ys = zip(*[(imu.to(device), (pose.to(device), tran)) for imu, pose, joint, tran in dataset])

    # setup Pose Evaluator
    evaluator = PoseEvaluator()

    # track errors
    online_errs = []
    
    model.eval()
      
    with torch.no_grad():
        for idx, (x, y) in enumerate(tqdm(zip(xs, ys), total=len(xs))):
            if use_cali:
                calibrator.reset()
                acc = x[:, :imu_num * 3] * amass.acc_scale
                rot = x[:, imu_num * 3:]
                
                rot_cali, acc_cali, _, _, use_calis = calibrator.run(rot, acc)
                acc_cali = acc_cali / amass.acc_scale
                
                x = torch.cat((acc_cali.flatten(1), rot_cali.flatten(1)), dim=1)
                
            model.reset()

            pose_t, _ = y
            
            pose_t = art.math.r6d_to_rotation_matrix(pose_t)
            pose_t = pose_t.view(-1, 24, 3, 3)  
            
            if model_config.winit:
                pose_p = model.predict(x, pose_t[0], poser_only=True)
            else:
                online_results = [model.forward_frame(f) for f in torch.cat((x, x[-1].repeat(model_config.future_frames, 1)))]
                pose_p = torch.stack(online_results[model_config.future_frames:], dim=0)
            
            online_errs.append(evaluator.eval(pose_p, pose_t))
            
            if save_dir:
                torch.save({'pose_t': pose_t, 
                            'pose_p': pose_p,
                            'use_cali': use_calis if use_cali else None,
                            },
                           save_dir / f"{idx}.pt")

    evaluator.print(torch.stack(online_errs).mean(dim=0))
        
    log_path = save_dir / 'log.txt'
        
    # 清空原有内容
    with open(log_path, 'w', encoding='utf-8') as f:
        pass
        
    for online_err in online_errs:
        with open(log_path, 'a', encoding='utf-8') as f:
            evaluator.print_single(online_err, file=f)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='data/checkpoints/mobileposer/lw_rp/base_model.pth')
    parser.add_argument('--dataset', type=str, default='dip')
    parser.add_argument('--use_cali', action='store_true')
    args = parser.parse_args()
    device = torch.device("cuda")

    # load mocap model
    model_name = model_config.name.split('_')[0]
    if model_name == 'imuposer':
        model = load_imuposer_model(model_path=args.model, combo_id=model_config.combo_id)
    elif model_name == 'mobileposer':
        model = load_mobileposer_model(model_path=args.model, combo_id=model_config.combo_id)
    elif model_name == 'heightposer':
        model = load_heightposer_model(model_path=args.model, combo_id=model_config.combo_id)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    
    # load calibrator model
    tic = TIC(stack=3, n_input=imu_num * (3 + 3 * 3), n_output=imu_num * 6)
    tic.restore("data/checkpoints/calibrator/TIC_MP/TIC_20.pth")
    
    net = tic.to(device).eval()
        
    ts = TicOperator(TIC_network=net, imu_num=imu_num, data_frame_rate=30)
    ts.reset()
        
    fold = 'test'
        
    dataset = PoseDataset(fold=fold, evaluate=args.dataset)
        
    save_dir = Path('data') / 'eval' / model_config.combo_id / args.dataset

    save_dir.mkdir(parents=True, exist_ok=True)
        
    # evaluate_pose(model, dataset, evaluate_tran=True, save_dir=save_dir)
    evaluate_pose(model=model, dataset=dataset, calibrator=ts, save_dir=save_dir, use_cali=args.use_cali)