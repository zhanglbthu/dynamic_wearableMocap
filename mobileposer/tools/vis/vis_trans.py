import torch
import os
import tqdm
import sys

pred_folder = "data/result/TotalCapture/test"
target_path = "data/dataset_work/TotalCapture/test.pt"

# 列出pred_folder下所有文件
pred_files = os.listdir(pred_folder)
target = torch.load(target_path)

pose_t_all, tran_t_all = target['pose'], target['tran']

for i in tqdm.tqdm(range(len(pred_files))):
        result = torch.load(os.path.join(pred_folder, '%d.pt' % i))
        pose_p, tran_p = result[0], result[1]
        pose_t, tran_t = pose_t_all[i], tran_t_all[i]
        print("pose_p shape: ", pose_p.shape)
        print("tran_p shape: ", tran_p.shape)
        print("pose_t shape: ", pose_t.shape)
        print("tran_t shape: ", tran_t.shape)
        
        sys.exit(0)