from pygame.time import Clock
import articulate as art
from articulate.utils.unity import MotionViewer
from utils import *
from config import paths
import torch
import os
import open3d as o3d
import numpy as np

body_model = art.ParametricModel(paths.smpl_file)
vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021])

deta_dir = 'data/debug/amass'

def visualize(pose, tran=None, rheight=None, joint=None, names=None):
    '''
    pose: [pose_sub1, pose_sub2, ...]
    tran: [tran_sub1, tran_sub2, ...]
    '''
    clock = Clock()
    sub_num = len(pose)
    with MotionViewer(sub_num, overlap=True, names=names) as viewer:
        for i in range(len(pose[0])):
            clock.tick(60)
            viewer.clear_line(render=False)
            viewer.clear_point(render=False)
            
            pose_list = [pose[sub_idx][i] for sub_idx in range(sub_num)]
            if tran:
                tran_list = [tran[sub_idx][i] for sub_idx in range(sub_num)]
            else:
                # generate zero tran on cpu
                tran_list = [torch.zeros(1, 3).cpu() for _ in range(sub_num)]
            
            viewer.update_all(pose_list, tran_list, render=False)
            
            viewer.render()
            
            print('\r', clock.get_fps(), end='')

def process_mobileposer_data(data, relative_height = False, joint = False):
    pose_t = data['pose_t']
    pose_p = data['pose_p']
    
    tran_t = data['tran_t']
    tran_p = data['tran_p']
    
    pose_t = pose_t.view(-1, 24, 3, 3)
    pose_p = pose_p.view(-1, 24, 3, 3)
    
    # convert to cpu
    pose_t = pose_t.cpu()
    pose_p = pose_p.cpu()
    tran_t = tran_t.cpu()
    tran_p = tran_p.cpu()
    
    # 将tran_t和tran_p的第一帧align到一起
    tran_t = tran_t - tran_t[0] + tran_p[0]
    
    return pose_t, pose_p, tran_t, tran_p

if __name__ == '__main__':
    
    data_dir = 'data/dataset_work/eval/30fps'
    dataset_name = 'dip_test'
    
    data_path = os.path.join(data_dir, dataset_name + '.pt')
    data = torch.load(data_path)
    
    pose = data['pose']
    tran = data['tran']

    idx = [0]
    
    for i in idx:
        
        # print idx and sequence length
        print("idx, seq_len: ", i, len(pose[i]))
        
        pose_list = []
        tran_list = []

        pose_list.append(pose[i])
        tran_list.append(tran[i])
        
        name_list = ['gt']
            
        visualize(pose_list, names=name_list)