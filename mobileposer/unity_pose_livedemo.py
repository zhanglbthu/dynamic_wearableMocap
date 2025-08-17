from pygame.time import Clock
import articulate as art
from articulate.utils.unity import MotionViewer
from utils import *
from config import paths, joint_set, imu_num
import torch
import os
import open3d as o3d
import numpy as np
import matplotlib
from articulate.utils.bullet.view_rotation_np import RotationViewer

body_model = art.ParametricModel(paths.smpl_file)
vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021])
colors = matplotlib.colormaps['tab10'].colors

deta_dir = 'data/debug/amass'

def value2color(value):
    value = torch.clamp(value, 0, 1).cpu().numpy()
    color = np.array([1, 1 - value, 1 - value])
    return color

class MotionViewerManager:
    def __init__(self, sub_num, overlap=True, names=None):
        self.viewer = MotionViewer(sub_num, overlap, names)
        self.viewer.connect()     

    def visualize(self, pose, tran=None, use_cali_list=None):
        clock = Clock()
        sub_num = len(pose)

        thigh_trigger = False
        thigh_idx = 0
        wrist_trigger = False
        wrist_idx = 0
        time = 20

        for i in range(len(pose[0])):
            clock.tick(30)
            self.viewer.clear_line(render=False)
            self.viewer.clear_point(render=False)
            self.viewer.clear_terrian(render=False)

            pose_list = [pose[sub][i] for sub in range(sub_num)]
            tran_list = [tran[sub][i] for sub in range(sub_num)] if tran else [torch.zeros(3) for _ in range(sub_num)]
            
            if use_cali_list is not None:
                for sub in range(1, sub_num):
                    _, _, glb_vert = body_model.forward_kinematics(pose_list[sub].unsqueeze(0), tran=tran_list[sub].unsqueeze(0), calc_mesh=True)
                    wrist_pos = glb_vert[0][vi_mask[0], :]
                    thigh_pos = glb_vert[0][vi_mask[3], :]
                    wrist_pos = wrist_pos.cpu().numpy() + self.viewer.offsets[sub]
                    thigh_pos = thigh_pos.cpu().numpy() + self.viewer.offsets[sub]

                    use_cali = use_cali_list[sub-1][i]
                    if use_cali[0]:
                        wrist_trigger = True
                        wrist_idx = 0
                        # self.viewer.draw_point(wrist_pos, color=[1, 0, 0], radius=0.2, render=False)
                    if use_cali[1]:
                        thigh_trigger = True
                        thigh_idx = 0
                        # self.viewer.draw_point(thigh_pos, color=[1, 0, 0], radius=0.2, render=False)
            
            # TODO: change point visualization, it's support tic in the list's last now
            if wrist_trigger:
                wrist_idx += 1
                self.viewer.draw_point(wrist_pos, color=[1, 0, 0], radius=0.2, render=False)
                if wrist_idx > time:
                    wrist_trigger = False
                    wrist_idx = 0    
            if thigh_trigger:
                thigh_idx += 1
                self.viewer.draw_point(thigh_pos, color=[1, 0, 0], radius=0.2, render=False)
                if thigh_idx > time:
                    thigh_trigger = False
                    thigh_idx = 0

            self.viewer.update_all(pose_list, tran_list, render=False)
            self.viewer.render()
            print('\r', clock.get_fps(), end='')

    def close(self):
        self.viewer.disconnect()

def process_full_data(data):    
        
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
    tran_p = tran_p - tran_p[0] + tran_t[0]
    
    return pose_t, pose_p, tran_t, tran_p

def process_pose_data(data):
    pose_t = data['pose_t']
    pose_p = data['pose_p']
    use_cali = data['use_cali'] if 'use_cali' in data else torch.zeros(pose_t.shape[0], imu_num, dtype=torch.bool)
    
    # if use_cali is None
    if use_cali is None:
        use_cali = torch.zeros(pose_t.shape[0], imu_num, dtype=torch.bool)
    
    pose_t = pose_t.view(-1, 24, 3, 3)
    pose_p = pose_p.view(-1, 24, 3, 3)
    use_cali = use_cali.view(-1, imu_num)

    # convert to cpu
    pose_t = pose_t.cpu()
    pose_p = pose_p.cpu()
    use_cali = use_cali.cpu()

    return pose_t, pose_p, use_cali

def process_livedemo_data(data):
    pose_p = data['pose_p'] if 'pose_p' in data else data['pose']
    use_cali = data['use_cali'] if 'use_cali' in data else torch.zeros(pose_p.shape[0], imu_num, dtype=torch.bool)
    
    # if use_cali is None
    if use_cali is None:
        use_cali = torch.zeros(pose_p.shape[0], imu_num, dtype=torch.bool)

    pose_p = pose_p.view(-1, 24, 3, 3)
    use_cali = use_cali.view(-1, imu_num)

    # convert to cpu
    pose_p = pose_p.cpu()
    use_cali = use_cali.cpu()

    return pose_p, use_cali

def get_name(model_list=["ours", "baseline"], i=0):
    name_list = ['gt'] + model_list
    name_list = [name.replace('_60fps', '') for name in name_list]
    name_list = [name + "_" + str(i) for name in name_list]
    
    return name_list

if __name__ == '__main__':
    
    data_dir = 'data/livedemo/sit_20250811'
    model_list = ['sit_20250811_180140_gt.pt', 'sit_20250811_180140.pt', 'sit_20250811_180140_tic.pt']

    # # 获取data_dir/model_list[0]/dataset_name/中以.pt结尾的文件个数
    # idx_num = len([name for name in os.listdir(os.path.join(data_dir, model_list[0], 'lw_rp', dataset_name)) if name.endswith('.pt')])
        
    pose_list, tran_list, use_cali_list = [], [], []
    for name in model_list:
        data_path = os.path.join(data_dir, name)
        data = torch.load(data_path)

        pose_p, use_cali = process_livedemo_data(data)
        use_cali_list.append(use_cali)
            
        pose_list.append(pose_p)
        
    name_list = ['gt_rot', 'real_rot', 'tic_rot']

    # rviewer = RotationViewer(3, order='wxyz'); rviewer.connect()
    # data_path = os.path.join(data_dir, model_list[1])
    # data = torch.load(data_path)
    # rot = data['rot']
    # rot_gt = data['rot_gt']
    
    viewer_manager = MotionViewerManager(len(pose_list), overlap=False, names=name_list)

    viewer_manager.visualize(pose_list, use_cali_list=None)
    viewer_manager.close()