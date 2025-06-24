from pygame.time import Clock
import articulate as art
from articulate.utils.unity import MotionViewer
from utils import *
from config import paths
import torch
import os
import open3d as o3d
import numpy as np
from utils.data_utils import _foot_min, _foot_contact_either, _foot_contact_both, _foot_ground_probs, _get_ground

body_model = art.ParametricModel(paths.smpl_file)
vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021])

deta_dir = 'data/debug/amass'

def value2color(value):
    value = torch.clamp(value, 0, 1).cpu().numpy()
    color = np.array([1, 1 - value, 1 - value])
    return color

class MotionViewerManager:
    def __init__(self, sub_num, overlap=True, names=None):
        self.viewer = MotionViewer(sub_num, overlap, names)
        self.viewer.connect()
        self.ground = None

    def plot_ground(self, f_pos, ground, color=[255, 255, 255]):
        grid_size = 10 
        sample_range = 1.0
        
        # 中心点的 x, z 坐标
        x_center = (f_pos[0][0] + f_pos[1][0]) / 2
        z_center = (f_pos[0][2] + f_pos[1][2]) / 2
        y = ground.item()  # y 坐标使用地面的值
        
        if self.ground is None:
            self.ground = y

        floor_color = color
        
        # 计算网格线的间距
        step = 2 * sample_range / (grid_size - 1)  # 网格线间隔

        # 绘制平行于 z 轴的网格线
        for i in range(grid_size):
            x = x_center - sample_range + i * step  # 当前网格线的 x 坐标
            z_start = z_center - sample_range  # 网格线起点
            z_end = z_center + sample_range  # 网格线终点
            
            # 绘制一条线
            self.viewer.draw_line([x, y, z_start], [x, y, z_end], color=floor_color, render=False)

        # 绘制平行于 x 轴的网格线
        for i in range(grid_size):
            z = z_center - sample_range + i * step  # 当前网格线的 z 坐标
            x_start = x_center - sample_range  # 网格线起点
            x_end = x_center + sample_range  # 网格线终点
            
            # 绘制一条线
            self.viewer.draw_line([x_start, y, z], [x_end, y, z], color=floor_color, render=False)        
    
    def plot_contact(self, pose_list, tran_list, contact_list):
        for i in range(len(pose_list)):
            pose = torch.tensor(pose_list[i]).unsqueeze(0)
            tran = torch.tensor(tran_list[i]).unsqueeze(0)
            
            _, glb_joint = body_model.forward_kinematics(pose, tran=tran, calc_mesh=False)
            
            self.viewer.draw_point(glb_joint[0][10], color=value2color(contact_list[i][0]), radius=0.2, render=False)
            self.viewer.draw_point(glb_joint[0][11], color=value2color(contact_list[i][1]), radius=0.2, render=False)
    
    def visualize(self, pose, tran=None, contact = None, f_pos=None, ground=None):
        clock = Clock()
        sub_num = len(pose)
        # floor_color = [blue, orange, green, red]
        floor_color = [[0, 0, 255], [255, 165, 0], [0, 128, 0], [255, 0, 0]]
        
        for i in range(len(pose[0])):
            clock.tick(60)
            self.viewer.clear_line(render=False)
            self.viewer.clear_point(render=False)

            pose_list = [pose[sub][i] for sub in range(sub_num)]
            tran_list = [tran[sub][i] for sub in range(sub_num)] if tran else [torch.zeros(3) for _ in range(sub_num)]

            if ground is not None:
                for sub in range(sub_num):
                    self.plot_ground(f_pos[sub][i], ground[sub][i], color=floor_color[sub])
                
            if contact is not None:
                contact_list = [contact[sub][i] for sub in range(sub_num)]
                self.plot_contact(pose_list, tran_list, contact_list)
            
            self.viewer.update_all(pose_list, tran_list, render=False)
            self.viewer.render()
            print('\r', clock.get_fps(), end='')

    def close(self):
        self.viewer.disconnect()

def process_mobileposer_data(data, relative_height = False, joint = False, contact = False):
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
    
    if joint:
        joint_t = data['joint_t'].cpu()
        joint_p = data['joint_p'].cpu()
        
        return pose_t, pose_p, tran_t, tran_p, joint_t, joint_p
    
    if contact:
        contact_t = data['contact_t'].cpu()
        contact_p = data['contact_p'].cpu()
        
        return pose_t, pose_p, tran_t, tran_p, contact_t, contact_p
    
    return pose_t, pose_p, tran_t, tran_p

def edit_height(pose_t, tran_t, pose_p, tran_p, contact_p):
    _, glb_joint, glb_vert = body_model.forward_kinematics(pose_t, tran=tran_t, calc_mesh=True)
    
    foot_min = _foot_min(glb_joint)
    init_ground = foot_min[0].item()
    cur_ground = init_ground
    
    init_tran_y = tran_t[0][1].item()
    cur_tran_y = init_tran_y
    
    pocket_height = glb_vert[:, vi_mask[3], 1].unsqueeze(1) - init_ground
    init_height = pocket_height[0].item()
    cur_height = init_height
    
    out_tran_y = [init_tran_y]
    out_ground = [init_ground]
    
    contact_num = 5
        
    for frame in range(1, len(pose_p)):
        h = pocket_height[frame].item()
        r = h - cur_height
        
        if frame >= contact_num:
            fp_last_n = contact_p[frame + 1 - contact_num: frame + 1]
        else:
            fp_last_n = contact_p[:frame+1]
        
        cur_tran_y = cur_tran_y + r
        out_tran_y.append(cur_tran_y)
        
        cur_height = h
        out_ground.append(cur_ground)
                                 
    return out_ground, out_tran_y

def get_gt_floor(pose, tran):
    _, joint = body_model.forward_kinematics(pose, tran=tran, calc_mesh=False)
    fc_probs = _foot_ground_probs(joint).clone()
    ground = _get_ground(joint, fc_probs, len(joint), contact_num=5)
    
    return ground

def get_name(model_list, i):
    name_list = ['gt'] + model_list
    name_list = [name.replace('_60fps', '') for name in name_list]
    name_list = [name + "_" + str(i) for name in name_list]
    
    return name_list

def get_foot_pos(pose, tran):
    _, joint = body_model.forward_kinematics(pose, tran=tran, calc_mesh=False)
    lfoot = joint[:, 10]
    rfoot = joint[:, 11]
    
    return torch.stack((lfoot, rfoot), dim=1)

if __name__ == '__main__':
    
    data_dir = 'data/eval'
    dataset_name = 'cmu'
    model_list = ['heights_60fps']
    
    # 获取data_dir/model_list[0]/dataset_name/中以.pt结尾的文件个数
    idx_num = len([name for name in os.listdir(os.path.join(data_dir, model_list[0], 'lw_rp_h', dataset_name)) if name.endswith('.pt')])

    idx = [i for i in range(4, idx_num)]
    print('len:', idx)
    
    for i in idx:
        
        pose_list, tran_list, contact_list, ground_list, fp_list = [], [], [], [], []
        
        for name in model_list:
            data_path = os.path.join(data_dir, name, 'lw_rp_h', dataset_name, str(i)+'.pt')
            data = torch.load(data_path)
            
            pose_t, pose_p, tran_t, tran_p, contact_t, contact_p = process_mobileposer_data(data, contact=True)
            if pose_list == []:
                pose_list.append(pose_t), tran_list.append(tran_t), contact_list.append(contact_t)
                fp_list.append(get_foot_pos(pose_t, tran_t))
                ground_list.append(get_gt_floor(pose_t, tran_t))
            
            pose_list.append(pose_p), tran_list.append(tran_p), contact_list.append(contact_p)
        
        name_list = get_name(model_list, i)
        
        for i in range(1, len(pose_list)):
            ground, tran_y = edit_height(pose_list[0], tran_list[0], pose_list[i], tran_list[i], contact_list[i])
            tran_list[i][:, 1] = torch.tensor(tran_y)
            fp_list.append(get_foot_pos(pose_list[i], tran_list[i]))
            ground_list.append(torch.tensor(ground))
        
        viewer_manager = MotionViewerManager(len(pose_list), overlap=True, names=name_list)
        viewer_manager.visualize(pose_list, tran=tran_list, contact=contact_list, ground=ground_list, f_pos=fp_list)
        viewer_manager.close()