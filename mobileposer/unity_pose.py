from pygame.time import Clock
import articulate as art
from articulate.utils.unity import MotionViewer
from utils import *
from config import paths, joint_set
import torch
import os
import open3d as o3d
import numpy as np
import matplotlib
from utils.data_utils import _foot_min, _foot_contact_either, _foot_contact_both, _foot_ground_probs, _get_ground

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
        self.ground = None
        self.init_ground = None       
    
    def plot_terrian(self, f_pos, ground, color=[255, 255, 255], width=1, offset = False):
        
        x_center = (f_pos[0][0] + f_pos[1][0]) / 2
        z_center = (f_pos[0][2] + f_pos[1][2]) / 2
        
        start_y = self.init_ground.clone()
        
        if offset:
            start_y -= 1e-2
        
        end_y = ground.item()
        
        start = [x_center, start_y, z_center]
        end = [x_center, end_y, z_center]
        
        self.viewer.draw_terrian(start, end, color=color, width=width, render=False)
        
    def plot_contact(self, pose_list, tran_list, contact_list):
        for i in range(len(pose_list)):
            pose = torch.tensor(pose_list[i]).unsqueeze(0)
            tran = torch.tensor(tran_list[i]).unsqueeze(0)
            
            _, glb_joint = body_model.forward_kinematics(pose, tran=tran, calc_mesh=False)
            
            self.viewer.draw_point(glb_joint[0][10], color=value2color(contact_list[i][0]), radius=0.2, render=False)
            self.viewer.draw_point(glb_joint[0][11], color=value2color(contact_list[i][1]), radius=0.2, render=False)
    
    def visualize(self, pose, tran=None, contact = None, f_pos=None, ground=None, sensor_point=True, flag=None):
        clock = Clock()
        sub_num = len(pose)
        
        for i in range(len(pose[0])):
            clock.tick(30)
            self.viewer.clear_line(render=False)
            self.viewer.clear_point(render=False)
            self.viewer.clear_terrian(render=False)

            pose_list = [pose[sub][i] for sub in range(sub_num)]
            tran_list = [tran[sub][i] for sub in range(sub_num)] if tran else [torch.zeros(3) for _ in range(sub_num)]

            if ground is not None:
                
                # # 绘制初始化地面
                self.plot_terrian(f_pos[0][i], self.init_ground - 1, color=[0.94, 0.78, 0.52], width=100, offset=True)
                
                for sub in range(sub_num):
                    self.plot_terrian(f_pos[sub][i], ground[sub][i], color=colors[sub], width=1)
                
            if contact is not None:
                contact_list = [contact[sub][i] for sub in range(sub_num)]
                self.plot_contact(pose_list, tran_list, contact_list)
                
            if sensor_point:
                gt_thigh_height = None
                gt_wrist_height = None
                for sub in range(sub_num):
                    _, glb_joint, glb_vert = body_model.forward_kinematics(pose_list[sub].unsqueeze(0), tran=tran_list[sub].unsqueeze(0), calc_mesh=True)
                    wrist_pos = glb_vert[0][vi_mask[0], :]
                    thigh_pos = glb_vert[0][vi_mask[3], :]
                    wrist_pos = wrist_pos.cpu().numpy() + self.viewer.offsets[sub]
                    thigh_pos = thigh_pos.cpu().numpy() + self.viewer.offsets[sub]
                    
                    # self.viewer.draw_point(wrist_pos, color=[1, 0, 0], radius=0.05, render=False)
                    # self.viewer.draw_point(thigh_pos, color=[1, 0, 0], radius=0.05, render=False)
                    
                    if gt_thigh_height is None and gt_wrist_height is None:
                        gt_ground = _foot_min(glb_joint).view(-1)
                        gt_thigh_height = glb_vert[0][vi_mask[3], 1] - gt_ground
                        gt_wrist_height = glb_vert[0][vi_mask[0], 1] - gt_ground
                        self.viewer.draw_point(wrist_pos, color=[1, 0, 0], radius=0.05, render=False)
                        self.viewer.draw_point(thigh_pos, color=[1, 0, 0], radius=0.05, render=False)
                    else:
                        # gt_wrist_pos = wrist_pos.copy()
                        # gt_wrist_pos[1] = thigh_pos[1] + gt_rh
                        # self.viewer.draw_line(wrist_pos, gt_wrist_pos, color=[1, 0, 0], width=0.02, render=False)
                        # self.viewer.draw_point(gt_wrist_pos, color=[0, 1, 0], radius=0.05, render=False)
                        cur_ground = _foot_min(glb_joint).view(-1)
                        offset = 0.5
                        thigh_pos[2] += offset
                        wrist_pos[2] += offset
                        gt_thigh_pos = thigh_pos.copy()
                        gt_wrist_pos = wrist_pos.copy()
                        gt_thigh_pos[1] = cur_ground + gt_thigh_height
                        gt_wrist_pos[1] = cur_ground + gt_wrist_height
                        self.viewer.draw_line(thigh_pos, gt_thigh_pos, color=[1, 0, 0], width=0.02, render=False)
                        self.viewer.draw_line(wrist_pos, gt_wrist_pos, color=[1, 0, 0], width=0.02, render=False)
                        self.viewer.draw_point(wrist_pos, color=[1, 0, 0], radius=0.05, render=False)
                        self.viewer.draw_point(thigh_pos, color=[1, 0, 0], radius=0.05, render=False)
                        self.viewer.draw_point(gt_thigh_pos, color=[0, 1, 0], radius=0.05, render=False)
                        self.viewer.draw_point(gt_wrist_pos, color=[0, 1, 0], radius=0.05, render=False)
            
            if flag is not None:
                if flag[i]:
                    self.viewer.draw_point([0, 0, 0], color=[1, 0, 0], radius=0.2, render=False)
                else:
                    self.viewer.draw_point([0, 0, 0], color=[0, 1, 0], radius=0.2, render=False)
            
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
    
    pose_t = pose_t.view(-1, 24, 3, 3)
    pose_p = pose_p.view(-1, 24, 3, 3)

    # convert to cpu
    pose_t = pose_t.cpu()
    pose_p = pose_p.cpu()
    
    return pose_t, pose_p

def process_imuposer_data(data):
    # data to cpu
    pose = data.cpu()
    
    return pose

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
        
    for frame in range(1, len(pose_p)):
        # h = pocket_height[frame].item()
        # r = h - cur_height
        
        # cur_tran_y = cur_tran_y + r
        # out_tran_y.append(cur_tran_y)
        
        # cur_height = h
        # TODO: refine translation estimation
        _, local_joint, local_vert = body_model.forward_kinematics(pose_p[frame].unsqueeze(0), calc_mesh=True)
        
        h = pocket_height[frame].item()
        local_h = local_vert[0, vi_mask[3], 1].item()
        
        local_r = local_joint[0, 0, 1]
        
        cur_tran_y = local_r + h - local_h + init_ground
        out_tran_y.append(cur_tran_y)
        
        
        # TODO: refine ground estimation
        cur_tran = tran_p[frame]
        cur_tran[1] = cur_tran_y
        _, joint = body_model.forward_kinematics(pose_p[frame].unsqueeze(0), tran=cur_tran.unsqueeze(0), calc_mesh=False)
        cur_ground = _foot_min(joint)[0].item()
        
        out_ground.append(cur_ground)
                                 
    return out_ground, out_tran_y

def get_gt_floor(pose, tran):
    _, joint = body_model.forward_kinematics(pose, tran=tran, calc_mesh=False)
    fc_probs = _foot_ground_probs(joint).clone()
    ground = _get_ground(joint, fc_probs, len(joint), contact_num=5)
    
    return ground

def get_name(model_list=["ours", "baseline"], i=0):
    name_list = ['gt'] + model_list
    name_list = [name.replace('_60fps', '') for name in name_list]
    name_list = [name + "_" + str(i) for name in name_list]
    
    return name_list

def get_foot_pos(pose, tran):
    _, joint = body_model.forward_kinematics(pose, tran=tran, calc_mesh=False)
    lfoot = joint[:, 10]
    rfoot = joint[:, 11]
    
    return torch.stack((lfoot, rfoot), dim=1)

def edit_rotation(pose, ori):
    pose = pose.view(-1, 24, 3, 3)
    ori = ori.view(-1, 5, 3, 3)
    
    glb_pose, _ = body_model.forward_kinematics(pose)
    glb_pose[:, 2] = ori[:, 3]
    
    pose_edit = body_model.inverse_kinematics_R(glb_pose)
    
    return pose_edit

def edit_pose(pose_p):
    pose = pose_p.clone().view(-1, 24, 3, 3)
    _, joint = body_model.forward_kinematics(pose=pose)
    
    _, joint_init = body_model.forward_kinematics(pose=pose[0].unsqueeze(0))
    joint_init = joint_init[0]
    N = pose.shape[0]
    
    vel_threshold = 1e-5
    init_threshold = 5e-3
    
    static = torch.zeros((N, 1), dtype=torch.bool, device=pose.device)    
    S = np.diag([1, -1, -1])
    S = torch.tensor(S, dtype=torch.float32, device=pose.device)
    
    start = False
    
    for i in range(1, N):
        last_larm = joint[i-1, joint_set.larm].clone().view(-1)
        last_rarm = joint[i-1, joint_set.rarm].clone().view(-1)
        cur_larm = joint[i, joint_set.larm].clone().view(-1)
        cur_rarm = joint[i, joint_set.rarm].clone().view(-1)
        init_larm = joint_init[joint_set.larm].clone().view(-1)
        init_rarm = joint_init[joint_set.rarm].clone().view(-1)
        
        vel_err = torch.mean((last_larm - cur_larm) ** 2)
        init_err = torch.mean((init_larm - cur_larm) ** 2)
        rarm_err = torch.mean((init_rarm - cur_rarm) ** 2)

        if vel_err > vel_threshold and init_err > init_threshold:
            start = True
            # pose[i, joint_set.rarm] = - pose[i, joint_set.larm].clone()
        
        if vel_err < vel_threshold and init_err < init_threshold:
            static[i] = True
            
        # 如果过去一段时间内，static为True，则将当前帧的右臂姿态设置为左臂姿态的镜像
        if i > 10 and torch.all(static[i-10:i]):
            start = False
            
        if not start:
            pose[i, joint_set.rarm] = torch.matmul(S, torch.matmul(pose[i, joint_set.larm], S))
            
    return pose

if __name__ == '__main__':
    
    data_dir = 'data/eval'
    dataset_name = 'totalcapture'
    model_list = ['heightposer_relative', 'mobileposer', 'imuposer']
    # model_list = ['heightposer_regression']
    
    # 获取data_dir/model_list[0]/dataset_name/中以.pt结尾的文件个数
    idx_num = len([name for name in os.listdir(os.path.join(data_dir, model_list[0], 'lw_rp', dataset_name)) if name.endswith('.pt')])

    idx = [i for i in range(0, 1)]
    print('len:', idx)
    
    for i in idx:
        
        pose_list = []
        
        for name in model_list:
            data_path = os.path.join(data_dir, name, 'lw_rp', dataset_name, str(i)+'.pt')
            data = torch.load(data_path)
            
            pose_t, pose_p = process_pose_data(data)
            
            if pose_list == []:
                pose_list.append(pose_t)
                # print frames
                print("frames:", pose_t.shape[0])
            
            # pose_list.append(pose_p)
        
        # name_list = get_name(model_list=model_list, i=i)
        # name_list = ['GT'+str(i), 'BaroPoser', 'MobilePoser', 'IMUPoser']
                
        # viewer_manager = MotionViewerManager(len(pose_list), overlap=False, names=name_list)

        name_list = ['GT']
        viewer_manager = MotionViewerManager(1, overlap=False, names=name_list)
        
        viewer_manager.visualize(pose_list, sensor_point=False)
        viewer_manager.close()