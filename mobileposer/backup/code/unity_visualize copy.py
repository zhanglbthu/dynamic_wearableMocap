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

class MotionViewerManager:
    def __init__(self, sub_num, overlap=True, names=None):
        self.viewer = MotionViewer(sub_num, overlap, names)
        self.viewer.connect()
        self.ground = None

    def plot_ground(self, f_pos, ground):
        grid_size = 10 
        sample_range = 1.0
        
        # 中心点的 x, z 坐标
        x_center = (f_pos[0][0] + f_pos[1][0]) / 2
        z_center = (f_pos[0][2] + f_pos[1][2]) / 2
        y = ground.item()  # y 坐标使用地面的值
        
        if self.ground is None:
            self.ground = y

        floor_color = [255, 255, 255]
        
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

        for i in range(len(pose[0])):
            clock.tick(60)
            self.viewer.clear_line(render=False)
            self.viewer.clear_point(render=False)

            pose_list = [pose[sub][i] for sub in range(sub_num)]
            tran_list = [tran[sub][i] for sub in range(sub_num)] if tran else [torch.zeros(3) for _ in range(sub_num)]

            if ground is not None:
                self.plot_ground(f_pos[i], ground[i])
                
            if contact is not None:
                contact_list = [contact[sub][i] for sub in range(sub_num)]
                self.plot_contact(pose_list, tran_list, contact_list)
            
            self.viewer.update_all(pose_list, tran_list, render=False)
            self.viewer.render()
            print('\r', clock.get_fps(), end='')

    def close(self):
        self.viewer.disconnect()

def vis_height():
    pose_path = os.path.join(deta_dir, 'pose.pt')
    tran_path = os.path.join(deta_dir, 'tran.pt')
    height_path = os.path.join(deta_dir, 'height.pt')
    
    poses = torch.load(pose_path)
    trans = torch.load(tran_path)
    heights = torch.load(height_path)
    
    index = 1
    pose = poses[index]
    tran = trans[index]
    height = heights[index]
    frame_num = pose.shape[0]
    
    clock = Clock()
    
    with MotionViewer(1, overlap=True, names=".") as viewer:
        for i in range(frame_num):
            clock.tick(60)
            viewer.clear_line(render=False)
            viewer.clear_point(render=False)
            
            pose_matrix = art.math.axis_angle_to_rotation_matrix(pose[i]).unsqueeze(0)
            
            _, _, glb_verts = body_model.forward_kinematics(pose_matrix, tran=tran[i], calc_mesh=True)
            
            viewer.update_all([pose_matrix.squeeze(0)], [tran[i]], render=False)
            
            pos1 = glb_verts[0, vi_mask[0]]
            pos2 = glb_verts[0, vi_mask[3]]
            h1 = height[i][0]
            h2 = height[i][1]
            viewer.draw_point(pos1, color=[255, 0, 0], radius=0.05, render=False)
            viewer.draw_point(pos2, color=[255, 0, 0], radius=0.05, render=False)
            viewer.draw_line(pos1, [pos1[0], pos1[1]-h1, pos1[2]], color=[255, 0, 0], render=False)
            viewer.draw_line(pos2, [pos2[0], pos2[1]-h2, pos2[2]], color=[255, 0, 0], render=False)
            
            viewer.render()
            
            print('\r', clock.get_fps(), end='')

def value2color(value):
    value = torch.clamp(value, 0, 1).cpu().numpy()
    color = np.array([1, 1 - value, 1 - value])
    return color

def vis_contact(viewer: MotionViewer, pose_list, tran_list, contact_list):
    for i in range(len(pose_list)):
        pose = torch.tensor(pose_list[i]).unsqueeze(0)
        tran = torch.tensor(tran_list[i]).unsqueeze(0)
        
        _, glb_joint = body_model.forward_kinematics(pose, tran=tran, calc_mesh=False)
        
        viewer.draw_point(glb_joint[0][10], color=value2color(contact_list[i][0]), radius=0.2, render=False)
        viewer.draw_point(glb_joint[0][11], color=value2color(contact_list[i][1]), radius=0.2, render=False)

def vis_rheight(viewer: MotionViewer, pose_list, tran_list, rheight_list, offsets=None):
    for i in range(len(pose_list)):
        pose = torch.tensor(pose_list[i]).unsqueeze(0)
        tran = torch.tensor(tran_list[i]).unsqueeze(0)
        
        _, _, glb_verts = body_model.forward_kinematics(pose, tran=tran, calc_mesh=True)
        
        pos1 = glb_verts[0, vi_mask[0]]
        pos2 = glb_verts[0, vi_mask[3]]
        
        if offsets:

            pos1 = pos1 + torch.tensor(offsets[i])
            pos2 = pos2 + torch.tensor(offsets[i])
        
        h = rheight_list[i].cpu()
        viewer.draw_point(pos1, color=[255, 0, 0], radius=0.05, render=False)
        viewer.draw_point(pos2, color=[255, 0, 0], radius=0.05, render=False)
        viewer.draw_line(pos1, [pos1[0], pos1[1]-h, pos1[2]], color=[255, 0, 0], width=0.02, render=False)

def plot_joint(viewer: MotionViewer, joint_list, offsets=None):
    for i in range(len(joint_list)):
        joint = joint_list[i]
        
        if offsets:
            joint = joint + torch.tensor(offsets[i])
        
        for j in range(joint.shape[0]):
            viewer.draw_point(joint[j], color=[255, 0, 0], radius=0.05, render=False)

def visualize(pose, tran=None, rheight=None, joint=None, contact=None, names=None):
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

            if rheight:
                rheight_list = [rheight[sub_idx][i] for sub_idx in range(sub_num)]
                vis_rheight(viewer, pose_list, tran_list, rheight_list, offsets=viewer.offsets)
            
            if joint:
                joint_list = [joint[sub_idx][i] for sub_idx in range(sub_num)]
                plot_joint(viewer, joint_list, offsets=viewer.offsets)
            
            if contact:
                contact_list = [contact[sub_idx][i] for sub_idx in range(sub_num)]
                vis_contact(viewer, pose_list, tran_list, contact_list)
            
            viewer.update_all(pose_list, tran_list, render=False)
            
            viewer.render()
            
            print('\r', clock.get_fps(), end='')

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

def edit_height(pose_t, tran_t, pose_p, contact_p):
    _, glb_joint, glb_vert = body_model.forward_kinematics(pose_t, tran=tran_t, calc_mesh=True)

if __name__ == '__main__':
    
    data_dir = 'data/eval'
    dataset_name = 'cmu'
    model_list = ['heights_60fps']
    
    pose_list = []
    tran_list = []
    joint_list = []
    
    # 获取data_dir/model_list[0]/dataset_name/中以.pt结尾的文件个数
    idx_num = len([name for name in os.listdir(os.path.join(data_dir, model_list[0], 'lw_rp_h', dataset_name)) if name.endswith('.pt')])

    idx = [i for i in range(idx_num)]
    print('len:', idx)
    
    for i in idx:
        
        pose_list = []
        tran_list = []
        contact_list = []
        
        for name in model_list:
            data_path = os.path.join(data_dir, name, 'lw_rp_h', dataset_name, str(i)+'.pt')
            data = torch.load(data_path)
            
            pose_t, pose_p, tran_t, tran_p, contact_t, contact_p = process_mobileposer_data(data, contact=True)
            if pose_list == []:
                pose_list.append(pose_t)
                tran_list.append(tran_t)
                contact_list.append(contact_t)
            
            pose_list.append(pose_p)
            tran_list.append(tran_p)
            contact_list.append(contact_p)
        
        name_list = ['gt'] + model_list
        name_list = [name.replace('_60fps', '') for name in name_list]
        name_list = [name + "_" + str(i) for name in name_list]
            
        # visualize(pose_list, tran=tran_list, names=name_list, contact=contact_list)
        viewer_manager = MotionViewerManager(len(pose_list), overlap=True, names=name_list)
        viewer_manager.visualize(pose_list, tran=tran_list, contact=contact_list)
        viewer_manager.close()