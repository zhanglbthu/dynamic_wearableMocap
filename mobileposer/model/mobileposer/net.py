import os
import numpy as np
import torch
torch.set_printoptions(sci_mode=False)
import torch.nn as nn
from torch.nn import functional as F
import lightning as L
from torch.optim.lr_scheduler import StepLR 
from tqdm import tqdm
import time

from config import *
from utils.model_utils import reduced_pose_to_full
import articulate as art
from model.mobileposer.poser import Poser
from model.mobileposer.joints import Joints
from model.mobileposer.velocity import Velocity
from model.mobileposer.footcontact import FootContact

class MobilePoserNet(L.LightningModule):
    """
    Inputs: N IMUs.
    Outputs: SMPL Pose Parameters (as 6D Rotations) and Translation. 
    """

    def __init__(self, poser: Poser=None, joints: Joints=None, foot_contact: FootContact=None, velocity: Velocity=None,
                 finetune: bool=False, 
                 wheights: bool=False,
                 combo_id: str="lw_rp_h"):
        super().__init__()

        # constants
        self.C = model_config
        self.finetune = finetune
        self.hypers = train_hypers 

        # body model
        self.bodymodel = art.model.ParametricModel(paths.smpl_file, device=self.C.device)
        self.global_to_local_pose = self.bodymodel.inverse_kinematics_R

        # model definitions
        self.pose = poser if poser else Poser(combo_id=combo_id)                   # pose estimation model
        self.joints = joints if joints else Joints(combo_id=combo_id)              # joint estimation model
        self.foot_contact = foot_contact if foot_contact else FootContact(combo_id=combo_id)    # foot contact estimation model
        self.velocity = velocity if velocity else Velocity(combo_id=combo_id)       # velocity estimation model

        # base joints
        self.j, _ = self.bodymodel.get_zero_pose_joint_and_vertex() # [24, 3]
        self.feet_pos = self.j[10:12].clone() # [2, 3]
        self.floor_y = self.j[10:12, 1].min().item() # [1]

        # constants
        self.gravity_velocity = torch.tensor([0, joint_set.gravity_velocity, 0]).to(self.C.device)
        self.prob_threshold = (0.5, 0.9)
        self.num_past_frames = model_config.past_frames
        self.num_future_frames = model_config.future_frames
        self.num_total_frames = self.num_past_frames + self.num_future_frames

        # variables
        self.last_lfoot_pos, self.last_rfoot_pos = (pos.to(self.C.device) for pos in self.feet_pos)
        self.last_root_pos = torch.zeros(3).to(self.C.device)
        self.last_joints = torch.zeros(24, 3).to(self.C.device)
        self.current_root_y = 0
        self.imu = None
        self.rnn_state = None

        # track stats
        self.validation_step_loss = []
        self.training_step_loss = []
        self.save_hyperparameters(ignore=['poser', 'joints', 'foot_contact', 'velocity'])

    @classmethod
    def from_pretrained(cls, model_path):
        # init pretrained-model
        model = MobilePoserNet.load_from_checkpoint(model_path)
        model.hypers = finetune_hypers
        model.finetune = True
        return model

    def reset(self):
        self.rnn_state = None
        self.imu = None
        self.current_root_y = 0
        self.last_root_pos = torch.zeros(3).to(self.C.device)

    def _prob_to_weight(self, p):
        return (p.clamp(self.prob_threshold[0], self.prob_threshold[1]) - self.prob_threshold[0]) / (self.prob_threshold[1] - self.prob_threshold[0])
    
    def _reduced_global_to_full(self, reduced_pose):
        pose = art.math.r6d_to_rotation_matrix(reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
        pose = reduced_pose_to_full(pose.unsqueeze(0)).squeeze(0).view(-1, 24, 3, 3)
        pred_pose = self.global_to_local_pose(pose)
        pred_pose[:, joint_set.ignored] = torch.eye(3, device=self.C.device)
        pred_pose[:, 0] = pose[:, 0]
        return pred_pose

    def forward(self, batch, input_lengths=None):
        '''
        batch: [1, total_frames, 36]
        '''
        # forward the joint prediction model
        pred_joints = self.joints(batch, input_lengths) # [batch_size, total_frames, 72]

        # forward the pose prediction model
        
        # select batch[:, :, :self.C.n_imu] as input to the pose model
        batch = batch[:, :, :self.C.n_imu]
        
        pose_input = torch.cat((pred_joints, batch), dim=-1) # [batch_size, total_frames, 110]
        pred_pose = self.pose(pose_input, input_lengths) # [total_frames, 24, 3, 3]
        # pred_pose: [1, 45, 96]
        
        # global pose to local
        pred_pose = self._reduced_global_to_full(pred_pose)
        
        # forward the foot-ground contact probability model
        tran_input = torch.cat((pred_joints, batch), dim=-1)
        foot_contact = self.foot_contact(tran_input, input_lengths) # [batch_size, total_frames, 2]

        # foward the foot-joint velocity model
        vel_input = torch.cat((pred_joints, batch), dim=-1)
        pred_vel = self.velocity.forward_online(vel_input, input_lengths).squeeze(0)

        return pred_pose, pred_joints, pred_vel, foot_contact

    @torch.no_grad()
    def forward_frame(self, data, input_lengths=None, tran=False):
        
        imu = data.repeat(self.num_total_frames, 1) if self.imu is None else torch.cat((self.imu[1:], data.view(1, -1)))

        # forward the pose prediction model
        pose, pred_joints, vel, contact = self.forward(imu.unsqueeze(0), [self.num_total_frames])

        # get pose
        pose = pose[self.num_past_frames].view(-1, 9)

        # compute the joint positions from predicted pose
        joints = pred_joints.squeeze(0)[self.num_past_frames].view(24, 3)
        
        # compute translation positions from predicted pose
        contact = contact[0][self.num_past_frames]
        lfoot_pos, rfoot_pos = joints[10], joints[11]
        if contact[0] > contact[1]:
            contact_vel = self.last_lfoot_pos - lfoot_pos + self.gravity_velocity
        else:
            contact_vel = self.last_rfoot_pos - rfoot_pos + self.gravity_velocity
        
        # velocity from network-based estimation
        root_vel = vel.view(-1, 24, 3)[:, 0] # select root velocity
        pred_vel = root_vel[self.num_past_frames] / (datasets.fps/amass.vel_scale)
        
        weight = self._prob_to_weight(contact.max())
        
        velocity = art.math.lerp(pred_vel, contact_vel, weight)

        # remove penetration
        current_foot_y = self.current_root_y + min(lfoot_pos[1].item(), rfoot_pos[1].item())
        if current_foot_y + velocity[1].item() <= self.floor_y:
            velocity[1] = self.floor_y - current_foot_y

        self.current_root_y += velocity[1].item()
        self.last_lfoot_pos, self.last_rfoot_pos = lfoot_pos, rfoot_pos
        self.imu = imu.squeeze(0)
        self.last_root_pos += velocity
        
        if tran:
            return pose, self.last_root_pos.clone()
        
        return pose