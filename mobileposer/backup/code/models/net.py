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
from models.poser import Poser
from models.joints import Joints
from models.footcontact import FootContact
from models.velocity import Velocity

class MobilePoserNet(L.LightningModule):
    """
    Inputs: N IMUs.
    Outputs: SMPL Pose Parameters (as 6D Rotations) and Translation. 
    """

    def __init__(self, poser: Poser=None, joints: Joints=None, foot_contact: FootContact=None, velocity: Velocity=None, 
                 finetune: bool=False,
                 wheights: bool=False):
        super().__init__()

        # constants
        self.C = model_config
        self.finetune = finetune
        self.hypers = train_hypers 

        # body model
        self.bodymodel = art.model.ParametricModel(paths.smpl_file, device=self.C.device)
        self.global_to_local_pose = self.bodymodel.inverse_kinematics_R

        # model definitions
        self.pose = poser if poser else Poser()                                 # pose estimation model
        self.joints = joints if joints else Joints(height=wheights)             # joint estimation model
        self.foot_contact = foot_contact if foot_contact else FootContact()     # foot-ground probability model
        self.velocity = velocity if velocity else Velocity()                    # joint velocity model

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
        batch: [1, total_frames, 38]
        '''
        # forward the joint prediction model
        pred_joints = self.joints(batch, input_lengths) # [batch_size, total_frames, 72]

        # forward the pose prediction model
        
        # select batch[:, :, :self.C.n_imu] as input to the pose model
        batch = batch[:, :, :self.C.n_imu]
        
        pose_input = torch.cat((pred_joints, batch), dim=-1) # [batch_size, total_frames, 110]
        pred_pose = self.pose(pose_input, input_lengths) # [total_frames, 24, 3, 3]
        
        # global pose to local
        pred_pose = self._reduced_global_to_full(pred_pose)

        # forward the foot-ground contact probability model
        tran_input = torch.cat((pred_joints, batch), dim=-1)
        foot_contact = self.foot_contact(tran_input, input_lengths) # [batch_size, total_frames, 2]

        # foward the foot-joint velocity model
        vel_input = torch.cat((pred_joints, batch), dim=-1)
        pred_vel = self.velocity.forward_online(vel_input, input_lengths).squeeze(0)

        return pred_pose, pred_joints, pred_vel, foot_contact

    def forward_vel(self, batch, input_lengths=None):
        vel_input = batch[:, :, :self.C.n_imu]
        pred_vel = self.velocity.forward_online(vel_input, input_lengths).squeeze(0)
        
        return pred_vel
    
    @torch.no_grad()
    def forward_offline(self, imu, input_lengths=None):
        '''
        imu: [batch_size, N, 60]
        '''
        # forward the predcition model
        pose, pred_joints, vel, contact = self.forward(imu, input_lengths)
        contact = contact.squeeze(0) 

        # compute joints from predicted pose
        joints = pred_joints.squeeze(0).view(-1, 24, 3)

        # calculate velocity from foot-ground contact
        gravity_velocity = torch.tensor([0, joint_set.gravity_velocity, 0]).to(self.C.device)
        floor_y = self.j[10:12, 1].min().item()
        contact_vel = gravity_velocity + art.math.lerp(
            torch.cat((torch.zeros(1, 3).to(self.C.device), joints[:-1, 10] - joints[1:, 10])),
            torch.cat((torch.zeros(1, 3).to(self.C.device), joints[:-1, 11] - joints[1:, 11])),
            contact.max(dim=1).indices.view(-1, 1)
        )

        # velocity from network-based estimation
        root_vel = vel.view(-1, 24, 3)[:, 0]
        pred_vel = root_vel / (datasets.fps/amass.vel_scale)

        # compute velocity as a weighted combination of network-based and foot-contact-based
        weight = self._prob_to_weight(contact.max(dim=1).values.sigmoid()).view(-1, 1)
        velocity = art.math.lerp(pred_vel, contact_vel, weight)

        # remove penetration
        current_root_y = 0
        for i in range(velocity.shape[0]):
            current_foot_y = current_root_y + joints[i, 10:12, 1].min().item()
            if current_foot_y + velocity[i, 1].item() <= floor_y:
                velocity[i, 1] = floor_y - current_foot_y
            current_root_y += velocity[i, 1].item()
        tran = torch.stack([velocity[:i+1].sum(dim=0) for i in range(velocity.shape[0])]) # velocity to root position

        # Use a Physics Optimizer
        if getenv("PHYSICS"):
            pose = pose.view(-1, 24, 3, 3)
            acc = torch.zeros((pose.shape[0], 5, 3))

            # compute joint velocities
            joint_velocity = vel.view(-1, 24, 3) * amass.vel_scale

            pose_opt, tran_opt = [], []
            for p, c, v, a in tqdm(zip(pose, contact, joint_velocity, acc), total=len(pose)):
                p, t = self.dynamics_optimizer.optimize_frame(p, v, c, a)
                pose_opt.append(p)
                tran_opt.append(t)
            pose, _ = torch.stack(pose_opt), torch.stack(tran_opt).unsqueeze(0)

        return pose, pred_joints, tran, contact

    @torch.no_grad()
    def forward_online_tran(self, data, input_lengths=None):
        imu = data.repeat(self.num_total_frames, 1) if self.imu is None else torch.cat((self.imu[1:], data.view(1, -1)))

        vel = self.forward_vel(imu.unsqueeze(0), [self.num_total_frames])
        
        root_vel = vel.view(-1, 24, 3)[:, 0]
        
        pred_vel = root_vel[self.num_past_frames] / (datasets.fps/amass.vel_scale)
        
        velocity = pred_vel
        
        self.last_root_pos += velocity
        self.imu = imu.squeeze(0)
        
        return self.last_root_pos.clone()
        
    @torch.no_grad()
    def forward_joint(self, data):
        
        imu = data.repeat(self.num_total_frames, 1) if self.imu is None else torch.cat((self.imu[1:], data.view(1, -1)))   
        
        pose, pred_joints, vel, contact = self.forward(imu.unsqueeze(0), [self.num_total_frames])
        
        return pred_joints.squeeze(0) 
    
    @torch.no_grad()
    def forward_online(self, data, input_lengths=None, debug=False, pred_joint = None):
        
        imu = data.repeat(self.num_total_frames, 1) if self.imu is None else torch.cat((self.imu[1:], data.view(1, -1)))

        # forward the pose prediction model
        pose, pred_joints, vel, contact = self.forward(imu.unsqueeze(0), [self.num_total_frames])

        # get pose
        pose = pose[self.num_past_frames].view(-1, 9)

        # compute the joint positions from predicted pose
        joints = pred_joints.squeeze(0)[self.num_past_frames].view(24, 3)

        # compute translation from foot-contact probability
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

        # physics module
        if getenv("PHYSICS"):
            joint_velocity = vel.view(-1, 24, 3) 

            # optimize pose
            pose, _ = self.dynamics_optimizer.optimize_frame(pose, joint_velocity[self.num_past_frames]*amass.vel_scale, contact, imu)
            pose = pose.view(24, 3, 3)
            return pose, pred_joints.squeeze(0), self.last_root_pos.clone(), contact

        if debug:
            return pose, joints, self.last_root_pos.clone(), contact
        
        return pose, pred_joints.squeeze(0), self.last_root_pos.clone(), contact