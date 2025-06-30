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
from models_new.poser import Poser
from models_new.joints import Joints
from models_new.velocity import Velocity

class PoseNet(L.LightningModule):
    """
    Inputs: N IMUs.
    Outputs: SMPL Pose Parameters (as 6D Rotations) and Translation. 
    """

    def __init__(self, poser: Poser=None, joints: Joints=None, velocity: Velocity=None,
                 finetune: bool=False, wheights: bool=False):
        super().__init__()

        # constants
        self.C = model_config
        self.finetune = finetune
        self.hypers = train_hypers 

        # body model
        self.bodymodel = art.model.ParametricModel(paths.smpl_file, device=self.C.device)
        self.global_to_local_pose = self.bodymodel.inverse_kinematics_R

        # model definitions
        self.pose = poser if poser else Poser()                   # pose estimation model
        self.joints = joints if joints else Joints()              # joint estimation model
        self.velocity = velocity if velocity else Velocity()      # velocity estimation model

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
        model = PoseNet.load_from_checkpoint(model_path)
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

    def predict(self, input, init_pose, gt_vel=None):
        
        input_lengths = input.shape[0]
        
        # predict velocity
        if gt_vel is None:
            init_vel = torch.zeros(9).to(self.device)
            pred_vel = self.velocity.predict_RNN(input, init_vel)
        else:
            pred_vel = gt_vel
        
        # predict joints
        input_joint = torch.cat((pred_vel, input), dim=1)
        pred_joint = self.joints.predict_RNN(input_joint, init_pose)
        
        # predict pose
        input_pose = torch.cat((pred_joint, input), dim=1)
        pred_pose = self.pose(input_pose.unsqueeze(0), [input_lengths])
        
        pred_pose = self._reduced_global_to_full(pred_pose.squeeze(0))
        
        return pred_pose
    
    def predict_vel(self, input):
        # predict velocity
        init_vel = torch.zeros(12).to(self.device)
        pred_vel = self.velocity.predict_RNN(input, init_vel)
        
        return pred_vel