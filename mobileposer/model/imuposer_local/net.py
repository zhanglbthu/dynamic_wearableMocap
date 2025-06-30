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

from model.imuposer_local.poser import Poser

class IMUPoserNet(L.LightningModule):
    """
    Inputs: N IMUs.
    Outputs: SMPL Pose Parameters (as 6D Rotations) and Translation. 
    """

    def __init__(self, poser: Poser=None, 
                 finetune: bool=False,
                 combo_id: str="lw_rp_h",):
        
        super().__init__()

        # constants
        self.C = model_config
        self.finetune = finetune
        self.hypers = train_hypers 

        # body model
        self.bodymodel = art.model.ParametricModel(paths.smpl_file, device=self.C.device)
        self.global_to_local_pose = self.bodymodel.inverse_kinematics_R

        # model definitions
        self.pose = poser if poser else Poser(combo_id=combo_id)                                 # pose estimation model

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
        self.save_hyperparameters(ignore=['poser'])

    @classmethod
    def from_pretrained(cls, model_path):
        # init pretrained-model
        model = IMUPoserNet.load_from_checkpoint(model_path)
        model.hypers = finetune_hypers
        model.finetune = True
        return model

    def reset(self):
        self.imu = None
    
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
        batch = batch[:, :, :self.C.n_imu]
        
        pose_input = batch
        pred_pose = self.pose(pose_input, input_lengths) # [total_frames, 24, 3, 3]
        
        # # global pose to local
        # pred_pose = self._reduced_global_to_full(pred_pose)
        
        # 6d to rotation matrix
        pred_pose = art.math.r6d_to_rotation_matrix(pred_pose).view(-1, 24, 3, 3)

        return pred_pose
    
    @torch.no_grad()
    def forward_online(self, data, input_lengths=None):
        
        imu = data.repeat(self.num_total_frames, 1) if self.imu is None else torch.cat((self.imu[1:], data.view(1, -1)))

        # forward the pose prediction model
        pose = self.forward(imu.unsqueeze(0), [self.num_total_frames])
        
        self.imu = imu

        # get pose
        pose = pose[self.num_past_frames].view(-1, 9)

        return pose