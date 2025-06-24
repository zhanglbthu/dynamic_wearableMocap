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

from model.heightposer.poser import Poser
from model.heightposer.velocity import Velocity
from model.heightposer.footcontact import FootContact
from torch.nn.functional import relu

class HeightPoserNet(L.LightningModule):
    """
    Inputs: N IMUs.
    Outputs: SMPL Pose Parameters (as 6D Rotations) and Translation. 
    """

    def __init__(self, 
                 poser: Poser=None,  
                 combo_id: str="lw_rp_h"):
        super().__init__()

        # constants
        self.C = model_config
        self.hypers = train_hypers 

        # body model
        self.bodymodel = art.model.ParametricModel(paths.smpl_file, device=self.C.device)
        self.global_to_local_pose = self.bodymodel.inverse_kinematics_R

        # model definitions
        self.pose = poser if poser else Poser(combo_id=combo_id)                   # pose estimation model

        # variables
        self.imu = None
        self.pose_rnn_state = None

        # track stats
        self.validation_step_loss = []
        self.training_step_loss = []
        self.save_hyperparameters(ignore=['poser', 'foot_contact', 'velocity'])

    @classmethod
    def from_pretrained(cls, model_path):
        # init pretrained-model
        model = HeightPoserNet.load_from_checkpoint(model_path)
        model.hypers = finetune_hypers
        model.finetune = True
        return model

    def reset(self):
        self.pose_rnn_state = None
        self.imu = None
    
    def _reduced_global_to_full(self, reduced_pose):
        pose = art.math.r6d_to_rotation_matrix(reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
        pose = reduced_pose_to_full(pose.unsqueeze(0)).squeeze(0).view(-1, 24, 3, 3)
        pred_pose = self.global_to_local_pose(pose)
        pred_pose[:, joint_set.leaf_joint] = torch.eye(3, device=self.C.device)
        pred_pose[:, 0] = pose[:, 0]
        return pred_pose

    def _reduced_glb_6d_to_full_local_mat(self, root_rotation, glb_reduced_pose):
        glb_reduced_pose = art.math.r6d_to_rotation_matrix(glb_reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
        glb_reduced_pose = root_rotation.unsqueeze(1).matmul(glb_reduced_pose)
        
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
        global_full_pose[:, joint_set.reduced] = glb_reduced_pose
        global_full_pose[:, 2] = root_rotation
        pose = self.global_to_local_pose(global_full_pose).view(-1, 24, 3, 3)
        
        pose[:, joint_set.leaf_joint] = torch.eye(3, device=pose.device)
        
        return pose

    def input_process(self, pose_input, add_noise=False):
        pose_input = pose_input.view(-1, 24)
        glb_acc = pose_input[:, :6].view(-1, 2, 3)
        glb_rot = pose_input[:, 6:24].view(-1, 2, 3, 3)
        
        if add_noise:
            pass
        
        input = torch.cat((glb_acc.flatten(1), glb_rot.flatten(1)), dim=1)
        return input

    @torch.no_grad()
    def forward_frame(self, input):
        '''
        input: 2 * glb_acc [3] + 2 * glb_rot [9] + rel_height [1]
        '''
        device = self.C.device
        
        # predict pose
        pose_init = torch.eye(3).repeat(1, 24, 1, 1).to(device)
        glb_rot, _ = self.bodymodel.forward_kinematics(pose_init)
        
        glb_rot_6d = art.math.rotation_matrix_to_r6d(glb_rot).view(-1, 24, 6)
        init_p = glb_rot_6d.view(-1, 24, 6)[:, joint_set.reduced].view(1, -1)
        
        if self.pose_rnn_state is None:
            h, c = self.pose.pose.init_net(init_p).view(-1, 2, 2, 512).permute(1, 2, 0, 3)
            self.pose_rnn_state = (h, c)
        
        pose_input = self.input_process(input, add_noise=False).view(-1, 24)
        data = relu(self.pose.pose.linear1(pose_input), inplace=True)
        data, self.pose_rnn_state = self.pose.pose.rnn(data.unsqueeze(0), self.pose_rnn_state)
        pred_pose = self.pose.pose.linear2(data)
        
        pred_pose = self._reduced_global_to_full(pred_pose)
        
        return pred_pose
        