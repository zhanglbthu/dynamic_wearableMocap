import os
import torch
import torch.nn as nn
import lightning as L
from torch.nn import functional as F
import numpy as np
import math
from config import *
from config import amass
from utils.model_utils import reduced_pose_to_full
import articulate as art
from model.base_model.rnn import RNNWithInit
from utils.data_utils import _foot_min, _get_heights

vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021])
ji_mask = torch.tensor([18, 19, 1, 2, 15, 0])

class Poser(L.LightningModule):
    """
    Inputs: N IMUs.
    Outputs: SMPL Pose Parameters (as 6D Rotations).
    """
    def __init__(self, combo_id: str="lw_rp_h", device='cuda'):
        super().__init__()
        
        # constants
        self.C = model_config
        self.hypers = train_hypers

        # input dimensions
        imu_set = amass.combos_mine[combo_id]
        imu_num = len(imu_set)
        imu_input_dim = imu_num * 12
        
        self.input_dim = imu_input_dim
        self.output_dim = 16 * 6
        self.init_size = self.output_dim
            
        # model definitions
        self.bodymodel = art.model.ParametricModel(paths.smpl_file, device=device)
        self.pose = RNNWithInit(n_input=self.input_dim, 
                                n_output=self.output_dim, 
                                n_hidden=512, 
                                n_rnn_layer=2, 
                                dropout=0.4, 
                                init_size=self.init_size,
                                bidirectional=False) # pose estimation model
        
        # log input and output dimensions
        if torch.cuda.current_device() == 0:
            print(f"Input dimensions: {self.input_dim}")
            print(f"Output dimensions: {self.output_dim}")
        
        # loss function
        self.loss = nn.MSELoss()

        # track stats
        self.validation_step_loss = []
        self.training_step_loss = []
        self.save_hyperparameters()

    @classmethod
    def from_pretrained(cls, model_path):
        # init pretrained-model
        model = Poser.load_from_checkpoint(model_path)
        model.hypers = finetune_hypers
        model.finetune = True
        return model

    def _reduced_global_to_full(self, reduced_pose):
        pose = art.math.r6d_to_rotation_matrix(reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
        pose = reduced_pose_to_full(pose.unsqueeze(0)).squeeze(0).view(-1, 24, 3, 3)
        pred_pose = self.global_to_local_pose(pose)
        for ignore in joint_set.ignored: pred_pose[:, ignore] = torch.eye(3, device=self.device)
        pred_pose[:, 0] = pose[:, 0]
        return pred_pose

    def predict_RNN(self, input, init_pose):
        input_lengths = input.shape[0]
        
        glb_rot, _ = self.bodymodel.forward_kinematics(init_pose.view(-1, 24, 3, 3))
        glb_rot_6d = art.math.rotation_matrix_to_r6d(glb_rot).view(-1, 24, 6)
        
        input = self.input_process(input, add_noise=False).view(-1, 24)
        init_p = glb_rot_6d.view(-1, 24, 6)[:, joint_set.reduced].view(1, -1)
            
        input = (input.unsqueeze(0), init_p)
        
        pred_pose = self.forward(input, [input_lengths])
        
        return pred_pose.squeeze(0)

    def forward(self, batch, input_lengths=None):
        # forward the pose prediction model
        pred_pose, _, _ = self.pose(batch, input_lengths)
        return pred_pose

    def input_process(self, pose_input, add_noise=False):
        if len(pose_input.shape) == 3:
            B, S, _ = pose_input.shape
        pose_input = pose_input.view(-1, 24)
        glb_acc = pose_input[:, :6].view(-1, 2, 3)
        glb_rot = pose_input[:, 6:24].view(-1, 2, 3, 3)
        
        if add_noise:
            pass
        
        input = torch.cat((glb_acc.flatten(1), glb_rot.flatten(1)), dim=1)
        return input
    
    def shared_step(self, batch, add_noise=False):
        # unpack data
        inputs, outputs = batch
        imu_inputs, input_lengths = inputs
        outputs, output_lengths = outputs

        # target pose
        target_pose = outputs['poses'] # [batch_size, window_length, 144] 
        B, S, _ = target_pose.shape

        # predict pose
        pose_input = imu_inputs  

        pose_input = self.input_process(pose_input, add_noise=False, global_coord=True).view(B, S, -1)
        init_pose = target_pose[:, 0].view(-1, 24, 6)[:, joint_set.reduced].view(B, -1)
        
        pose_input = (pose_input, init_pose)
        pose_t = target_pose.view(B, S, 24, 6)[:, :, joint_set.reduced].view(B, S, -1)
        
        pose_p = self(pose_input, input_lengths)

        # compute pose loss
        loss = self.loss(pose_p, pose_t)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, add_noise=True)
        self.log("training_step_loss", loss.item(), batch_size=self.hypers.batch_size)
        self.training_step_loss.append(loss.item())
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, add_noise=False)
        self.log("validation_step_loss", loss.item(), batch_size=self.hypers.batch_size)
        self.validation_step_loss.append(loss.item())
        return {"loss": loss}
    
    def predict_step(self, batch, batch_idx):
        inputs, target = batch
        imu_inputs, input_lengths = inputs
        return self(imu_inputs, input_lengths)

    # train epoch start
    def on_fit_start(self):
        self.bodymodel = art.model.ParametricModel(paths.smpl_file, device=self.device)
        self.global_to_local_pose = self.bodymodel.inverse_kinematics_R
    
    def on_train_epoch_end(self):
        self.epoch_end_callback(self.training_step_loss, loop_type="train")
        self.training_step_loss.clear()    # free memory

    def on_validation_epoch_end(self):
        self.epoch_end_callback(self.validation_step_loss, loop_type="val")
        self.validation_step_loss.clear()  # free memory

    def on_test_epoch_end(self, outputs):
        self.epoch_end_callback(outputs, loop_type="test")

    def epoch_end_callback(self, outputs, loop_type):
        # log average loss
        average_loss = torch.mean(torch.Tensor(outputs))
        self.log(f"{loop_type}_loss", average_loss, prog_bar=True, batch_size=self.hypers.batch_size)
        # log learning late
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", lr, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hypers.lr) 
        return optimizer