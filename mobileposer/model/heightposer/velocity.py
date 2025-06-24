import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import lightning as L
from torch.optim.lr_scheduler import StepLR 

from articulate.model import ParametricModel
from model.base_model.rnn import RNNWithInit, RNN
from config import *
import articulate as art

class Velocity(L.LightningModule):
    """
    Inputs: N IMUs.
    Outputs: Per-Frame Root Velocity. 
    """

    def __init__(self, combo_id = 'lw_rp_h'):
        super().__init__()
        
        self.imu_set = amass.combos_mine[combo_id]
        self.imu_nums = len(self.imu_set)
        
        # constants
        self.C = model_config
        self.hypers = train_hypers
        self.bodymodel = ParametricModel(paths.smpl_file, device=self.C.device)
        self.input_size = 12 * self.imu_nums + 1 if self.C.vel_wh else 12 * self.imu_nums
        self.vel_joint = amass.vel_joint
        self.output_size = len(self.vel_joint) * 2
        
        # model definitions
        self.vel = RNN(n_input=self.input_size, n_output=self.output_size, n_hidden=512,
                                 n_rnn_layer=3, dropout=0.4, bidirectional=False)
        
        self.rnn_state = None

        # log input and output dimensions
        print(f"combo_id: {combo_id}")
        print(f"Input dimensions: {self.input_size}")
        print(f"Output dimensions: {self.output_size}")
        
        # loss function 
        self.loss = nn.MSELoss()

        # track stats
        self.validation_step_loss = []
        self.training_step_loss = []
        self.save_hyperparameters()

    def reset(self):
        self.rnn_state = None
        self.imu = None
        self.last_root_pos = torch.zeros(3).to(self.C.device)

    def forward(self, batch, input_lengths=None):
        # forward joint model
        vel, _, _ = self.vel(batch, input_lengths)

        return vel

    def predict_RNN(self, input):
        input_lengths = input.shape[0]
        
        input = input.unsqueeze(0)
        
        pred_vel = self.forward(input, [input_lengths])
        
        return pred_vel.squeeze(0)
        
    def forward(self, batch, input_lengths=None):
        # forward velocity model
        vel, _, _ = self.vel(batch, input_lengths)
        return vel

    def forward_online(self, batch, input_lengths=None):
        # forward velocity model
        vel, _, self.rnn_state = self.vel(batch, input_lengths, self.rnn_state)
        return vel
    
    def input_process(self, inputs):
        # process input
        _, _, input_dim = inputs.shape
        inputs = inputs.view(-1, input_dim)
        
        imu_inputs = inputs[:, :12 * self.imu_nums]
        h_inputs = inputs[:, -1:].view(-1, 1)
        
        inputs = torch.cat([imu_inputs, h_inputs], dim=-1)
        
        return inputs
        
    def shared_step(self, batch, add_noise=False):
        # unpack data
        inputs, outputs = batch
        imu_inputs, input_lengths = inputs
        outputs, _ = outputs
        
        # target joints
        joints = outputs['joints']
        B, S, _, _ = joints.shape

        # target velocity
        target_vel = outputs['vels'][:, :, amass.vel_joint].view(B, S, -1)
        target_vel = target_vel[:, :, [0, 2]]

        # predict joint velocity
        # change: add noise
        if add_noise:
            imu_inputs_noisy = imu_inputs.clone()
            rot = imu_inputs_noisy[..., 6:24].view(B, S, 2, 3, 3)   # shape: [B, S, 18]
            
            axis_angle_thigh = torch.randn(B, 1, 3).to(self.device) * self.C.noise_std
            axis_angle_wrist = torch.randn(B, 1, 3).to(self.device) * self.C.noise_std
            
            wrist_rot = rot[:, :, 0].view(B, S, 3, 3)
            thigh_rot = rot[:, :, 1].view(B, S, 3, 3)
            
            wrist_rot_noisy = torch.matmul(wrist_rot, art.math.axis_angle_to_rotation_matrix(axis_angle_wrist).view(B, 1, 3, 3)).view(B, S, 9)
            thigh_rot_noisy = torch.matmul(thigh_rot, art.math.axis_angle_to_rotation_matrix(axis_angle_thigh).view(B, 1, 3, 3)).view(B, S, 9)
            
            rot_noisy = torch.cat([wrist_rot_noisy, thigh_rot_noisy], dim=-1)

            imu_inputs_noisy[..., 6:24] = rot_noisy
            
            imu_inputs = imu_inputs_noisy 
            
        imu_inputs = self.input_process(imu_inputs).view(B, S, -1)

        pred_vel, _, _ = self.vel(imu_inputs, input_lengths)
        
        # # velocity loss
        # loss = self.loss(pred_vel, target_vel)

        # position loss
        pred_tran = torch.sum(pred_vel, dim=1)
        target_tran = torch.sum(target_vel, dim=1)
        loss = self.loss(pred_tran, target_tran)
        
        # # mobileposer loss
        # loss = self.compute_loss(pred_vel, target_vel)

        return loss

    def compute_loss(self, pred_vel, gt_vel):
        loss = sum(self.compute_vel_loss(pred_vel, gt_vel, i) for i in [1, 3, 9])
        return loss

    def compute_vel_loss(self, pred_vel, gt_vel, n=1):
        T = pred_vel.shape[1]
        loss = 0.0

        for m in range(0, T//n):
            end = min(n*m+n, T)
            loss += self.loss(pred_vel[:, m*n:end, :], gt_vel[:, m*n:end, :])

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, add_noise=True)
        self.log("training_step_loss", loss.item(), batch_size=self.hypers.batch_size)
        self.training_step_loss.append(loss.item())
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("validation_step_loss", loss.item(), batch_size=self.hypers.batch_size)
        self.validation_step_loss.append(loss.item())
        return {"loss": loss}
    
    def predict_step(self, batch, batch_idx):
        inputs, target = batch
        imu_inputs, input_lengths = inputs
        return self(imu_inputs, input_lengths)

    def on_train_epoch_end(self):
        self.epoch_end_callback(self.training_step_loss, loop_type="train")
        self.training_step_loss.clear()    # free memory

    def on_validation_epoch_end(self):
        self.epoch_end_callback(self.validation_step_loss, loop_type="val")
        self.validation_step_loss.clear()  # free memory

    def on_test_epoch_end(self, outputs):
        self.epoch_end_callback(outputs, loop_type="test")

    def epoch_end_callback(self, outputs, loop_type):
        average_loss = torch.mean(torch.Tensor(outputs))
        self.log(f"{loop_type}_loss", average_loss, prog_bar=True, batch_size=self.hypers.batch_size)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hypers.lr) 