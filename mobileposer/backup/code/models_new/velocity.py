import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import lightning as L
from torch.optim.lr_scheduler import StepLR 

from mobileposer.articulate.model import ParametricModel
from mobileposer.models.rnn import RNN, RNNWithInit
from mobileposer.config import *

class Velocity(L.LightningModule):
    """
    Inputs: N IMUs.
    Outputs: Per-Frame Root Velocity. 
    """

    def __init__(self, finetune: bool=False, imu_num: int=3, combo_id = 'lw_rp_h', height: bool=False, winit=False):
        super().__init__()
        
        self.imu_set = amass.combos_mine[combo_id]
        self.imu_nums = len(self.imu_set)
        
        # constants
        self.C = model_config
        self.hypers = train_hypers
        self.bodymodel = ParametricModel(paths.smpl_file, device=self.C.device)
        self.input_size = 12 * self.imu_nums
        self.vel_joint = amass.vel_joint
        self.output_size = len(self.vel_joint) * 3
        
        # model definitions
        self.vel = RNNWithInit(n_input=self.input_size, n_output=self.output_size, n_hidden=512,
                                 n_rnn_layer=2, dropout=0.4)
        
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
        
        # constants
        self.num_past_frames = model_config.past_frames
        self.num_future_frames = model_config.future_frames
        self.num_total_frames = self.num_past_frames + self.num_future_frames

    def reset(self):
        self.rnn_state = None
        self.imu = None
        self.last_root_pos = torch.zeros(3).to(self.C.device)

    def forward(self, batch, input_lengths=None):
        # forward joint model
        vel, _, _ = self.vel(batch, input_lengths)

        return vel

    def predict_RNN(self, input, init_vel):
        input_lengths = input.shape[0]
        init_vel = init_vel.view(1, self.output_size) 
        
        input = (input.unsqueeze(0), init_vel)
        
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
    
    def shared_step(self, batch):
        # unpack data
        inputs, outputs = batch
        imu_inputs, input_lengths = inputs
        outputs, _ = outputs

        # target joints
        joints = outputs['joints']
        B, S, _, _ = joints.shape

        # target velocity
        target_vel = outputs['vels'][:, :, amass.vel_joint].view(B, S, -1)
        
        # predict joint velocity
        # change: add init vel
        imu_inputs = (imu_inputs, target_vel[:, 0])
        
        pred_vel, _, _ = self.vel(imu_inputs, input_lengths)
        loss = self.compute_loss(pred_vel, target_vel)

        return loss

    def compute_loss(self, pred_vel, gt_vel):
        loss = sum(self.compute_vel_loss(pred_vel, gt_vel, i) for i in [1, 3, 9, 27])
        return loss

    def compute_vel_loss(self, pred_vel, gt_vel, n=1):
        T = pred_vel.shape[1]
        loss = 0.0

        for m in range(0, T//n):
            end = min(n*m+n, T)
            loss += self.loss(pred_vel[:, m*n:end, :], gt_vel[:, m*n:end, :])

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
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