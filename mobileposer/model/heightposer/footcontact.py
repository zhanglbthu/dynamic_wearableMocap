import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import lightning as L
import numpy as np

from config import *
import articulate as art
from model.base_model.rnn import RNN


class FootContact(L.LightningModule):
    """
    Inputs: N IMUs.
    Outputs: Foot Contact Probability ([s_lfoot, s_rfoot]).
    """

    def __init__(self, combo_id = 'lw_rp_h'):
        super().__init__()
        
        # constants
        self.C = model_config
        self.hypers = train_hypers

        # input dimensions
        imu_set = amass.combos_mine[combo_id]
        imu_num = len(imu_set)
        self.imu_nums = imu_num
        imu_input_dim = imu_num * 12 + 1

        self.input_dim = self.C.n_output_joints*3 + imu_input_dim 

        # model definitions
        self.footcontact = RNN(self.input_dim, 2, 64, bidirectional=False)  # foot-ground probability model

        # log input and output dimensions
        print(f"Input dimensions: {self.input_dim}")
        print(f"Output dimensions: 2")
        
        # loss function (binary cross-entropy)
        self.loss = nn.BCEWithLogitsLoss()

        # track stats
        self.validation_step_loss = []
        self.training_step_loss = []
        self.save_hyperparameters()

    def forward(self, batch, input_lengths=None):
        # forward foot contact model
        foot_contact, _, _ = self.footcontact(batch, input_lengths)
        return foot_contact

    def input_process(self, inputs):
        # process input
        _, _, input_dim = inputs.shape
        inputs = inputs.view(-1, input_dim)
        
        imu_inputs = inputs[:, :12 * self.imu_nums]
        h_inputs = inputs[:, -1:].view(-1, 1)
        
        inputs = torch.cat([imu_inputs, h_inputs], dim=-1)
        
        return inputs

    def shared_step(self, batch):
        # unpack data
        inputs, outputs = batch
        imu_inputs, input_lengths = inputs
        outputs, _ = outputs

        # target joints
        joints = outputs['joints']
        B, S, _, _ = joints.shape
        target_joints = joints.view(joints.shape[0], joints.shape[1], -1)

        # ground-truth foot contacts
        foot_contacts = outputs['foot_contacts']
        
        # process input
        imu_inputs = self.input_process(imu_inputs).view(B, S, -1)
        
        # predict foot-ground contact probability
        tran_input = torch.cat((target_joints, imu_inputs), dim=-1)
        pred_contacts, _, _ = self.footcontact(tran_input, input_lengths)
        loss = self.loss(pred_contacts, foot_contacts)

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

    def on_fit_start(self):
        self.bodymodel = art.model.ParametricModel(paths.smpl_file, device=self.device)
    
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