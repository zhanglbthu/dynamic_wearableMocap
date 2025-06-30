import os
import torch
import torch.nn as nn
import lightning as L
from torch.nn import functional as F
import numpy as np

from config import *
from utils.model_utils import reduced_pose_to_full
import articulate as art
from model.base_model.rnn import RNN

class Poser(L.LightningModule):
    """
    Inputs: N IMUs.
    Outputs: SMPL Pose Parameters (as 6D Rotations).
    """
    def __init__(self, finetune: bool=False, combo_id: str="lw_rp_h"):
        super().__init__()
        
        # constants
        self.C = model_config
        self.finetune = finetune
        self.hypers = finetune_hypers if finetune else train_hypers
        imu_set = amass.combos_mine[combo_id]
        imu_num = len(imu_set)
        
        # input dimensions
        imu_input_dim = imu_num * 12

        self.input_dim = imu_input_dim
        
        # output dimensions
        self.output_dim = len(joint_set.full)*6

        # model definitions
        self.pose = RNN(self.input_dim, n_output=self.output_dim, n_hidden=256) # pose estimation model
        
        # log input and output dimensions in one line
        print("Poser", f"Input dimensions: {self.input_dim}", f"Output dimensions: {self.output_dim}")
        
        # loss function
        self.loss = nn.MSELoss()
        self.use_pos_loss = True

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

    def forward(self, batch, input_lengths=None):
        # forward the pose prediction model
        pred_pose, _, _ = self.pose(batch, input_lengths)
        return pred_pose

    def shared_step(self, batch):
        # unpack data
        inputs, outputs = batch
        imu_inputs, input_lengths = inputs
        outputs, output_lengths = outputs

        # target pose
        target_pose = outputs['poses'] # [batch_size, window_length, 144] 
        B, S, _ = target_pose.shape

        # target joints
        joints = outputs['joints']
        target_joints = joints.view(B, S, -1)

        # predict pose
        pose_input = imu_inputs
        pose_p = self(pose_input, input_lengths)
        
        # convert target pose to local pose
        target_pose = art.math.r6d_to_rotation_matrix(target_pose).view(-1, 24, 3, 3) # r6d to rotation matrix
        target_pose = self.global_to_local_pose(target_pose)                          # global to local
        
        target_pose = art.math.rotation_matrix_to_r6d(target_pose)                    # rotation matrix to r6d

        # compute pose loss
        pose_t = target_pose.view(B, S, 24, 6)[:, :, joint_set.full].view(B, S, -1)
        loss = self.loss(pose_p, pose_t)

        # joint position loss
        if self.use_pos_loss:
            # full_pose_p = self._reduced_global_to_full(pose_p)
            full_pose_p = art.math.r6d_to_rotation_matrix(pose_p).view(-1, 24, 3, 3)
            joints_p = self.bodymodel.forward_kinematics(pose=full_pose_p.view(-1, 216))[1].view(B, S, -1)
            loss += self.loss(joints_p, target_joints)

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