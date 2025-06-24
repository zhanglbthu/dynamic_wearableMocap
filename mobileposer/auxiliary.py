__all__ = ['set_pose', 'smpl_to_rbdl', 'rbdl_to_smpl', 'normalize_and_concat', 'print_title', 'Body', 'smpl_to_rbdl_data']


import enum
import torch
import numpy as np
import pybullet as p
from articulate.math import rotation_matrix_to_euler_angle_np, euler_angle_to_rotation_matrix_np, euler_convert_np, \
    normalize_angle

import cv2
import threading
from queue import Queue
import articulate as art
import cv2.aruco as aruco
import time

_smpl_to_rbdl = [0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 28, 29, 3, 4, 5, 12, 13, 14, 21, 22, 23, 30, 31, 32, 6, 7, 8,
                 15, 16, 17, 24, 25, 26, 36, 37, 38, 45, 46, 47, 51, 52, 53, 57, 58, 59, 63, 64, 65, 39, 40, 41,
                 48, 49, 50, 54, 55, 56, 60, 61, 62, 66, 67, 68, 33, 34, 35, 42, 43, 44]
_rbdl_to_smpl = [0, 1, 2, 12, 13, 14, 24, 25, 26, 3, 4, 5, 15, 16, 17, 27, 28, 29, 6, 7, 8, 18, 19, 20, 30, 31, 32,
                 9, 10, 11, 21, 22, 23, 63, 64, 65, 33, 34, 35, 48, 49, 50, 66, 67, 68, 36, 37, 38, 51, 52, 53, 39,
                 40, 41, 54, 55, 56, 42, 43, 44, 57, 58, 59, 45, 46, 47, 60, 61, 62]
_rbdl_to_bullet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                   27, 28, 29, 30, 31, 32, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 33, 34, 35,
                   36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 63, 64, 65, 66, 67, 68]
smpl_to_rbdl_data = _smpl_to_rbdl


def set_pose(id_robot, q):
    r"""
    Set the robot configuration.
    """
    p.resetJointStatesMultiDof(id_robot, list(range(1, p.getNumJoints(id_robot))), q[6:][_rbdl_to_bullet].reshape(-1, 1))
    glb_rot = p.getQuaternionFromEuler(euler_convert_np(q[3:6], 'zyx', 'xyz')[[2, 1, 0]])
    p.resetBasePositionAndOrientation(id_robot, q[:3], glb_rot)


def smpl_to_rbdl(poses, trans):
    r"""
    Convert smpl poses and translations to robot configuration q. (numpy, batch)

    :param poses: Array that can reshape to [n, 24, 3, 3].
    :param trans: Array that can reshape to [n, 3].
    :return: Ndarray in shape [n, 75] (3 root position + 72 joint rotation).
    """
    poses = np.array(poses).reshape(-1, 24, 3, 3)
    trans = np.array(trans).reshape(-1, 3)
    euler_poses = rotation_matrix_to_euler_angle_np(poses[:, 1:], 'XYZ').reshape(-1, 69)
    euler_glbrots = rotation_matrix_to_euler_angle_np(poses[:, :1], 'xyz').reshape(-1, 3)
    euler_glbrots = euler_convert_np(euler_glbrots[:, [2, 1, 0]], 'xyz', 'zyx')
    qs = np.concatenate((trans, euler_glbrots, euler_poses[:, _smpl_to_rbdl]), axis=1)
    qs[:, 3:] = normalize_angle(qs[:, 3:])
    return qs


def rbdl_to_smpl(qs):
    r"""
    Convert robot configuration q to smpl poses and translations. (numpy, batch)

    :param qs: Ndarray that can reshape to [n, 75] (3 root position + 72 joint rotation).
    :return: Poses ndarray in shape [n, 24, 3, 3] and translation ndarray in shape [n, 3].
    """
    qs = qs.reshape(-1, 75)
    trans, euler_glbrots, euler_poses = qs[:, :3], qs[:, 3:6], qs[:, 6:][:, _rbdl_to_smpl]
    euler_glbrots = euler_convert_np(euler_glbrots, 'zyx', 'xyz')[:, [2, 1, 0]]
    glbrots = euler_angle_to_rotation_matrix_np(euler_glbrots, 'xyz').reshape(-1, 1, 3, 3)
    poses = euler_angle_to_rotation_matrix_np(euler_poses, 'XYZ').reshape(-1, 23, 3, 3)
    poses = np.concatenate((glbrots, poses), axis=1)
    return poses, trans


def concat(glb_acc, glb_rot, imu_set):
    imu_nums = len(imu_set)
    acc_selected = glb_acc[:, imu_set, :]
    rot_selected = glb_rot[:, imu_set, :, :]
    
    # glb_acc = glb_acc.view(-1, 6, 3)
    # glb_rot = glb_rot.view(-1, 6, 3, 3)
    glb_acc = acc_selected.view(-1, imu_nums, 3)
    glb_rot = rot_selected.view(-1, imu_nums, 3, 3)
    
    data = torch.cat((glb_acc.flatten(1), glb_rot.flatten(1)), dim=1)
    return data

def normalize_and_concat(glb_acc, glb_rot):
    glb_acc = glb_acc.view(-1, 6, 3)
    glb_rot = glb_rot.view(-1, 6, 3, 3)
    acc = torch.cat((glb_acc[:, :5] - glb_acc[:, 5:], glb_acc[:, 5:]), dim=1).bmm(glb_rot[:, -1])
    ori = torch.cat((glb_rot[:, 5:].transpose(2, 3).matmul(glb_rot[:, :5]), glb_rot[:, 5:]), dim=1)
    data = torch.cat((acc.flatten(1), ori.flatten(1)), dim=1)
    return data


def print_title(s):
    print('============ %s ============' % s)


class Body(enum.Enum):
    r"""
    Prefix L = left; Prefix R = right.
    """
    ROOT = 2
    PELVIS = 2
    SPINE = 2
    LHIP = 5
    RHIP = 17
    SPINE1 = 29
    LKNEE = 8
    RKNEE = 20
    SPINE2 = 32
    LANKLE = 11
    RANKLE = 23
    SPINE3 = 35
    LFOOT = 14
    RFOOT = 26
    NECK = 68
    LCLAVICLE = 38
    RCLAVICLE = 53
    HEAD = 71
    LSHOULDER = 41
    RSHOULDER = 56
    LELBOW = 44
    RELBOW = 59
    LWRIST = 47
    RWRIST = 62
    LHAND = 50
    RHAND = 65


__all__ = ['ArucoCamera', 'calibrate_q']


class ArucoCamera:
    local_imu_pos = torch.tensor([0.15, 0., 0.])
    cameraMatrix = np.array([[615.81579672, 0.,           637.41012256],
                             [0.,           615.31695619, 370.66354516],
                             [0.,           0.,           1.          ]])  # camera intrinsic 1280x720
    distCoeffs = np.array([0.09684284, -0.1012923, -0.00026771, -0.00030583, 0.05960547])  # camera distortion
    objPoints = np.array([[-0.1/2,  0.1/2, 0],
                          [ 0.1/2,  0.1/2, 0],
                          [ 0.1/2, -0.1/2, 0],
                          [-0.1/2, -0.1/2, 0]])  # in frame at the center of the marker

    def __init__(self, cam_idx=0):
        self.cap = cv2.VideoCapture(cam_idx)  # , cv2.CAP_DSHOW
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ARUCO_PARAMETERS = aruco.DetectorParameters()
        ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)  # 4X4, id 0~49
        self.tframe = None
        self.pcs = None
        self.detector = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMETERS)
        self.im_queue = Queue()
        self.thread = threading.Thread(target=self._run)
        self.thread.setDaemon(True)
        self.thread.start()

    def _run(self):
        while True:
            ret, frame = self.cap.read()
            # tframe = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            tframe = time.time()
            self.im_queue.put((tframe, frame))

    def get(self):
        qco, pcs, vcs = None, None, None
        if not self.im_queue.empty():
            tframe, frame = self.im_queue.get()
            corners, ids, rejectedCandidates = self.detector.detectMarkers(frame)
            if ids is not None:
                aruco.drawDetectedMarkers(frame, corners, ids)
                _, rvec, tvec = cv2.solvePnP(self.objPoints, corners[0], self.cameraMatrix, self.distCoeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                cv2.drawFrameAxes(frame, self.cameraMatrix, self.distCoeffs, rvec, tvec, 0.1)
                Rco = art.math.axis_angle_to_rotation_matrix(torch.from_numpy(rvec).float())[0]
                pco = torch.from_numpy(tvec).float().view(3)
                pcs = pco + Rco.mm(self.local_imu_pos.view(3, 1)).view(3)
                qco = art.math.axis_angle_to_quaternion(art.math.rotation_matrix_to_axis_angle(Rco)).view(-1)
                if self.tframe is not None and tframe - self.tframe < 0.1:
                    vcs = (pcs - self.pcs) / (tframe - self.tframe)
                self.pcs = pcs
                self.tframe = tframe
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
        return qco, pcs, vcs


def calibrate_q(qIS, qCO):
    r"""
    minimize || qIC qCO qOS - qIS || assuming qOS approximates 1.
    """
    qOS = torch.tensor([1., 0, 0, 0], requires_grad=True)
    qIC = art.math.normalize_tensor(art.math.quaternion_product(qIS, art.math.quaternion_inverse(qCO)).mean(dim=0))
    qIC.requires_grad_(True)
    optim = torch.optim.Adam([qOS, qIC])
    for i in range(400):
        qIS_est = art.math.quaternion_product(art.math.quaternion_product(qIC.expand_as(qCO), qCO), qOS.expand_as(qCO))
        delta_q = art.math.quaternion_product(qIS_est, art.math.quaternion_inverse(qIS))
        loss = delta_q[:, 1:].pow(2).mean() + (delta_q[:, 0].abs() - 1).pow(2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        with torch.no_grad():
            qOS.data = art.math.normalize_tensor(qOS)
            qIC.data = art.math.normalize_tensor(qIC)
        # if i % 50 == 0:
        #     print(i, loss.item())
    return art.math.quaternion_inverse(qIC.detach()).clone(), art.math.quaternion_inverse(qOS.detach()).clone()

def quaternion_inverse(q):
    
    w, x, y, z = q
    
    q_conjugate = np.array([w, -x, -y, -z])
    
    norm_sq = w**2 + x**2 + y**2 + z**2
    
    q_inv = q_conjugate / norm_sq
    
    return q_inv