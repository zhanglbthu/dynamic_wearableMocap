import time
import torch
from pygame.time import Clock

import articulate as art
import os
from config import *
from articulate.utils.noitom import *
from articulate.utils.unity import MotionViewer

from articulate.utils.wearable import WearableSensorSet
from articulate.utils.pygame import StreamingDataViewer
from articulate.utils.bullet.view_rotation_np import RotationViewer
from auxiliary import calibrate_q, quaternion_inverse

from utils.model_utils import load_mobileposer_model, load_heightposer_model
import numpy as np
import matplotlib
from argparse import ArgumentParser
from collections import deque

from TicOperator import *
from my_model import *
import keyboard
import datetime

colors = matplotlib.colormaps['tab10'].colors
body_model = art.ParametricModel(paths.smpl_file, device='cuda')

class IMUSet:
    g = 9.8

    def __init__(self, udp_port=7777):
        app = MCPApplication()
        settings = MCPSettings()
        settings.set_udp(udp_port)
        settings.set_calc_data()
        app.set_settings(settings)
        app.open()
        time.sleep(0.5)

        sensors = [None for _ in range(6)]
        evts = []
        print('Waiting for sensors...')
        while len(evts) == 0:
            evts = app.poll_next_event()
            for evt in evts:  
                assert evt.event_type == MCPEventType.SensorModulesUpdated
                sensor_module_handle = evt.event_data.sensor_module_data.sensor_module_handle
                sensor_module = MCPSensorModule(sensor_module_handle)
                sensors[sensor_module.get_id() - 1] = sensor_module

        print('find %d sensors' % len([_ for _ in sensors if _ is not None]))
        self.app = app
        self.sensors = sensors
        self.t = 0

    def get(self):
        evts = self.app.poll_next_event()
        if len(evts) > 0:
            self.t = evts[0].timestamp
        q, a = [], []
        for sensor in self.sensors:
            q.append(sensor.get_posture())
            a.append(sensor.get_accelerated_velocity())

        # assuming g is positive (= 9.8), we need to change left-handed system to right-handed by reversing axis x, y, z
        R = art.math.quaternion_to_rotation_matrix(torch.tensor(q))  # rotation is not changed
        a = -torch.tensor(a) / 1000 * self.g                         # acceleration is reversed
        a = R.bmm(a.unsqueeze(-1)).squeeze(-1) + torch.tensor([0., 0., self.g])   # calculate global free acceleration
        return self.t, R, a

    def clear(self):
        pass

def tpose_calibration(n_calibration):
    print('Calibrating Wearable Devices T-pose...')
    c = input('Used cached RMI? [y]/n    (If you choose no, put imu 1 straight (x = Right, y = Forward, z = Down, Left-handed).')
    if c == 'n' or c == 'N':
        imu_set.clear()
        RSI = imu_set.get()[1][0].view(3, 3).t()
        RMI = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0.]]).mm(RSI)
        torch.save(RMI, os.path.join(paths.temp_dir, 'RMI.pt'))
    else:
        RMI = torch.load(os.path.join(paths.temp_dir, 'RMI.pt'))
    print(RMI)

    input('Stand straight in T-pose and press enter. The calibration will begin in 3 seconds')
    time.sleep(3)
    
    RIS = torch.eye(3).repeat(6, 1, 1)
    
    for i in range(n_calibration):
        qCO_sensor = torch.tensor(sensor_set.get()[i].orientation).float()
        qIS_sensor = art.math.quaternion_product(art.math.quaternion_product(qIC_list[i], qCO_sensor), qOS_list[i])
        RIS_sensor = art.math.quaternion_to_rotation_matrix(qIS_sensor)
        
        if i == 0:
            index = 3
        else:
            index = 0
            
        RIS[index, :, :] = RIS_sensor[0, :, :]
    
    RSB = RMI.matmul(RIS).transpose(1, 2).matmul(torch.eye(3))
    return RMI, RSB

def tpose_calibration_noitom(imu_set):
     print('Calibrating Noitom Devices T-pose...')
     c = input('Used cached RMI? [y]/n    (If you choose no, put imu 1 straight (x = Right, y = Forward, z = Down, Left-handed).')
     if c == 'n' or c == 'N':
         imu_set.clear()
         RSI_gt = imu_set.get()[1][0].view(3, 3).t()
         RMI_gt = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0.]]).mm(RSI_gt)
         torch.save(RMI_gt, os.path.join(paths.temp_dir, 'RMI.pt'))
     else:
         RMI_gt = torch.load(os.path.join(paths.temp_dir, 'RMI.pt'))
    #  RMI_gt = torch.tensor([[-1., 0., 0.], [0., 0., -1.], [0., -1., 0.]]).float()
     print(RMI_gt)
 
     input('Stand straight in T-pose and press enter. The calibration will begin in 3 seconds')
     time.sleep(3)
     imu_set.clear()
     RIS_gt = imu_set.get()[1]

     RSB_gt = RMI_gt.matmul(RIS_gt).transpose(1, 2).matmul(torch.eye(3))
     return RMI_gt, RSB_gt

def align_sensor_noitom(imu_set, sensor_set, n_calibration):
     r"""Align noitom and sensor imu data"""
     print('Rotate the sensor & imu together.')
     qIC_list, qOS_list = [], []
 
     for i in range(n_calibration):
         qIS, qCO = [], []
         while len(qIS) < 1:
             imu_set.app.poll_next_event()
             sensor_data = sensor_set.get()
             if not 0 in sensor_data:
                 continue
 
             qIS.append(torch.tensor(imu_set.sensors[0].get_posture()).float()) # noitom
             qCO.append(torch.tensor(sensor_data[i].orientation).float()) # wearable sensor
             print('\rCalibrating... (%d/%d)' % (i, n_calibration), end='')
 
         qCI, qSO = calibrate_q(torch.stack(qIS), torch.stack(qCO))
         print('\tfinished\nqCI:', qCI, '\tqSO:', qSO)
         qIC_list.append(quaternion_inverse(qCI))
         qOS_list.append(quaternion_inverse(qSO))
     return qIC_list, qOS_list

def align_sensor(imu_set, sensor_set, n_calibration, using_cached=False):
    r"""Align noitom and sensor imu data"""
    print('Rotate the sensor & imu together.')
    qIC_list, qOS_list = [], []
 
    for i in range(n_calibration):
        qIS, qCO = [], []
        if using_cached:
            if i == 0:
                qCI = torch.tensor([0.0137, -0.7311, -0.6820, -0.0151]).float()
            elif i == 1:
                qCI = torch.tensor([-0.0125, -0.7841, -0.6205,  0.0041]).float()
            qSO = torch.tensor([1., -0., -0., -0.]).float() 
        else:
            while len(qIS) < 1:
                imu_set.app.poll_next_event()
                sensor_data = sensor_set.get()
                if not 0 in sensor_data:
                    continue
    
                qIS.append(torch.tensor(imu_set.sensors[3].get_posture()).float()) # noitom
                qCO.append(torch.tensor(sensor_data[i].orientation).float()) # wearable sensor
                print('\rCalibrating... (%d/%d)' % (i, n_calibration), end='')
    
            qCI, qSO = calibrate_q(torch.stack(qIS), torch.stack(qCO))
        print('\tfinished\nqCI:', qCI, '\tqSO:', qSO)
        qIC_list.append(quaternion_inverse(qCI))
        qOS_list.append(quaternion_inverse(qSO))
    return qIC_list, qOS_list

if __name__ == '__main__':
    parser = ArgumentParser()
    os.makedirs(paths.temp_dir, exist_ok=True)
    os.makedirs(paths.live_record_dir, exist_ok=True)
    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--type', type=str, default='RIS')
    parser.add_argument('--rviewer', action='store_true')
    args = parser.parse_args()
    
    print('using type:', args.type)
    device = torch.device("cuda")
    sensor_set = WearableSensorSet()
    imu_set = IMUSet()
    clock = Clock()
    
    n_calibration = 1

    qIC_list, qOS_list = align_sensor(imu_set, sensor_set, n_calibration, using_cached=True)
    RMI, RSB = tpose_calibration(n_calibration)

    # add ground truth readings
    RMI_gt, RSB_gt = tpose_calibration_noitom(imu_set=imu_set)
    
    idx = 0
    sviewer = StreamingDataViewer(3, y_range=(-180, 180), window_length=200, names=['Y', 'Z', 'X']); sviewer.connect()
    # sviewer = StreamingDataViewer(2, y_range=(-180, 180), window_length=60, names=['Y_real', 'Y_gt']); sviewer.connect()
    if args.rviewer:
        rviewer = RotationViewer(2, order='wxyz'); rviewer.connect()
        
    history_gt = deque(maxlen=100)
    N = 2
    
    accs, oris = [], []
    accs_gt, oris_gt = [], []
    
    while True:
        clock.tick(30)
        
        RIS = torch.eye(3).repeat(6, 1, 1)
        aI = torch.zeros(6, 3)

        sensor_data = sensor_set.get()
        combo = [0, 3]
        # gt readings
        tframe, RIS_gt, aI_gt = imu_set.get()
        history_gt.append(RIS_gt.clone())
        
        for i in range(n_calibration):
            qCO_sensor = torch.tensor(sensor_data[i].orientation).float()
            aSS_sensor = torch.tensor(sensor_data[i].raw_acceleration).float() # [3]
            qIS_sensor = art.math.quaternion_product(art.math.quaternion_product(qIC_list[i], qCO_sensor), qOS_list[i])
            RIS_sensor = art.math.quaternion_to_rotation_matrix(qIS_sensor) # [1, 3, 3]
            
            aIS_sensor = RIS_sensor.squeeze(0).mm( - aSS_sensor.unsqueeze(-1)).squeeze(-1) + torch.tensor([0., 0., 9.8]) 
            
            index = 3 if i == 0 else 0

            RIS[index, :, :] = RIS_sensor[0, :, :]

            aI[index, :] = aIS_sensor

        if len(history_gt) > N:
            RIS_gt = history_gt[-N]
        else:
            RIS_gt = RIS_gt

        RMB_gt = RMI_gt.matmul(RIS_gt).matmul(RSB_gt)[combo].to(device)
        RIB_gt = RIS_gt.matmul(RSB_gt)[combo].to(device)
        aM_gt = aI_gt.mm(RMI_gt.t())[combo].to(device)
        
        oris_gt.append(RMB_gt)
        accs_gt.append(aM_gt)

        RMB = RMI.matmul(RIS).matmul(RSB)[combo].to(device)
        RIB = RIS.matmul(RSB)[combo].to(device)
        aM = aI.mm(RMI.t())[combo].to(device)

        # # calibrate acc and ori
        # RMB, aM = ts.run_livedemo_tic(RMB, aM, trigger_t=1, idx=idx)

        RMB = RMB.view(imu_num, 3, 3)
        aM = aM.view(imu_num, 3)
        RMB_gt = RMB_gt.view(imu_num, 3, 3)
        aM_gt = aM_gt.view(imu_num, 3)
        
        oris.append(RMB)
        accs.append(aM)
        
        if args.type == 'RIS':
            delta_R = RIS[3].t() @ RIS_gt[3]
            # delta_R = RIS[3]
        elif args.type == 'RMB':
            delta_R = RMB[1].t() @ RMB_gt[1]
        elif args.type == 'RIB':
            delta_R = RIB[1].t() @ RIB_gt[1]
        if args.rviewer:
            r_list = []
            if args.type == 'RIS':
                RIS_axis, RIS_gt_axis = art.math.rotation_matrix_to_axis_angle(RIS).view(-1, 3), art.math.rotation_matrix_to_axis_angle(RIS_gt).view(-1, 3)
                RIS_q, RIS_gt_q = art.math.axis_angle_to_quaternion(RIS_axis).view(-1, 4), art.math.axis_angle_to_quaternion(RIS_gt_axis).view(-1, 4)
                r_list.append(RIS_q[3].cpu().numpy())
                r_list.append(RIS_gt_q[3].cpu().numpy())
            
            if args.type == 'RIB':
                RIB_axis, RIB_gt_axis = art.math.rotation_matrix_to_axis_angle(RIB).view(-1, 3), art.math.rotation_matrix_to_axis_angle(RIB_gt).view(-1, 3)
                RIB_q, RIB_gt_q = art.math.axis_angle_to_quaternion(RIB_axis).view(-1, 4), art.math.axis_angle_to_quaternion(RIB_gt_axis).view(-1, 4)
                r_list.append(RIB_q[1].cpu().numpy())
                r_list.append(RIB_gt_q[1].cpu().numpy())
            if args.type == 'RMB':
                RMB_axis, RMB_gt_axis = art.math.rotation_matrix_to_axis_angle(RMB).view(-1, 3), art.math.rotation_matrix_to_axis_angle(RMB_gt).view(-1, 3)
                RMB_q, RMB_gt_q = art.math.axis_angle_to_quaternion(RMB_axis).view(-1, 4), art.math.axis_angle_to_quaternion(RMB_gt_axis).view(-1, 4)
                r_list.append(RMB_q[1].cpu().numpy())
                r_list.append(RMB_gt_q[1].cpu().numpy())
            
            rviewer.update_all(r_list)

        rot_axis = art.math.rotation_matrix_to_axis_angle(delta_R).view(-1, 3)
        rot_euler = art.math.rotation_matrix_to_euler_angle(delta_R, seq='YZX').view(3)
        rot_euler = rot_euler * 180 / np.pi
        
        rot_axis_real = art.math.rotation_matrix_to_axis_angle(RIS).view(-1, 3)
        rot_euler_real = art.math.rotation_matrix_to_euler_angle(RIS, seq='YZX').view(-1, 3)
        rot_euler_real = rot_euler_real * 180 / np.pi
        
        rot_axis_gt = art.math.rotation_matrix_to_axis_angle(RIS_gt).view(-1, 3)
        rot_euler_gt = art.math.rotation_matrix_to_euler_angle(RIS_gt, seq='YZX').view(-1, 3)
        rot_euler_gt = rot_euler_gt * 180 / np.pi
    
        sviewer.plot(rot_euler)
        
        # # # plot rot_euler_real and rot_euler_gt的Y
        # sviewer.plot([rot_euler_real[3][2].cpu().numpy(), rot_euler_gt[3][2].cpu().numpy()])
        print('\r', clock.get_fps(), end='')
        
        if keyboard.is_pressed('q'):
                break
        
    accs = torch.stack(accs)
    oris = torch.stack(oris)
    accs_gt = torch.stack(accs_gt)
    oris_gt = torch.stack(oris_gt)
    
    print(f"accs: {accs.shape}, oris: {oris.shape}, accs_gt: {accs_gt.shape}, oris_gt: {oris_gt.shape}")
    
    print('Frames: %d' % accs.shape[0])
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    data_filename = f"{args.name}_{timestamp}.pt"
    
    torch.save({'acc': accs,       # [N,  2,  3]
                'ori': oris,       # [N,  2,  3,  3]
                'acc_gt': accs_gt, # [N,  2, 3]
                'ori_gt': oris_gt  # [N,  2,  3,  3]
                }, os.path.join(paths.live_record_dir, data_filename))
    
    print('\rFinish.')
        