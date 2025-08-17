import time
import torch
from pygame.time import Clock

import articulate as art
import os
from config import *
from articulate.utils.noitom import *
from articulate.utils.unity import MotionViewer

from articulate.utils.wearable import WearableSensorSet
from auxiliary import calibrate_q, quaternion_inverse

from utils.model_utils import load_mobileposer_model, load_heightposer_model
import numpy as np
import matplotlib
from argparse import ArgumentParser
import keyboard
import datetime

from TicOperator import *
from my_model import *

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
    print('Calibrating T-pose...')
    RMI = torch.tensor([[-1., 0., 0.], [0., 0., -1.], [0., -1., 0.]]).float()
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

def align_sensor(sensor_set, n_calibration):
    r"""Align noitom and sensor imu data"""
    print('Rotate the sensor & imu together.')
    qIC_list, qOS_list = [], []
    
    for i in range(n_calibration):
        # qIS, qCO = [], []

        # qIS.append(torch.tensor([0., 0., 1.0, 0.]).float()) # noitom
        
        # # align wearable sensor
        # while len(qCO) < 1:
        #     sensor_data = sensor_set.get()
        #     if not 0 in sensor_data or not 1 in sensor_data:
        #         continue
        #     qCO.append(torch.tensor(sensor_data[i].orientation).float()) # wearable sensor
        #     print('\rCalibrating... (%d/%d)' % (i, n_calibration), end='')
            
        # qCI, qSO = calibrate_q(torch.stack(qIS), torch.stack(qCO))
        # print('\tfinished\nqCI:', qCI, '\tqSO:', qSO)
        
        # using cached qCI and qSO
        if i == 0:
            qCI = torch.tensor([-0.0050, -0.7486, -0.6630,  0.0054]).float()
        elif i == 1:
            qCI = torch.tensor([-0.0125, -0.7841, -0.6205,  0.0041]).float()
        qSO = torch.tensor([1., -0., -0., -0.]).float()  
        qIC_list.append(quaternion_inverse(qCI))
        qOS_list.append(quaternion_inverse(qSO))
    return qIC_list, qOS_list

def tpose_calibration_noitom(imu_set):
     print('Calibrating T-pose...')
     c = input('Used cached RMI? [y]/n    (If you choose no, put imu 1 straight (x = Right, y = Forward, z = Down, Left-handed).')
     if c == 'n' or c == 'N':
         imu_set.clear()
         RSI_gt = imu_set.get()[1][0].view(3, 3).t()
         RMI_gt = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0.]]).mm(RSI_gt)
         torch.save(RMI_gt, os.path.join(paths.temp_dir, 'RMI.pt'))
     else:
         RMI_gt = torch.load(os.path.join(paths.temp_dir, 'RMI.pt'))
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

if __name__ == '__main__':
    parser = ArgumentParser()
    os.makedirs(paths.temp_dir, exist_ok=True)
    os.makedirs(paths.live_record_dir, exist_ok=True)
    parser.add_argument('--name', type=str, default='default')
    args = parser.parse_args()
    
    device = torch.device("cuda")
    sensor_set = WearableSensorSet()
    clock = Clock()
    
    n_calibration = 2
    
    # # set baseline network (heightposer_version)
    # ckpt_path = "data/checkpoints/heightposer_RNNwInit/lw_rp/base_model.pth"
    # net = load_heightposer_model(ckpt_path, combo_id=model_config.combo_id)
    # print('HeightPoser model loaded.')
    
    # set mobileposer network
    ckpt_path = "data/checkpoints/mobileposer/lw_rp/base_model.pth"
    net = load_mobileposer_model(ckpt_path, combo_id=model_config.combo_id)
    print('Mobileposer model loaded.')
    
    # set calibrator model
    tic = TIC(stack=3, n_input=imu_num * (3 + 3 * 3), n_output=imu_num * 6)
    tic.restore("data/checkpoints/calibrator/TIC/TIC_20.pth")
    tic = tic.to(device).eval()
    print('TIC model loaded.')
    
    # # set LSTM calibrator model
    # lstmic = LSTMIC(n_input=imu_num * (3 + 3 * 3), n_output=imu_num * 6)
    # lstmic.restore("data/checkpoints/calibrator/LSTMIC_frame/LSTMIC_frame_20.pth")
    # lstmic = lstmic.to(device).eval()
    # print('LSTMIC model loaded.')
    
    # set operator model
    ts = TicOperator(TIC_network=tic, imu_num=imu_num, data_frame_rate=30)
    ts.reset()
    print('TicOperator model loaded.')

    qIC_list, qOS_list = align_sensor(sensor_set=sensor_set, n_calibration=2)
    RMI, RSB = tpose_calibration(n_calibration)

    # # add ground truth readings
    # imu_set = IMUSet()
    # RMI_gt, RSB_gt = tpose_calibration_noitom(imu_set=imu_set)

    accs, oris = [], []
    accs_gt, oris_gt = [], []
    poses, trans = [], []
    
    net.eval()
    
    idx = 0

    with torch.no_grad(), MotionViewer(1, overlap=False) as viewer:
        while True:
            clock.tick(30)
            viewer.clear_line(render=False)
            viewer.clear_point(render=False)
            viewer.clear_terrian(render=False)
            
            RIS = torch.eye(3).repeat(6, 1, 1)
            aI = torch.zeros(6, 3)

            sensor_data = sensor_set.get()
            combo = [0, 3]
            # # # gt readings
            # tframe, RIS_gt, aI_gt = imu_set.get()
            # RMB_gt = RMI_gt.matmul(RIS_gt).matmul(RSB_gt)[combo].to(device)
            # aM_gt = aI_gt.mm(RMI_gt.t())[combo].to(device)

            # oris_gt.append(RMB_gt)
            # accs_gt.append(aM_gt)
            
            pressures = []
            
            for i in range(n_calibration):
                qCO_sensor = torch.tensor(sensor_data[i].orientation).float()
                aSS_sensor = torch.tensor(sensor_data[i].raw_acceleration).float() # [3]
                qIS_sensor = art.math.quaternion_product(art.math.quaternion_product(qIC_list[i], qCO_sensor), qOS_list[i])
                RIS_sensor = art.math.quaternion_to_rotation_matrix(qIS_sensor) # [1, 3, 3]
                
                aIS_sensor = RIS_sensor.squeeze(0).mm( - aSS_sensor.unsqueeze(-1)).squeeze(-1) + torch.tensor([0., 0., 9.8]) 
                
                index = 3 if i == 0 else 0

                RIS[index, :, :] = RIS_sensor[0, :, :]

                aI[index, :] = aIS_sensor
            
            RMB = RMI.matmul(RIS).matmul(RSB)[combo].to(device)
            aM = aI.mm(RMI.t())[combo].to(device)

            oris.append(RMB)
            accs.append(aM)

            # calibrate acc and ori
            RMB, aM = ts.run_livedemo_tic(RMB, aM, trigger_t=1, idx=idx)

            RMB = RMB.view(imu_num, 3, 3)
            aM = aM.view(imu_num, 3)
            
            aM = aM / amass.acc_scale
            input = torch.cat([aM.flatten(), RMB.flatten()], dim=0).to("cuda")
            
            pose = net.forward_frame(input)
            
            poses.append(pose)
            
            pose = pose.cpu().numpy()      
            
            zero_tran = np.array([0, 0, 0])  
            viewer.update_all([pose], [zero_tran], render=False)
            viewer.render()
            
            idx += 1
            
            print('\r', clock.get_fps(), end='')
            
            if keyboard.is_pressed('q'):
                break
        
    accs = torch.stack(accs)
    oris = torch.stack(oris)
    poses = torch.stack(poses)
    accs_gt = torch.stack(accs_gt)
    oris_gt = torch.stack(oris_gt)
    
    print(f"accs: {accs.shape}, oris: {oris.shape}, poses: {poses.shape}, accs_gt: {accs_gt.shape}, oris_gt: {oris_gt.shape}")
    
    print('Frames: %d' % accs.shape[0])
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    data_filename = f"{args.name}_{timestamp}.pt"
    
    torch.save({'acc': accs, 
                'ori': oris,  
                'pose': poses, 
                'acc_gt': accs_gt,
                'ori_gt': oris_gt
                }, os.path.join(paths.live_record_dir, data_filename))
    
    print('\rFinish.')
        