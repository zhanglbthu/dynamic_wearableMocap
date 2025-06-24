import time
import socket
import torch
from pygame.time import Clock

import articulate as art
import win32api
import os
from config import *
import keyboard
import datetime
from articulate.utils.noitom import *
from articulate.utils.unity import MotionViewer

from articulate.utils.wearable import WearableSensorSet
from auxiliary import calibrate_q, quaternion_inverse

from mobileposer.utils.model_utils import load_model
from mobileposer.models import MobilePoserNet
from mobileposer.data import PoseDataset
from kalman import KalmanFilter
import numpy as np
import sys

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
    imu_set.clear()
    RIS = imu_set.get()[1]
    
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

if __name__ == '__main__':
    os.makedirs(paths.temp_dir, exist_ok=True)
    os.makedirs(paths.live_record_dir, exist_ok=True)
    
    device = torch.device("cuda")
    imu_set = IMUSet()
    sensor_set = WearableSensorSet()
    clock = Clock()
    
    k = - 850
    
    # region: calculate sensors pressure bias
    pressures = {"sensor0": [], "sensor1": []}
    while True:
        data = sensor_set.get()
        if 0 in data.keys() and 1 in data.keys():
            pressures["sensor0"].append(data[0].pressure)
            pressures["sensor1"].append(data[1].pressure)
            if len(pressures["sensor0"]) > 100:
                p_bias = np.mean(pressures["sensor0"]) - np.mean(pressures["sensor1"])
                print('p_bias:', p_bias)
                break
    # endregion
    
    # region: set network
    ckpt_path = "data/checkpoints/mobileposer_finetuneddip/model_finetuned.pth"
    net = load_model(ckpt_path)
    print('Model loaded.')
    # endregion

    # region: align noitom and sensor imu data
    
    r"""calibration"""
    print('Rotate the sensor & imu together.')
    n_calibration = 2
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
    # endregion
    
    # region: align inertial and global coordinate & T-pose calibration
    RMI, RSB = tpose_calibration(n_calibration)
    # endregion
    
    # region: calculate height bias
    b_window = 100
    bs = []
    while True:
        data = sensor_set.get()
        if 0 in data.keys():
            pressure = data[0].pressure
            bs.append(k * pressure)
            if len(bs) > b_window:
                h_bias = - np.mean(bs)
                print('h_bias:', h_bias)
                break
    # endregion
    
    # set kalman filter
    kfs = [KalmanFilter(k, h_bias) for _ in range(n_calibration)]
    
    imu_set.clear()
    
    accs, oris = [], []
    poses, trans = [], []
    data = {'RMI': RMI, 'RSB': RSB, 'aM': [], 'RMB': []}
    
    net.eval()
    
    from articulate.utils.pygame import StreamingDataViewer
    sviewer = StreamingDataViewer(2, y_range=(-10, 10), window_length=500, names=['noitom', 'phone']); sviewer.connect()
    
    with torch.no_grad(), MotionViewer(1, names=['Wearable Sensors']) as viewer:
        while True:
            clock.tick(60)
            
            # region: read noitom and sensor data
            tframe, RIS, aI = imu_set.get()
            
            sensor_data = sensor_set.get()
            
            heights = []
            
            for i in range(n_calibration):
                qCO_sensor = torch.tensor(sensor_data[i].orientation).float()
                aSS_sensor = torch.tensor(sensor_data[i].raw_acceleration).float() # [3]
                qIS_sensor = art.math.quaternion_product(art.math.quaternion_product(qIC_list[i], qCO_sensor), qOS_list[i])
                RIS_sensor = art.math.quaternion_to_rotation_matrix(qIS_sensor) # [1, 3, 3]
                
                aIS_sensor = RIS_sensor.squeeze(0).mm( - aSS_sensor.unsqueeze(-1)).squeeze(-1) + torch.tensor([0., 0., 9.8]) 
                
                if i == 0:
                    index = 3
                else:
                    index = 0
                
                RIS[index, :, :] = RIS_sensor[0, :, :]

                aI[index, :] = aIS_sensor
                
                pressure = sensor_data[i].pressure
                
                if i == 1:
                    pressure += p_bias
                
                z = np.array([[pressure]])
                kfs[i].predict()
                kfs[i].update(z)
                h_filtered = kfs[i].get_height() * 0.01
                heights.append(h_filtered)
            
            RMB = RMI.matmul(RIS).matmul(RSB) # [6, 3, 3]
            aM = aI.mm(RMI.t()) # [6, 3]
            
            oris.append(RMB)
            accs.append(aM)
            
            # select combo
            combo = [0, 3, 4]
            
            aM = aM[combo] / amass.acc_scale
            RMB = RMB[combo]
            # endregion
            
            # compute relative height
            rheight = torch.tensor([heights[1] - heights[0]]).float()
            print("heights: ", heights[0], end=' ')
            
            input = torch.cat([aM.flatten(), RMB.flatten()], dim=0).to("cuda")  
            
            pose, _, tran, _ = net.forward_online(input)
            
            # edit translation z
            tran[1] = heights[0]
            
            data['aM'].append(aM)
            data['RMB'].append(RMB)
            
            poses.append(pose)
            trans.append(tran)
            
            # convert tensor to numpy
            pose = pose.cpu().numpy()
            tran = tran.cpu().numpy()
            
            viewer.update_all([pose], [tran], render=False)
            viewer.render()
            
            print('\r', clock.get_fps(), end='')

            if keyboard.is_pressed('q'):
                # save oris, accs, trans and poses
                sub_dir = 'test' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
                os.makedirs(os.path.join(paths.live_record_dir, sub_dir), exist_ok=True)
                torch.save(oris, os.path.join(paths.live_record_dir, sub_dir, 'oris.pt'))
                torch.save(accs, os.path.join(paths.live_record_dir, sub_dir, 'accs.pt'))
                torch.save(poses, os.path.join(paths.live_record_dir, sub_dir, 'poses.pt'))
                torch.save(trans, os.path.join(paths.live_record_dir, sub_dir, 'trans.pt'))
                break
            
            print(f'\rfps: {clock.get_fps():.2f}', end='')

    oris = torch.stack(oris)
    accs = torch.stack(accs)
    
    # print frames num
    print('Frames: %d' % accs.shape[0])
    
    torch.save({'acc': accs, 'ori': oris}, os.path.join(paths.live_record_dir, 'test' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.pt'))
    
    print('\rFinish.')