import torch
import math
import articulate as art
from articulate.utils.noitom import *
from articulate.utils.wearable import WearableSensorSet
from articulate.utils.noitom import IMUSet
from articulate.utils.bullet.view_rotation_np import RotationViewer
from articulate.utils.pygame import StreamingDataViewer
from articulate.utils.opencv import ProbabilityViewer
from pygame.time import Clock
from auxiliary import calibrate_q, quaternion_inverse
import time
import numpy as np

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

def tpose_calibration_noitom(imu_set, sensor_set, n_calibration, qIC_list=None, qOS_list=None):
     print('Calibrating T-pose...')
     c = input('Used cached RMI? [y]/n    (If you choose no, put imu 1 straight (x = Right, y = Forward, z = Down, Left-handed).')
     if c == 'n' or c == 'N':
         imu_set.clear()
         RSI = imu_set.get()[1][0].view(3, 3).t()
         RMI = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0.]]).mm(RSI)
     else:
         raise NotImplementedError('No cached RMI available. Please run the calibration with n_calibration=1 first.')
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
             index = 4
         else:
             index = 1

         RIS[index, :, :] = RIS_sensor[0, :, :]
 
     RSB = RMI.matmul(RIS).transpose(1, 2).matmul(torch.eye(3))
     return RMI, RSB

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

def test_wearable_noitom(n_calibration=1):   # use imu 0 and sensor 0
    clock = Clock() 
    rviewer = RotationViewer(3, order='wxyz'); rviewer.connect()
    sviewer = StreamingDataViewer(3, y_range=(-90, 90), window_length=200, names=['Y', 'Z', 'X']); sviewer.connect()
    sensor_set = WearableSensorSet()
    imu_set = IMUSet()
    
    qIC_list, qOS_list = align_sensor_noitom(imu_set, sensor_set, n_calibration)
    RMI, RSB = tpose_calibration_noitom(imu_set, sensor_set, n_calibration, qIC_list, qOS_list)

    r"""comparison"""
    while True:
        clock.tick(30)
        tframe, RIS, aI = imu_set.get()
        
        sensor_data = sensor_set.get()
        
        for i in range(n_calibration):        
            qCO_sensor = torch.tensor(sensor_data[i].orientation).float()
            aSS_sensor = torch.tensor(sensor_data[i].raw_acceleration).float() # [3]
            qIS_sensor = art.math.quaternion_product(art.math.quaternion_product(qIC_list[i], qCO_sensor), qOS_list[i])
            RIS_sensor = art.math.quaternion_to_rotation_matrix(qIS_sensor) # [1, 3, 3]
            
            aIS_sensor = RIS_sensor.squeeze(0).mm( - aSS_sensor.unsqueeze(-1)).squeeze(-1) + torch.tensor([0., 0., 9.8]) 

            RIS[4, :, :] = RIS_sensor[0, :, :]
        
        RMB = RMI.matmul(RIS).matmul(RSB)
        
        RMB_axis = art.math.rotation_matrix_to_axis_angle(RMB).view(6, 3)
        RMB_q = art.math.axis_angle_to_quaternion(RMB_axis).view(6, 4)
        
        rot_noitom, rot_sensor = RMB[3], RMB[4]
        rot = rot_sensor.t() @ rot_noitom
        
        rot_axis = art.math.rotation_matrix_to_axis_angle(rot).view(1, 3)
        rot_euler = art.math.rotation_matrix_to_euler_angle(rot, seq='YZX').view(3)
        # convert rot_euler from radiance to degree
        rot_euler = rot_euler * 180 / np.pi
        rot_q = art.math.axis_angle_to_quaternion(rot_axis).view(4)
        
        r_list = []
        r_list.append(RMB_q[3])
        r_list.append(RMB_q[4])
        r_list.append(rot_q)
        
        rviewer.update_all(r_list)
        
        # visualize euler using sviewer
        sviewer.plot(rot_euler)

        print('\r', clock.get_fps(), end='')

if __name__ == '__main__':
    test_wearable_noitom(n_calibration=1)