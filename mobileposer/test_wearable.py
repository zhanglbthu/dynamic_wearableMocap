import torch
import math
import articulate as art
from articulate.utils.wearable import WearableSensorSet
from articulate.utils.noitom import IMUSet
from articulate.utils.bullet.view_rotation_np import RotationViewer
from articulate.utils.pygame import StreamingDataViewer
from articulate.utils.opencv import ProbabilityViewer
from pygame.time import Clock
from auxiliary import calibrate_q, quaternion_inverse
import time
import numpy as np
import keyboard
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def test_wearables_fps(n=1):
    clock = Clock()
    sensor_set = WearableSensorSet()
    
    last_timestamp = time.time()
    pred_rot = None
    frame_intervals = []
    while True:
        clock.tick(60)
        data = sensor_set.get()
        if data == {}:
            last_timestamp = time.time()
            continue
        data = data[0]
        
        cur_rot = data.orientation
        cur_timestamp = time.time()
        
        if pred_rot is not None:
            if not np.array_equal(cur_rot, pred_rot):
                frame_intervals.append(cur_timestamp - last_timestamp)
                last_timestamp = cur_timestamp
                pred_rot = cur_rot  
            
        else:
            pred_rot = cur_rot
            last_timestamp = cur_timestamp
            
        

        if len(frame_intervals) == 1000:
            # 剔除>0.01的异常值
            frame_intervals = np.array(frame_intervals)
            frame_intervals = frame_intervals[frame_intervals > 0.01]
            print('len:', len(frame_intervals))
            print('frame per second:', 1 / np.mean(frame_intervals))
            #print min and max
            print('min:', 1 / np.max(frame_intervals))
            print('max:', 1 / np.min(frame_intervals))
            break

def test_wearable(n=2):
    clock = Clock()
    rviewer = RotationViewer(n, order='wxyz'); rviewer.connect()
    sviewer = StreamingDataViewer(3, (-20, 20), 200, names=['x', 'y', 'z']); sviewer.connect()
    sensor_set = WearableSensorSet()
    while True:
        data = sensor_set.get()
        for i in range(n):
            if i in data.keys():
                rviewer.update(data[i].orientation, i)

        if 0 in data.keys():
            sviewer.plot(data[0].acceleration)

        clock.tick(30)
        print('\r', clock.get_fps(), end='')

def test_wearable_noitom(n_calibration=1):   # use imu 0 and sensor 0
    clock = Clock() 
    rviewer = RotationViewer(n_calibration+1, order='wxyz'); rviewer.connect()
    sviewer = StreamingDataViewer(n_calibration+1, (-20, 20), 200, names=['noitom', 'sensor']); sviewer.connect()
    sensor_set = WearableSensorSet()
    imu_set = IMUSet()

    r"""calibration"""
    print('Rotate the sensor & imu together.')
    qIC_list, qOS_list = [], []
    
    for i in range(n_calibration):
        qIS, qCO = [], []
        while len(qIS) < 1:
            imu_set.app.poll_next_event()
            qIS.append(torch.tensor(imu_set.sensors[0].get_posture()).float()) # noitom
            qCO.append(torch.tensor(sensor_set.get()[i].orientation).float()) # wearable sensor
            print('\rCalibrating... (%d/%d)' % (len(qIS), n_calibration), end='')
            
        qCI, qSO = calibrate_q(torch.stack(qIS), torch.stack(qCO))
        print('\tfinished\nqCI:', qCI, '\tqSO:', qSO)
        qIC_list.append(quaternion_inverse(qCI))
        qOS_list.append(quaternion_inverse(qSO))

    r"""comparison"""
    while True:
        clock.tick(30)
        imu_set.app.poll_next_event()
        qIS_noitom = torch.tensor(imu_set.sensors[0].get_posture()).float()
        aSS_noitom = torch.tensor(imu_set.sensors[0].get_accelerated_velocity()).float() / 1000 * 9.8
        
        # qCO_sensor = torch.tensor(sensor_set.get()[0].orientation).float()
        # aSS_sensor = torch.tensor(sensor_set.get()[0].raw_acceleration).float()
        
        sensor_data = sensor_set.get()
        
        # # convert CO to IS
        # qIC = quaternion_inverse(qCI)
        # qOS = quaternion_inverse(qSO)
        qIS_sensor_list = [qIS_noitom]
        aSS_sensor_list = [aSS_noitom[0]]
        for i in range(n_calibration):        
            # qIS_sensor = art.math.quaternion_product(art.math.quaternion_product(qIC, qCO_sensor), qOS)
            qCO_sensor = torch.tensor(sensor_data[i].orientation).float()
            aSS_sensor = torch.tensor(sensor_data[i].raw_acceleration).float()
            qIS_sensor = art.math.quaternion_product(art.math.quaternion_product(qIC_list[i], qCO_sensor), qOS_list[i])
            
            qIS_sensor_list.append(qIS_sensor)
            aSS_sensor_list.append(aSS_sensor[0])
        
        # rviewer.update_all([qCO_noitom, qCO_sensor])

        rviewer.update_all(qIS_sensor_list)
        sviewer.plot(aSS_sensor_list)
        print('\r', clock.get_fps(), end='')

def test_wearable_light_proximity():
    clock = Clock()
    sviewer = StreamingDataViewer(2, y_range=(-0.2, 1.2), window_length=200, names=['light intensity', 'proximity distance']); sviewer.connect()
    sensor_set = WearableSensorSet()
    while True:
        data = sensor_set.get()
        if 0 in data.keys():
            light_intensity_value = 1 - math.exp(-data[0].light_intensity / 4)
            proximity_distance_value = 1 - data[0].proximity_distance / 5
            sviewer.plot([light_intensity_value, proximity_distance_value])
            clock.tick(60)

def test_wearable_light():
    clock = Clock()
    sviewer = StreamingDataViewer(1, y_range=(0.0, 500.0), window_length=200, names=['light intensity']); sviewer.connect()
    sensor_set = WearableSensorSet()
    while True:
        data = sensor_set.get()
        if 0 in data.keys():
            light_intensity_value = data[0].light_intensity
            sviewer.plot([light_intensity_value])
            clock.tick(60)

def test_wearable_place():
    clock = Clock()
    sviewer = StreamingDataViewer(2, y_range=(-0.2, 1.2), window_length=200, names=['light intensity', 'proximity distance']); sviewer.connect()
    pviewer = ProbabilityViewer(3, name_pairs=[('', 'leg'), ('', 'hand'), ('', 'head')]); pviewer.connect()
    sensor_set = WearableSensorSet()
    while True:
        data = sensor_set.get()
        if 0 in data.keys():
            light_intensity_value = 1 - math.exp(-data[0].light_intensity / 4)
            proximity_distance_value = 1 - data[0].proximity_distance / 5
            if light_intensity_value < 0.01:
                pviewer.update([1, 0, 0])
            elif light_intensity_value < 0.8 and proximity_distance_value == 1:
                pviewer.update([0, 1, 1])
            else:
                pviewer.update([0, 1, 0])
            sviewer.plot([light_intensity_value, proximity_distance_value])
            clock.tick(60)

def test_wearable_height(filtered=True):
    clock = Clock()
    sensor_set = WearableSensorSet()
    
    window_size = 100000  # 可视化最近50个高度数据
    plt.ion()  # 打开交互模式
    fig, ax = plt.subplots()
    height_line, = ax.plot([], [], 'r-', label='Estimated Height(cm)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Estimated Height')
    ax.text(0.05, 0.95, 'Estimated Height(cm)', transform=ax.transAxes, fontsize=12, verticalalignment='top', color='darkblue')
    
    def update_plot(height_data):
        # 只绘制窗口内的数据
        if len(height_data) > window_size:
            display_data = height_data[-window_size:]  # 获取窗口内的最新数据
        else:
            display_data = height_data

        height_line.set_ydata(display_data)
        height_line.set_xdata(range(len(display_data)))

        ax.relim()  # 重新调整坐标轴范围
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

    heights = [0, 10, 20, 30, 40, 50]
    pressures = []
    coefs = []
    
    cur_pressure = []
    len_threshold = 10

    def on_key_press(event):
        
        if event.name == 'q':
            # 收集压力数据
            if len(pressures) >= len(heights):
                print('Pressures already collected.')
                return
            if len(cur_pressure) >= len_threshold:
                mean_pressure = np.mean(cur_pressure)
                pressures.append(mean_pressure)
                print('mean pressure:', mean_pressure)
                cur_pressure.clear()
                return
                
            # 获取新的压力数据
            data = sensor_set.get()
            if 0 in data.keys():
                pressure = data[0].pressure
                print('pressure:', pressure)
                cur_pressure.append(pressure)

        elif event.name == 'w':
            print('Collected pressures:', pressures)
            
            # 检查是否收集到了足够的压力数据
            if len(pressures) == len(heights):
                # 执行线性回归
                model = LinearRegression()
                
                heights_reshaped = np.array(heights).reshape(-1, 1)  # 转换形状以适应模型
                pressures_array = np.array(pressures)
                
                model.fit(pressures_array.reshape(-1, 1), heights_reshaped)
                
                # 获取回归系数和截距
                slope = model.coef_[0]
                intercept = model.intercept_
                print('Linear Regression Coefficients:')
                print('Slope (Coefficient):', slope)
                print('Intercept:', intercept)
                # 获取 R^2 值
                r = model.score(pressures_array.reshape(-1, 1), heights_reshaped)
                print('R^2:', r)
                
                coefs.append((slope, intercept))
                
            else:
                print(f"Not enough data collected: {len(pressures)} of {len(heights)} required.")
            
    # 注册按键监听事件
    keyboard.on_press(on_key_press)
    
    estimated_heights = []
    MA_window_size = 2  # 滑动平均滤波窗口大小
    pressure_window = []
    while True:
        data = sensor_set.get()
        if 0 in data.keys():
            pressure = data[0].pressure
            
            if len(coefs) > 0:
                
                if filtered:
                    pressure_window.append(pressure)
                    if len(pressure_window) > MA_window_size:
                        pressure_window.pop(0)

                    pressure = np.mean(pressure_window)
                
                
                calculated_height = coefs[0][0] * pressure + coefs[0][1]
                estimated_heights.append(calculated_height)
                # print(f'Current pressure: {pressure}, Estimated height: {calculated_height}', end='\r')
                
                # 可视化高度
                update_plot(estimated_heights)
                
            print(f'\rfps: {clock.get_fps():.2f}', end='')
            clock.tick(60)

def test_wearable_pressure(filtered=True, sensor_count=1):
    clock = Clock()
    sensor_set = WearableSensorSet()
    
    window_size = 10 # 可视化最近50个高度数据
    plt.ion()  # 打开交互模式
    fig, ax = plt.subplots()

    pressure_lines = {}
    colors = ['r', 'b', 'g', 'c', 'm']
    for i in range(sensor_count):
        pressure_lines[i], = ax.plot([], [], f'{colors[i % len(colors)]}-', label=f'Sensor {i+1} Pressure(hPa)')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Pressure')
    ax.text(0.05, 0.95, 'Pressure(hPa)', transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black')
    
    def update_plot(pressure_data, pressure_data_filtered=None):
        for i in range(sensor_count):
            # 只绘制窗口内的数据
            if len(pressure_data[i]) > window_size:
                display_data = pressure_data[i][-window_size:]  # 获取窗口内的最新数据
            else:
                display_data = pressure_data[i]
        
        ax.relim()  # 重新调整坐标轴范围
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    # 初始化存储多个传感器数据的字典
    pressures_raw = {i: [] for i in range(sensor_count)}
    
    MA_window_size = 10  # 滑动平均滤波窗口大小
    
    while True:
        data = sensor_set.get()
        for i in range(sensor_count):
            if i in data.keys():

                pressure = data[i].pressure - 1000
                
                pressures_raw[i].append(pressure)

        update_plot(pressures_raw)
        clock.tick(60)
        print(f'\rfps: {clock.get_fps():.2f}', end='')

if __name__ == '__main__':
    test_wearable(n=1)
    # test_wearable_noitom(n_calibration=1)
    # test_wearable_light_proximity()
    # test_wearable_light()
    # test_wearable_place()
    # test_wearable_height()
    # test_wearable_pressure(filtered=False, sensor_count=1)
    # test_wearable_gps(record=True)