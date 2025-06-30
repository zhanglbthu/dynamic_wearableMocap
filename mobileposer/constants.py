import os

# from models import MobilePoserNet, Poser, Joints, FootContact, Velocity
from model.imuposer_local.poser import Poser as IMUPoser

from model.joints import Joints as MobileJoints
from model.poser import Poser as MobilePoser
from model.velocity import Velocity as MobileVelocity
from model.footcontact import FootContact as MobileFootContact

from model.heightposer.poser import Poser as HeightPoser
from model.heightposer.velocity import Velocity as HeightVelocity
from model.heightposer.footcontact import FootContact as HeightFootContact

# # List of available modules
# MODULES = {
#     'joints': Joints, 
#     'poser': Poser,
#     'velocity': Velocity,
#     # 'foot_contact': FootContact,
#     # 'velocity': Velocity,
# }

IMUPOSER = {
    'poser': IMUPoser,
}

MOBILEPOSER = {
    'joints': MobileJoints,
    'poser': MobilePoser,
    'velocity': MobileVelocity,
    'foot_contact': MobileFootContact,
}

HEIGHTPOSER = {
    'poser': HeightPoser,
}

# Calibration constants
KEYS = ['unix_timestamp', 'sensor_timestamp', 'accel_x', 'accel_y', 'accel_z', 'quart_x', 'quart_y', 'quart_z', 'quart_w', "roll", "pitch", "yaw"]
STOP = "stop"
SEP = ":"

# Socket configurations
OUT_UDP_PORT = 7777
OUT_UDP_IP = "127.0.0.1"
HOST = "0.0.0.0"
PORTS = [8001, 8002, 8003, 8004, 8005]

# Buffer settings
CHUNK = 2048
BUFFER_SIZE = 50
min_time_diff = 1/25.6  # seconds

# Display settings
H, W = 860, 860

# SMPL constants
NUM_VERTICES = 6890