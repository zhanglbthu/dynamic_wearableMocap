import torch
from pathlib import Path
from enum import Enum, auto

class train_hypers:
    """Hyperparameters for training."""
    batch_size = 256
    num_workers = 8
    num_epochs = 200
    accelerator = "gpu"
    device = 1
    lr = 3e-4

class finetune_hypers:
    """Hyperparamters for finetuning."""
    batch_size = 32
    num_workers = 8
    num_epochs = 15
    accelerator = "gpu"
    device = [0, 1, 2, 3, 4, 5, 6, 7]
    lr = 5e-5

class paths:
    """Relevant paths for MobilePoser. Change as necessary."""
    root_dir = Path().absolute()
    checkpoint = root_dir / "data" / "checkpoints"
    smpl_file = root_dir / "smpl/basicmodel_m.pkl"
    weights_file = root_dir / "data" / "checkpoints/official/weights.pth"
    raw_amass = Path("/root/autodl-tmp/mobileposer/dataset_raw/AMASS")           # TODO: replace with your path
    raw_dip = Path("/root/autodl-tmp/mobileposer/dataset_raw/DIP_IMU")           # TODO: replace with your path
    raw_imuposer = Path("/root/autodl-tmp/mobileposer/dataset_raw/imuposer_dataset")     # TODO: replace with your path
    # eval_dir = root_dir / "data/processed_datasets/eval"
    # processed_datasets = root_dir / "data/processed_datasets"
    eval_dir = Path("/root/autodl-tmp/mobileposer/eval")
    processed_datasets = Path("/root/autodl-tmp/mobileposer/dataset_work")
    # processed_datasets = Path("data/dataset_work")

    # livedemo record directory
    temp_dir = Path("data/livedemo/temp")
    live_record_dir = Path("data/livedemo/record")
    
    # TotalCapture dataset
    calibrated_totalcapture = Path("/root/autodl-tmp/mobileposer/dataset_raw/TotalCapture/calibrated")
    raw_totalcapture_official = Path("/root/autodl-tmp/mobileposer/dataset_raw/TotalCapture/official")
    processed_totalcapture = Path("/root/autodl-tmp/mobileposer/dataset_work/TotalCapture")
    
    # Physics file
    physics_model_file = "/home/project/mocap/MobilePoser/mobileposer/data/physics/physics.urdf"
    physics_parameter_file = Path("/home/project/mocap/MobilePoser/mobileposer/data/physics/physics_parameters.json")
    plane_file = Path("/home/project/mocap/MobilePoser/mobileposer/data/physics/plane.urdf")
    
    vicon_gt_dir = Path('/root/autodl-tmp/mobileposer/dataset_raw/TotalCapture/official')
    imu_dir = Path('/root/autodl-tmp/mobileposer/dataset_raw/TotalCapture/gryo_mag')
    calib_dir = Path('/root/autodl-tmp/mobileposer/dataset_raw/TotalCapture/imu')
    AMASS_smpl_dir = Path('/root/autodl-tmp/mobileposer/dataset_raw/AMASS/TotalCapture')
    DIP_smpl_dir = Path('/root/autodl-tmp/mobileposer/dataset_raw/TotalCapture/calibrated')
    
class model_config:
    """MobilePoser Model configurations."""
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # joint set
    n_joints = 3                        # (head, right-wrist, left-wrist, right-hip, left-hip)
    n_imu = 12*n_joints                 # 60 (3 accel. axes + 3x3 orientation rotation matrix) * 5 possible IMU locations
    n_output_joints = 24                # 24 output joints
    n_pose_output = n_output_joints*6   # 144 pose output (24 output joints * 6D rotation matrix)

    # model config
    past_frames = 40
    future_frames = 5
    total_frames = past_frames + future_frames

    # data config
    data_heights = False

    # height poser config
    winit = True
    
    poser_wh =  True
    vel_wh = True
    
    noise_std = 0.1
    
    # mobile poser config
    physics = False
    
    # loss config
    jerk_loss = False
    jerk_loss_weight = 1e-5
    
    symmetry_loss = True
    sym_loss_weight = 1e-3
    
    combo_id = 'lw_rp'
    name = 'heightposer_thigh_full_noise'

class amass:    
    """AMASS dataset information."""
    # device-location combinationsa
    combos = {
        'lw_rp': [0, 3],
        # 'rw_rp_h': [1, 3, 4],
        # 'lw_lp_h': [0, 2, 4],
        # 'rw_lp_h': [1, 2, 4],
        # 'lw_lp': [0, 2],
        # 'lw_rp': [0, 3],
        # 'rw_lp': [1, 2],
        # 'rw_rp': [1, 3],
        # 'lp_h': [2, 4],
        # 'rp_h': [3, 4],
        # 'lp': [2],
        # 'rp': [3],
     }
    
    combos_mine = {
        'lw_rw_lp_rp_h': [0, 1, 2, 3, 4],
        'lw_rp_h': [0, 3, 4],
        'lw_rw_h': [0, 1, 4],
        'lw_rp': [0, 3],
        'lw': [0],
        'rp': [3],
        'h': [4]
    }
    
    acc_scale = 30
    vel_scale = 2

    # left wrist, right wrist, left thigh, right thigh, head, pelvis
    all_imu_ids = [0, 1, 2, 3, 4] 
    imu_ids = [0, 1, 2, 3]

    pred_joints_set = [*range(24)]
    joint_sets = [18, 19, 1, 2, 15, 0]
    ignored_joints = list(set(pred_joints_set) - set(joint_sets))
    
    norm_joint = 0
    normalize_joints = [i for i in range(24) if i != 0]
    
    vel_joint = [*range(1)]

class datasets:
    """Dataset information."""
    # FPS of data
    fps = 30

    # DIP dataset
    dip_test = "dip_test.pt"
    dip_train = "dip_train.pt"

    # TotalCapture dataset
    totalcapture = "totalcapture.pt"

    # IMUPoser dataset
    imuposer = "imuposer.pt"
    imuposer_train = "imuposer_train.pt"
    imuposer_test = "imuposer_test.pt"

    # uneven terrain dataset
    cmu = "CMU.pt"
    
    # Predict datasets
    predict_datasets = {
        'cmu': cmu
    }

    # Test datasets
    test_datasets = {
        'dip': dip_test,
        'totalcapture': totalcapture,
        'imuposer': imuposer_test,
        'cmu': cmu
    }

    # Finetune datasets
    finetune_datasets = {
        'dip': dip_train,
        'imuposer': imuposer_train
    }

    # AMASS datasets (add more as they become available in AMASS!)
    amass_datasets = ['ACCAD', 'BioMotionLab_NTroje', 'BMLhandball', 'BMLmovi', 'CMU', 
                      'DanceDB', 'DFaust_67', 'EKUT', 'Eyes_Japan_Dataset', 'HUMAN4D',
                      'HumanEva', 'KIT', 'MPI_HDM05', 'MPI_Limits', 'MPI_mosh', 'SFU',
                      'SSM_synced', 'TCD_handMocap', 'TotalCapture', 'Transitions_mocap']
    
    # amass_datasets = ['TotalCapture']

    # Root-relative joint positions
    root_relative = False

    # Window length of IMU and Pose data 
    window_length = 125


class joint_set:
    """Joint sets configurations."""
    gravity_velocity = -0.018

    full = list(range(0, 24))
    reduced = [0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    leaf_joint = [7, 8, 10, 11, 20, 21, 22, 23]
    
    ignored = [0, 7, 8, 10, 11, 20, 21, 22, 23]
    
    joint_init = [i for i in range(24) if i != 2]
    
    eval_select = [2, 3, 5, 6, 9, 13, 16, 18]
    
    rarm = [14, 17, 19, 21, 23]
    larm = [13, 16, 18, 20, 22]

    n_full = len(full)
    n_ignored = len(ignored)
    n_reduced = len(reduced)
    n_pose_init = len(joint_init)

    lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    lower_body_parent = [None, 0, 0, 1, 2, 3, 4, 5, 6]


class sensor: 
    """Sensor parameters."""
    device_ids = {
        'Left_phone': 0,
        'Left_watch': 1,
        'Left_headphone': 2,
        'Right_phone': 3,
        'Right_watch': 4
    }


class Devices(Enum):
    """Device IDs."""
    Left_Phone = auto()
    Left_Watch = auto()
    Right_Headphone = auto()
    Right_Phone = auto()
    Right_Watch = auto()