from my_data import *
from my_trainner import *
from my_model import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = IMUData.load_data(folder_path=paths.real_dataset_processed_dir, step=1)

data_len = len(dataset['rot'])

# debug
rot = dataset['rot']
acc = dataset['acc']
rot_gt = dataset['rot_gt']
acc_gt = dataset['acc_gt']
# print(rot.shape, acc.shape, rot_gt.shape, acc_gt.shape)
print(f"rot: {rot.shape}, acc: {acc.shape}, rot_gt: {rot_gt.shape}, acc_gt: {acc_gt.shape}")

data_train = IMUData(rot=rot, acc=acc, rot_gt=rot_gt, acc_gt=acc_gt, seg_info=dataset['seg_info'], seq_len=128)

model = LSTMIC(n_input=config.imu_num*(3+3*3), n_output=config.imu_num*6).to(device)

optimizer = [torch.optim.Adam(model.parameters(), lr=1e-3)]
trainner = TicTrainner(model=model, data=data_train, optimizer=optimizer, batch_size=128)

# trainner.restore(checkpoint_path='checkpoint/TIC_13.pth', load_optimizer=True)
epoch=20
model_name = f'RealData_0915_4pants_2345'
ckpt_path = f'./data/checkpoints/calibrator/{model_name}'
os.makedirs(ckpt_path, exist_ok=True)
print(model_name)
for i in range(epoch):
    trainner.run(epoch=1, data_shuffle=True)
    trainner.save(folder_path=ckpt_path, model_name=model_name)
    trainner.log_export(f'./data/checkpoints/calibrator/log/{model_name}.xlsx')