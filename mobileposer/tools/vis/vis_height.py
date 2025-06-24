from pygame.time import Clock
from articulate.utils.unity import MotionViewer
from pygame.time import Clock
import torch
import os


deta_dir = 'data/debug/amass'

def vis_height():
    pose_path = os.path.join(deta_dir, 'pose.pt')
    tran_path = os.path.join(deta_dir, 'tran.pt')
    height_path = os.path.join(deta_dir, 'height.pt')
    
    poses = torch.load(pose_path)
    trans = torch.load(tran_path)
    heights = torch.load(height_path)
    
    index = 0
    pose = poses[index]
    tran = trans[index]
    height = heights[index]
    frame_num = pose.shape[0]
    
    clock = Clock()
    
    with MotionViewer(1, overlap=True, names="gt") as viewer:
        for i in range(frame_num):
    
            viewer.update_all([pose[i]], [tran[i]])
            viewer.render()
            clock.tick(60)
            print('\r', i, end='')

if __name__ == '__main__':
    vis_height()