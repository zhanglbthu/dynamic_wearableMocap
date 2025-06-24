import torch
import os
import tqdm
import sys

seq_path = "data/livedemo/record/test_trans.pt"

seq_data = torch.load(seq_path)

print("end")