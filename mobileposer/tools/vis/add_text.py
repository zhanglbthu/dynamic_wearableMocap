import cv2
import numpy as np
from tqdm import tqdm
import sys
import os

# 将项目根目录添加到环境变量中
search_dir = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(search_dir)

from config import combos_color

# 视频路径
video_path = 'data/result/TotalCapture/video/tran/2.mp4'

# 创建视频读取对象
cap = cv2.VideoCapture(video_path)

# 获取视频的帧宽度、高度、FPS 和总帧数
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数

# 创建视频写入对象
output_path = 'data/result/TotalCapture/video/tran/2_with_text.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 定义方格和文字的起始位置
start_x = width - 250  # 距离右上角的横向偏移
start_y = 30  # 距离顶部的纵向偏移
box_size = 30  # 每个颜色方格的大小
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
font_thickness = 2
spacing = 40  # 方格与文字行之间的间距

# 将 RGB 颜色转换为 BGR 格式
def rgb_to_bgr(color):
    return [int(c * 255) for c in color[::-1]]

# 创建进度条
with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 画图示部分
        for i, (combo, color) in enumerate(combos_color.items()):
            # 计算当前方格和文本的y坐标
            box_y = start_y + i * spacing
            text_y = box_y + 20

            # 画颜色方格
            box_color = rgb_to_bgr(color)
            cv2.rectangle(frame, (start_x, box_y), (start_x + box_size, box_y + box_size), box_color, -1)

            # 写combo名称
            cv2.putText(frame, combo, (start_x + box_size + 10, text_y), font, font_scale, (0, 0, 0), font_thickness)

        # 写入修改后的帧
        out.write(frame)

        # 更新进度条
        pbar.update(1)

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
