import os
from moviepy.editor import VideoFileClip, clips_array, TextClip, CompositeVideoClip

# 定义源文件夹和目标文件夹
pose_dir = "data/result/TotalCapture/video/pose"
folders = ["gt", "PIP", "lw_rw_lk_rk_h_r", "lw_rw_lk_rk_h", "lw_rw_h"]
output_dir = "data/result/TotalCapture/video/output"

# 如果目标文件夹不存在，创建它
os.makedirs(output_dir, exist_ok=True)

# 获取其中一个文件夹下的所有 .mp4 文件的数量
video_files = sorted(os.listdir(os.path.join(pose_dir, folders[0])))
video_files = [f for f in video_files if f.endswith('.mp4')]

# 遍历每个视频文件
for video_file in video_files:
    video_clips = []
    
    # 从每个文件夹中读取相同名称的文件
    for folder in folders:
        video_path = os.path.join(pose_dir, folder, video_file)
        video_clip = VideoFileClip(video_path)
        
        # 创建文本剪辑
        txt_clip = TextClip(folder, fontsize=32, color='black', bg_color='transparent')
        txt_clip = txt_clip.set_duration(video_clip.duration)
        
        # 计算文本位置，距离右上角10像素
        position = (video_clip.w - txt_clip.w - 10, 10)
        
        # 将文本覆盖到视频上
        composite_clip = CompositeVideoClip([video_clip, txt_clip.set_position(position)])
        video_clips.append(composite_clip)
    
    # 将视频水平拼接
    final_clip = clips_array([video_clips])
    
    # 保存拼接后的视频到新的文件夹
    output_path = os.path.join(output_dir, video_file)
    final_clip.write_videofile(output_path, codec="libx264")
    
    # 关闭每个视频以释放资源
    for clip in video_clips:
        clip.close()

print("视频拼接完成")
