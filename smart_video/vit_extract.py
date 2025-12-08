import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
from scipy.spatial.distance import cosine
import numpy as np
#默认在~/.cache/huggingface/hub
custom_cache_dir = "/workspace/work/zhipeng16/yolo8-plus-iopaint/models/"
# 步骤 1: 加载预训练的 CLIP 模型和处理器
# 使用 "openai/clip-vit-base-patch32" 是一个不错的选择，性能和速度均衡
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name,cache_dir=custom_cache_dir)
processor = CLIPProcessor.from_pretrained(model_name,cache_dir=custom_cache_dir)

# 检查是否有可用的GPU，并把模型移动到GPU上
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

def detect_scene_changes_with_clip(video_path, threshold=0.8):
    """
    使用 CLIP 模型提取帧embedding，并计算余弦相似度来检测视频中的场景切换。

    Args:
        video_path (str): 视频文件的本地路径。
        threshold (float): 余弦相似度阈值。如果相似度低于此值，则认为发生场景切换。
                           阈值范围在 0 到 1 之间。值越小，检测越敏感。

    Returns:
        list: 一个包含所有场景切换点（以秒为单位）的列表。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: 无法打开视频文件 {video_path}")
        return []

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    if frame_rate <= 0:
        print("Error: 无法获取视频帧率，使用默认值 30 FPS。")
        frame_rate = 30.0

    scene_change_points = []
    prev_frame_embedding = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 将 OpenCV 的 BGR 格式转换为 RGB 格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 步骤 2 & 3: 预处理帧并提取 embedding
        inputs = processor(images=rgb_frame, return_tensors="pt").to(device)
        
        with torch.no_grad():
            current_frame_embedding = model.get_image_features(**inputs)
            # 归一化 embedding
            current_frame_embedding /= current_frame_embedding.norm(p=2, dim=-1, keepdim=True)
            current_frame_embedding = current_frame_embedding.squeeze().cpu().numpy()

        if prev_frame_embedding is not None:
            # 步骤 4: 计算两帧 embedding 之间的余弦相似度
            # scipy 的 cosine 函数计算的是余弦距离，其值为 1 - 余弦相似度
            similarity = 1 - cosine(prev_frame_embedding, current_frame_embedding)
            
            # 步骤 5: 阈值判断
            if similarity < threshold:
                time_in_seconds = frame_count / frame_rate
                scene_change_points.append(round(time_in_seconds, 2))
                
        prev_frame_embedding = current_frame_embedding
        frame_count += 1
    
    cap.release()
    return scene_change_points

if __name__ == '__main__':
    # 示例用法，请将 'your_video.mp4' 替换为你的视频文件路径
    # video_file_path = 'your_video.mp4'
    # 假设你有一个名为 'test_video.mp4' 的视频文件
    video_file_path = 'test_video.mp4'

    print(f"开始使用 CLIP 模型分析视频: {video_file_path}")
    
    # 使用 CLIP 模型时，相似度阈值通常可以设定得更高，例如 0.8 或 0.85
    change_points = detect_scene_changes_with_clip(video_file_path, threshold=0.82)
    
    if change_points:
        print("检测到的场景切换点 (以秒为单位):")
        for point in change_points:
            print(f"- {point}s")
    else:
        print("未检测到明显的场景切换。")