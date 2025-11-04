import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from scipy.spatial.distance import cosine
import numpy as np

# 步骤1: 加载预训练模型
# 使用 ResNet50，并移除其最后一层以获取特征向量
model = models.resnet50(pretrained=True)
# 移除最后一层，使其只输出特征向量
model = nn.Sequential(*list(model.children())[:-1])
model.eval() # 切换到评估模式，不进行梯度计算

# 检查是否有可用的GPU，并把模型移动到GPU上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义图像预处理
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def detect_scene_changes_with_embedding(video_path, threshold=0.6):
    """
    使用预训练的ResNet模型提取帧embedding，并计算余弦相似度来检测场景切换。

    Args:
        video_path (str): 视频文件的路径。
        threshold (float): 余弦相似度阈值。如果相似度低于此值，则认为发生场景切换。
                           阈值范围通常在 0 到 1 之间。值越小，检测越敏感。

    Returns:
        list: 包含所有场景切换点（以秒为单位）的列表。
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

        # 步骤2: 预处理帧并提取 embedding
        input_tensor = preprocess(frame)
        input_batch = input_tensor.unsqueeze(0).to(device) # 创建一个minibatch

        with torch.no_grad(): # 在推理时禁用梯度计算，以提高性能
            current_frame_embedding = model(input_batch).squeeze().cpu().numpy()
            
        if prev_frame_embedding is not None:
            # 步骤3: 计算两帧embedding之间的余弦相似度
            similarity = 1 - cosine(prev_frame_embedding, current_frame_embedding)
            
            # 步骤4: 阈值判断
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
    video_file_path = '/workspace/work/zhipeng16/datasets/videos/video_13m.mp4'

    print(f"开始使用深度学习模型分析视频: {video_file_path}")
    
    # 调整阈值可以改变检测的敏感度，例如 0.6 或 0.5
    change_points = detect_scene_changes_with_embedding(video_file_path, threshold=0.55)
    
    if change_points:
        print("检测到的场景切换点 (以秒为单位):")
        for point in change_points:
            print(f"- {point}s")
    else:
        print("未检测到明显的场景切换。")