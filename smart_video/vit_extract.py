import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
from scipy.spatial.distance import cosine
import numpy as np

# 默认在~/.cache/huggingface/hub
custom_cache_dir = "/workspace/work/zhipeng16/yolo8-plus-iopaint/models/"
# 步骤 1: 加载预训练的 CLIP 模型和处理器
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name, cache_dir=custom_cache_dir)
processor = CLIPProcessor.from_pretrained(model_name, cache_dir=custom_cache_dir)

# 检查是否有可用的GPU，并把模型移动到GPU上
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

def asr_ms_to_hmsms(ms):
    """将毫秒转换为 HH:MM:SS,ms 格式"""
    seconds = int(ms / 1000)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    ms_remain = int(ms % 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms_remain:03d}"

def detect_scene_changes_with_clip(video_path, threshold=0.8):
    """
    使用 CLIP 模型检测场景切换，并返回切分后的详细片段信息。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: 无法打开视频文件 {video_path}")
        return []

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    if frame_rate <= 0:
        print("Error: 无法获取视频帧率，使用默认值 30 FPS。")
        frame_rate = 30.0
    
    # 计算视频总时长(ms)
    video_duration_ms = (total_frames / frame_rate) * 1000

    segments = []
    prev_frame_embedding = None
    frame_count = 0
    
    # 记录上一段的结束时间（也就是当前段的开始时间），单位毫秒
    last_end_ms = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 简单优化：每隔几帧检测一次可以大幅提升速度，这里保持逐帧以保证精度
        # 如果需要加速，可以添加 if frame_count % skip_frames != 0: continue
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        inputs = processor(images=rgb_frame, return_tensors="pt").to(device)
        
        with torch.no_grad():
            current_frame_embedding = model.get_image_features(**inputs)
            current_frame_embedding /= current_frame_embedding.norm(p=2, dim=-1, keepdim=True)
            current_frame_embedding = current_frame_embedding.squeeze().cpu().numpy()

        if prev_frame_embedding is not None:
            similarity = 1 - cosine(prev_frame_embedding, current_frame_embedding)
            
            # 阈值判断：如果相似度低，说明发生了场景切换
            if similarity < threshold:
                # 当前切分点时间 (ms)
                current_time_ms = (frame_count / frame_rate) * 1000
                
                # 构建片段数据
                dur_start = last_end_ms
                dur_end = current_time_ms
                
                segment = {
                    "text": str(similarity), # 无法计算，留空
                    "beginTime": round(dur_start / 1000, 2),
                    "endTime": round(dur_end / 1000, 2),
                    "start_str": asr_ms_to_hmsms(dur_start),
                    "end_str": asr_ms_to_hmsms(dur_end),
                    "duration": int(dur_end - dur_start),
                    "theme": "", # 无法计算，留空
                    "type": "Semantic",
                    "by": "CLIP_Scene_Detect"
                }
                segments.append(segment)
                
                # 更新下一段的开始时间
                last_end_ms = current_time_ms
                
        prev_frame_embedding = current_frame_embedding
        frame_count += 1
    
    cap.release()
    
    # --- 处理最后一段 ---
    # 循环结束后，将最后一次切分点到视频结束的时间作为最后一段
    if last_end_ms < video_duration_ms:
        dur_start = last_end_ms
        dur_end = video_duration_ms
        
        # 防止最后一段极短（例如最后一帧刚切分）
        if (dur_end - dur_start) > 100: # 只有大于100ms才保留
            segment = {
                "text": "",
                "beginTime": round(dur_start / 1000, 2),
                "endTime": round(dur_end / 1000, 2),
                "start_str": asr_ms_to_hmsms(dur_start),
                "end_str": asr_ms_to_hmsms(dur_end),
                "duration": int(dur_end - dur_start),
                "theme": "",
                "type": "Semantic",
                "by": "CLIP_Scene_Detect"
            }
            segments.append(segment)

    return segments

if __name__ == '__main__':
    video_path = '/workspace/work/zhipeng16/datasets/videos/3分钟游戏集锦.mp4'
    print(f"开始使用 CLIP 模型分析视频: {video_path}")
    
    # 获取切分后的片段列表
    scene_segments = detect_scene_changes_with_clip(video_path, threshold=0.82)
    
    if scene_segments:
        print(f"检测到 {len(scene_segments)} 个场景片段:")
        import json
        # 打印前2个做示例，避免刷屏
        print(json.dumps(scene_segments, indent=4, ensure_ascii=False))
        # print(segments)
        if len(scene_segments) > 2:
            print("...")
    else:
        print("未检测到明显的场景切换或视频无法读取。")