import os
import json
import sys
import numpy as np
# import tensorflow as tf

# 确保能引用到同目录下的 transnetv2.py
# 如果脚本不在 inference 目录下，请修改此路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from transnetv2 import TransNetV2
except ImportError:
    print("❌ 错误：未找到 'transnetv2.py'。请将此脚本放在 TransNetV2/inference/ 目录下，或添加到 PYTHONPATH。")
    sys.exit(1)

def run_transnet_v2(video_path, model_dir=None):
    """
    运行 TransNet V2 推理
    """
    print(f"🚀 初始化 TransNet V2 模型...")
    
    # 自动寻找权重路径
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(__file__), "transnetv2-weights/")
        
    if not os.path.exists(model_dir):
        print(f"❌ 权重目录不存在: {model_dir}，请先下载 TransNet V2 预训练权重。")
        return [], 0

    # 初始化模型
    model = TransNetV2(model_dir)
    
    print(f"🎬 开始处理视频: {video_path}")
    
    # predict_video 会自动处理 ffmpeg 读取、resize (27x48) 和 inference
    # predictions shape: [frames, 1] (logits)
    video_frames, single_frame_predictions, all_frame_predictions = \
        model.predict_video(video_path)
    
    # 将 logits 转换为场景列表 [start_frame, end_frame]
    scenes_arr = model.predictions_to_scenes(single_frame_predictions)
    
    # 我们需要获取 FPS 来计算时间戳，TransNetV2 内部其实并不直接返回 FPS
    # 这里我们用 ffmpeg-python 再次快速获取一下 FPS，或者假设 video_frames 的长度对应时长
    # 更稳妥的方式是利用 ffmpeg probe
    import ffmpeg
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        fps_str = video_stream['r_frame_rate']
        num, den = map(int, fps_str.split('/'))
        fps = num / den
    except Exception as e:
        print(f"⚠️ 无法获取精确 FPS，尝试通过总帧数估算 (可能不准): {e}")
        # 如果获取失败，这里需要手动处理，通常建议必须获取到 FPS
        fps = 30.0 

    return scenes_arr, fps

def format_to_sota_style(scenes_arr, fps):
    """
    将 TransNet V2 的帧索引转换为 SOTA 格式的 JSON List
    Reference: 拆条sota输出.md
    """
    output_list = []
    
    for scene in scenes_arr:
        start_frame, end_frame = scene
        
        # 转换为秒，保留2位小数
        start_time = round(start_frame / fps, 2)
        end_time = round(end_frame / fps, 2)
        
        # 过滤掉极短的闪烁 (可选，例如小于0.5秒)
        if end_time - start_time < 0.2:
            continue

        item = {
            "beginTime": start_time,
            "endTime": end_time,
            # TransNet 是纯视觉模型，没有语义主题，留空
            "theme": "", 
            "type": "common",
            # 标记来源，方便你做对比
            "by": "transnet_v2"
        }
        output_list.append(item)
        
    return output_list

if __name__ == "__main__":
    # ================= 配置 =================
    # 替换为你的视频路径
    target_video_path = "/workspace/work/zhipeng16/datasets/videos/3分钟游戏集锦.mp4"
    output_json_path = "./smart_video/transnet_output.json"
    
    # 权重路径 (默认假设在脚本同级目录的 transnetv2-weights 下)
    # weights_dir = "/path/to/transnetv2-weights/" 
    weights_dir = None 
    # =======================================

    if not os.path.exists(target_video_path):
        print("❌ 视频文件不存在")
        sys.exit(1)

    # 1. 运行推理
    scenes, fps = run_transnet_v2(target_video_path, weights_dir)
    
    if len(scenes) == 0:
        print("⚠️ 未检测到场景或运行出错。")
        sys.exit(0)
        
    print(f"✅ 检测到 {len(scenes)} 个镜头 (FPS: {fps:.2f})")

    # 2. 格式化输出
    sota_formatted_data = format_to_sota_style(scenes, fps)
    
    # 3. 保存结果
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(sota_formatted_data, f, ensure_ascii=False, indent=4)
        
    print(f"💾 结果已保存至: {output_json_path}")
    print("💡 下一步：你可以使用之前的 verify_cuts.py 读取此 JSON 进行物理切片验证。")