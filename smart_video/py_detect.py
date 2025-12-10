import os
#pip install scenedetect opencv-python numpy
from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import ContentDetector, AdaptiveDetector
from scenedetect.scene_manager import save_images
import json # 导入 json 模块用于打印示例输出

def split_highlight_video(video_path, output_dir, threshold=27.0, min_duration=2.0):
    """
    针对游戏集锦/纯音乐视频的视觉切分方案
    
    Args:
        video_path: 视频路径
        output_dir: 输出目录
        threshold: 镜头切换阈值 (越小越敏感，游戏视频通常画面变化快，建议 27-30)
        min_duration: 最小切片时长(秒)，小于此长度的片段会被合并
    """
    
    if not os.path.exists(video_path):
        print(f"❌ 视频不存在: {video_path}")
        return

    # 1. 建立场景管理器
    video = open_video(video_path)
    scene_manager = SceneManager()
    
    # 2. 添加检测器
    # ContentDetector: 比较相邻帧的内容差异（适合硬切）
    # AdaptiveDetector: 适合检测渐变/快速运动（对游戏可能更鲁棒，可选）
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=int(min_duration * 30))) 

    print(f"🎬 开始分析视频视觉转场: {os.path.basename(video_path)} ...")
    scene_manager.detect_scenes(video, show_progress=True)

    # 3. 获取切分点列表
    scene_list = scene_manager.get_scene_list()
    print(f"👀 检测到 {len(scene_list)} 个原始视觉镜头")

    # 4. 后处理：合并过短的碎片 (Merge short segments)
    # 集锦有时候剪辑极快（0.5秒一闪），作为拆条素材太短了，需要合并
    final_scenes = []
    final_scenes_detail = [] # <-- 新增的详细信息列表
    
    if not scene_list:
        print("⚠️ 未检测到明显转场，可能是长镜头")
        return

    current_start = scene_list[0][0] # FrameTimecode object
    current_end = scene_list[0][1]
    
    for i in range(1, len(scene_list)):
        next_start = scene_list[i][0]
        next_end = scene_list[i][1]
        
        # 计算当前段落时长 (秒)
        duration = current_end.get_seconds() - current_start.get_seconds()
        
        if duration < min_duration:
            # 如果当前段太短，就吃掉下一段（合并）
            current_end = next_end
        else:
            # 否则，保存当前段，开启新的一段
            final_scenes.append((current_start, current_end))
            current_start = next_start
            current_end = next_end
    
    # 添加最后一段
    final_scenes.append((current_start, current_end))

    print(f"✂️ 优化合并后，最终输出 {len(final_scenes)} 个片段")

    # 4.1 填充 final_scenes_detail 详细信息 <-- 新增逻辑
    for start_timecode, end_timecode in final_scenes:
        begin_time = round(start_timecode.get_seconds(),2)
        end_time = round(end_timecode.get_seconds(),2)
        
        # 转换时间格式，将 scenedetect 的 HH:MM:SS.mmm 格式的句点 '.' 替换为逗号 ','
        # 匹配用户所需的字幕时间格式 (SRT 格式)
        start_str = start_timecode.get_timecode().replace('.', ',')
        end_str = end_timecode.get_timecode().replace('.', ',')
        
        # 计算毫秒时长
        # 确保 duration 是整数毫秒，使用 round() 以防浮点数误差
        duration_ms = int(round((end_time - begin_time) * 1000,2))
        
        detail = {
            "text": "",
            "beginTime": begin_time,
            "endTime": end_time,
            "start_str": start_str,
            "end_str": end_str,
            "duration": duration_ms,
            "theme": "",
            "type": "Semantic",
            "by": "CLIP_Scene_Detect"
        }
        final_scenes_detail.append(detail)

    # 打印第一个详细片段用于验证
    if final_scenes_detail:
        print("\n📝 第一个片段详细信息 (final_scenes_detail[0]):")
        print(json.dumps(final_scenes_detail[0], indent=4, ensure_ascii=False))
    return final_scenes_detail

    # 5. 调用 ffmpeg 进行物理切分
    # SceneDetect 自带的 split_video_ffmpeg 非常好用
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    split_video_ffmpeg(video_path, final_scenes, output_dir=output_dir, show_progress=True)
    print(f"✅ 处理完成，结果已保存在: {output_dir}")

if __name__ == "__main__":
    # 替换为你的集锦视频路径
    # 假设这是你的那个集锦视频
    video_file = '/workspace/work/zhipeng16/datasets/videos/4分钟游戏教学视频.mp4' 
    output_folder = './smart_video/highlight_results2'
    
    # 阈值说明：
    # 游戏视频画面变化剧烈，threshold 设置过低会导致疯狂误切
    # 建议设置在 27 - 35 之间
    split_highlight_video(video_file, output_folder, threshold=30.0, min_duration=5.0)