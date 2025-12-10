import os
import subprocess
import json
import numpy as np
import cv2
import torch
from funasr import AutoModel
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from transformers import CLIPProcessor, CLIPModel
from scipy.spatial.distance import cosine

class SmartVideoCutter:
    def __init__(self, model_dir, device="cuda"):
        self.model_dir = model_dir
        self.device = device
        self.asr_model = None
        self.clip_model = None
        self.clip_processor = None
        
        # 配置参数
        self.speech_density_threshold = 0.4  # 语音占比阈值 (20%)
        self.clip_similarity_threshold = 0.85 # CLIP 语义相似度阈值 (高于此值则认为是一个场景，需合并)

    def _init_asr(self):
        """懒加载 ASR 模型"""
        if self.asr_model is None:
            print("🚀 Loading FunASR Model...")
            self.asr_model = AutoModel(
                model=os.path.join(self.model_dir, "speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"),
                vad_model=os.path.join(self.model_dir, "speech_fsmn_vad_zh-cn-16k-common-pytorch"),
                punc_model=os.path.join(self.model_dir, "ct-punc"),
                device=self.device,
                disable_update=True
            )

    def _init_clip(self):
        """懒加载 CLIP 模型"""
        if self.clip_model is None:
            print("👁️ Loading CLIP Model...")
            model_name = "openai/clip-vit-base-patch32"
            # 指向你的本地缓存目录
            self.clip_model = CLIPModel.from_pretrained(model_name, cache_dir=self.model_dir)
            self.clip_processor = CLIPProcessor.from_pretrained(model_name, cache_dir=self.model_dir)
            self.clip_model.to(self.device)
            self.clip_model.eval()

    def _extract_audio(self, video_path):
        """提取 16k wav 用于 ASR"""
        wav_path = os.path.splitext(video_path)[0] + ".wav"
        if not os.path.exists(wav_path):
            print(f"🔊 Extracting audio to {wav_path}...")
            subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", 
                 "-i", video_path, "-ac", "1", "-ar", "16000", wav_path],
                check=True
            )
        return wav_path

    def _ms_to_hmsms(self, ms):
        """辅助函数：毫秒转格式化字符串"""
        return f"{int(ms)//3600000:02d}:{int(ms)%3600000//60000:02d}:{int(ms)%60000//1000:02d}.{int(ms)%1000:03d}"

    def run_asr_pipeline(self, video_path):
        """执行语音切分逻辑 (对应 asr_cut.py)"""
        self._init_asr()
        wav_path = self._extract_audio(video_path)
        
        print("🎙️ Running ASR Inference...")
        res = self.asr_model.generate(wav_path, return_raw_text=True, return_timestamp=True)[0]
        
        # 1. 计算语音密度用于路由判断
        total_speech_ms = 0
        if 'timestamp' in res:
             # timestamp 格式通常是 [[start, end], [start, end]...]
            for t in res['timestamp']:
                total_speech_ms += (t[1] - t[0])
        
        # 获取视频总时长 (估算)
        video_len_ms = res['timestamp'][-1][1] if res['timestamp'] else 1
        speech_ratio = total_speech_ms / video_len_ms
        
        print(f"📊 Speech Density: {speech_ratio:.2%}")

        # 如果语音占比极低，直接返回 None，触发视觉路由
        if speech_ratio < self.speech_density_threshold:
            print("⚠️ 语音占比过低，判定为集锦/纯音乐视频，切换至视觉管线。")
            return None

        # 2. 如果是语音视频，执行切分逻辑
        results = []
        raw_text = [x for x in res["raw_text"].split(' ') if x]
        timestamps = res["timestamp"]
        punc_array = res["punc_array"]
        text_segments = res["text"] # 带标点的文本列表

        para = ""
        ti = 0
        dur_start = -1
        
        # asr_cut 切分逻辑
        for i in range(len(raw_text)):
            if len(para) == 0:
                dur_start = timestamps[i][0]
            
            para += raw_text[i]
            punc_signal = int(punc_array[i])
            
            if punc_signal != 1: # 非静音/非连续
                if ti < len(text_segments):
                    para += text_segments[ti][-1] if text_segments[ti] else "" # 加上标点
                ti += 1
                
                # 核心切分条件：句号(3)且长度够，或者强制切分(4)
                if (punc_signal == 3 and len(para) > 10) or punc_signal == 4:
                    dur_end = timestamps[i][1]
                    results.append({
                        "text": para,
                        "beginTime": round(dur_start/1000, 2),
                        "endTime": round(dur_end/1000, 2),
                        "start_str": self._ms_to_hmsms(dur_start),
                        "end_str": self._ms_to_hmsms(dur_end),
                        "duration": int(dur_end - dur_start),
                        "theme": "speech",
                        "type": "Semantic",
                        "by": "FunASR_Router"
                    })
                    para = ""
        
        return results

    def _get_clip_embedding(self, frame_img):
        """提取单帧的 CLIP Embedding"""
        inputs = self.clip_processor(images=frame_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.clip_model.get_image_features(**inputs)
            emb /= emb.norm(p=2, dim=-1, keepdim=True)
        return emb.cpu().numpy().flatten()

    def run_visual_pipeline(self, video_path):
        """
        执行视觉切分逻辑：
        1. PySceneDetect 获取候选切点 (快)
        2. CLIP ViT 验证切点前后的语义一致性 (准)
        """
        print("🎬 Running Visual Pipeline (PySceneDetect + CLIP)...")
        
        # --- Step 1: PySceneDetect ---
        video = open_video(video_path)
        scene_manager = SceneManager()
        # 游戏视频 threshold 建议 27-30
        scene_manager.add_detector(ContentDetector(threshold=27.0, min_scene_len=30)) 
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()
        
        if not scene_list:
            return []

        # --- Step 2: CLIP Refinement (修边界) ---
        self._init_clip()
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        final_scenes = []
        current_start_frame = scene_list[0][0].get_frames()
        current_end_frame = scene_list[0][1].get_frames() # 候选结束点
        
        # 遍历所有 PyDetect 找到的切点，用 CLIP 验证
        # 逻辑：检查 Scene A 的末尾 和 Scene B 的开头 是否语义相似
        for i in range(1, len(scene_list)):
            next_scene_start = scene_list[i][0].get_frames()
            next_scene_end = scene_list[i][1].get_frames()
            
            # 读取切点前后的帧 (前一镜头的最后一帧 vs 后一镜头的第一帧)
            # 为了容错，取切点前后各偏移几帧
            frame_a_idx = current_end_frame - 5
            frame_b_idx = next_scene_start + 5
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_a_idx)
            ret1, frame_a = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_b_idx)
            ret2, frame_b = cap.read()
            
            is_same_scene = False
            if ret1 and ret2:
                # 转 RGB
                frame_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2RGB)
                frame_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2RGB)
                
                # 计算相似度
                emb_a = self._get_clip_embedding(frame_a)
                emb_b = self._get_clip_embedding(frame_b)
                similarity = 1 - cosine(emb_a, emb_b)
                
                # 如果相似度非常高 (例如 > 0.85)，说明虽然像素变了(PyDetect触发)，但语义没变
                # 比如：游戏UI弹窗、闪光弹、人物简单转身
                if similarity > self.clip_similarity_threshold:
                    is_same_scene = True
                    # print(f"  Merged split at frame {current_end_frame} (Sim: {similarity:.2f})")

            if is_same_scene:
                # 合并：当前段的结束点延长到下一段的结束点
                current_end_frame = next_scene_end
            else:
                # 确认切分：保存当前段，开启新段
                final_scenes.append((current_start_frame, current_end_frame))
                current_start_frame = next_scene_start
                current_end_frame = next_scene_end
        
        # 添加最后一段
        final_scenes.append((current_start_frame, current_end_frame))
        cap.release()
        
        # --- Step 3: 格式化输出 ---
        formatted_results = []
        for start_f, end_f in final_scenes:
            begin_sec = round(start_f / fps, 2)
            end_sec = round(end_f / fps, 2)
            duration_ms = int((end_sec - begin_sec) * 1000)
            
            if duration_ms < 1000: # 过滤小于1秒的碎片
                continue
                
            formatted_results.append({
                "text": "", # 视觉模式无文本
                "beginTime": begin_sec,
                "endTime": end_sec,
                "start_str": self._ms_to_hmsms(begin_sec * 1000),
                "end_str": self._ms_to_hmsms(end_sec * 1000),
                "duration": duration_ms,
                "theme": "visual_highlight",
                "type": "Visual",
                "by": "PySceneDetect_CLIP_Refined"
            })
            
        print(f"✅ Visual processing done. Original scenes: {len(scene_list)}, Refined scenes: {len(formatted_results)}")
        return formatted_results

    def process(self, video_path):
        """
        主入口函数：智能切分
        """
        if not os.path.exists(video_path):
            print(f"❌ Video not found: {video_path}")
            return []

        print(f"\n====== Processing: {os.path.basename(video_path)} ======")
        
        # 1. 尝试 ASR 路径
        try:
            asr_results = self.run_asr_pipeline(video_path)
            if asr_results is not None:
                print("✅ 已采用 ASR 语义拆条方案。")
                return asr_results
        except Exception as e:
            print(f"⚠️ ASR pipeline failed or skipped: {e}")
            print("🔄 Switching to Visual pipeline...")

        # 2. 回退到 Visual 路径
        visual_results = self.run_visual_pipeline(video_path)
        return visual_results

# ================= 使用示例 =================
if __name__ == "__main__":
    # 配置你的模型目录
    MODELS_DIR = "/workspace/work/zhipeng16/yolo8-plus-iopaint/models/"
    
    # 初始化切割器
    cutter = SmartVideoCutter(model_dir=MODELS_DIR)
    
    # 测试案例 1: 教学视频 (应走 ASR)
    # video_path_1 = '/workspace/work/zhipeng16/datasets/videos/4分钟游戏教学视频.mp4'
    # res1 = cutter.process(video_path_1)
    # print(json.dumps(res1[:2], indent=2, ensure_ascii=False))
    
    # 测试案例 2: 集锦视频 (应走 Visual)
    video_path_2 = '/workspace/work/zhipeng16/datasets/videos/3分钟游戏集锦.mp4'
    res2 = cutter.process(video_path_2)
    
    # 保存结果
    output_path = "./smart_cut_result.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(res2, f, indent=4, ensure_ascii=False)
    print(f"结果已保存: {output_path}")