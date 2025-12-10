#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/12/10 19:22
# @Author  : zhipeng16
# @Email   : zzzp50@ustc.edu
# @File    : pipeline_video_split_v2.py
# @Usage   : Describe the file's purpose
import os
import subprocess
import json
import numpy as np
import cv2
import torch
import gc
import sys,os
sys.path.append(os.getcwd())
from funasr import AutoModel
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from transformers import CLIPProcessor, CLIPModel
from scipy.spatial.distance import cosine
from util.mylogging import get_logger
class SmartVideoCutter:
    def __init__(self, model_dir = 'models/', device="cuda", log = get_logger("util_log/pipeline_video_split")):
        self.model_dir = model_dir
        self.device = device
        self.asr_model = None
        self.clip_model = None
        self.clip_processor = None
        
        # 默认阈值配置;语音占比<0.5则走视觉
        self.speech_density_threshold = 0.5
        self.clip_similarity_threshold = 0.75
        self.log = log
        
    def _ms_to_hmsms(self, ms):
        """辅助格式化"""
        ms = int(ms)
        return f"{ms//3600000:02d}:{ms%3600000//60000:02d}:{ms%60000//1000:02d}.{ms%1000:03d}"

    def _init_asr(self):
        if self.asr_model is None:
            self.log.info("🚀 Loading FunASR Model...")
            self.asr_model = AutoModel(
                model=os.path.join(self.model_dir, "speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"),
                vad_model=os.path.join(self.model_dir, "speech_fsmn_vad_zh-cn-16k-common-pytorch"),
                punc_model=os.path.join(self.model_dir, "ct-punc"),
                device=self.device,
                disable_update=True
            )

    def _init_clip(self):
        if self.clip_model is None:
            self.log.info("👁️ Loading CLIP Model...")
            model_name = "openai/clip-vit-base-patch32"
            self.clip_model = CLIPModel.from_pretrained(model_name, cache_dir=self.model_dir)
            self.clip_processor = CLIPProcessor.from_pretrained(model_name, cache_dir=self.model_dir)
            self.clip_model.to(self.device)
            self.clip_model.eval()

    def _safe_release_clip(self):
        if self.clip_model is not None:
            del self.clip_model
            del self.clip_processor
            self.clip_model = None
            self.clip_processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def _extract_audio(self, video_path):
        wav_path = os.path.splitext(video_path)[0] + ".wav"
        if not os.path.exists(wav_path):
            subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", 
                            "-i", video_path, "-ac", "1", "-ar", "16000", wav_path], check=True)
        return wav_path

    # ================= 阶段 1: 基础生成器 =================

    def run_asr_stage1(self, video_path):
        self._init_asr()
        wav_path = self._extract_audio(video_path)
        self.log.info("🎙️ Running ASR Inference...")
        try:
            res = self.asr_model.generate(wav_path, return_raw_text=True, return_timestamp=True)[0]
        except Exception as e:
            self.log.info(f"❌ ASR Runtime Error: {e}")
            return None

        total_speech_ms = sum([t[1]-t[0] for t in res.get('timestamp', [])])
        video_len_ms = res['timestamp'][-1][1] if res.get('timestamp') else 1
        
        if (total_speech_ms / video_len_ms) < self.speech_density_threshold:
            self.log.info(f"⚠️ 语音占比 ({total_speech_ms/video_len_ms:.2%}) 过低，切换至视觉管线。")
            return None

        segments = []
        raw_text = [x for x in res["raw_text"].split(' ') if x]
        timestamps = res["timestamp"]
        punc_array = res["punc_array"]
        text_segs = res["theme"]
        
        para, ti, dur_start = "", 0, -1
        for i in range(len(raw_text)):
            if len(para) == 0: dur_start = timestamps[i][0]
            para += raw_text[i]
            punc = int(punc_array[i])
            if punc != 1:
                if ti < len(text_segs): para += text_segs[ti][-1] if text_segs[ti] else ""
                ti += 1
                if (punc == 3 and len(para) > 10) or punc == 4:
                    dur_end = timestamps[i][1]
                    segments.append({
                        "beginTime": round(dur_start/1000, 2),
                        "endTime": round(dur_end/1000, 2),
                        "theme": para,
                        "type": "Semantic",
                        "by": "stage1"
                    })
                    para = ""
        return segments

    def run_visual_stage1(self, video_path):
        self.log.info("🎬 Running Visual Detection (Stage 1)...")
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=27.0, min_scene_len=30))
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()
        
        segments = []
        for start_t, end_t in scene_list:
            segments.append({
                "beginTime": round(start_t.get_seconds(), 2),
                "endTime": round(end_t.get_seconds(), 2),
                "theme": "",
                "type": "Scene",
                "by": "stage1"
            })
        return segments

    # ================= 阶段 2: 带约束的 CLIP 合并 =================

    def _refine_segments_with_clip(self, video_path, raw_segments, part_time_limit):
        """
        CLIP 边界融合器 (支持时长约束)
        Args:
            part_time_limit: 合并后的片段允许的最大时长(秒)。如果超过此值，禁止合并。
        """
        if not raw_segments or len(raw_segments) < 2:
            return raw_segments

        self.log.info(f"🧠 Running CLIP Refinement (Max duration limit: {part_time_limit}s)...")
        
        try:
            self._init_clip()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.log.info("❌ GPU OOM during CLIP load. Skipping refinement.")
                self._safe_release_clip()
                return raw_segments
            raise e

        cap = cv2.VideoCapture(video_path)
        merged_segments = []
        current_seg = raw_segments[0]

        try:
            for next_seg in raw_segments[1:]:
                # === 新增约束逻辑 1：时长检查 ===
                # 预测合并后的结束时间
                potential_end_time = next_seg['endTime']
                potential_duration = potential_end_time - current_seg['beginTime']
                
                # 如果合并后的时长超过了期望的 part_time，直接不合并，强制断开
                if potential_duration > part_time_limit:
                    merged_segments.append(current_seg)
                    current_seg = next_seg
                    continue # 跳过 CLIP 检测，直接进入下一轮

                # === 常规 CLIP 检测逻辑 ===
                t_end_curr = current_seg['endTime'] * 1000
                t_start_next = next_seg['beginTime'] * 1000
                
                time_gap = t_start_next - t_end_curr
                if time_gap > 2000: # 物理间隔过大，不合并
                    merged_segments.append(current_seg)
                    current_seg = next_seg
                    continue

                # 读取用于对比的帧
                cap.set(cv2.CAP_PROP_POS_MSEC, t_end_curr - 200)
                ret1, frame_a = cap.read()
                cap.set(cv2.CAP_PROP_POS_MSEC, t_start_next + 200)
                ret2, frame_b = cap.read()

                is_similar = False
                if ret1 and ret2:
                    with torch.no_grad():
                        inputs = self.clip_processor(images=[frame_a, frame_b], return_tensors="pt").to(self.device)
                        embs = self.clip_model.get_image_features(**inputs)
                        embs /= embs.norm(p=2, dim=-1, keepdim=True)
                        sim = torch.cosine_similarity(embs[0].unsqueeze(0), embs[1].unsqueeze(0))
                        
                        if sim.item() > self.clip_similarity_threshold:
                            is_similar = True
                
                if is_similar:
                    # 执行合并
                    current_seg['endTime'] = next_seg['endTime']
                    if current_seg["theme"] and next_seg["theme"]:
                        current_seg["theme"] += " " + next_seg["theme"]
                    # 标记来源为合并
                    if "Merged" not in current_seg['by']:
                        current_seg['by'] += "_Merged"
                else:
                    merged_segments.append(current_seg)
                    current_seg = next_seg
            
            merged_segments.append(current_seg)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.log.info("⚠️ GPU OOM. Returning raw segments.")
                self._safe_release_clip()
                return raw_segments
            else:
                self.log.info(f"⚠️ CLIP Error: {e}")
                return raw_segments
        finally:
            cap.release()

        return merged_segments

    def _enrich_metadata(self, segments, material_id, video_duration):
        """补充外部传入的 metadata"""
        for seg in segments:
            dur_ms = int((seg['endTime'] - seg['beginTime']) * 1000)
            seg['duration'] = dur_ms
            seg['start_str'] = self._ms_to_hmsms(seg['beginTime'] * 1000)
            seg['end_str'] = self._ms_to_hmsms(seg['endTime'] * 1000)
            
            # 添加/修正 metadata
            seg['material_id'] = material_id
            
            # 简单校验，防止 endTime 超过视频总时长 (如果有偏差)
            if video_duration > 0 and seg['endTime'] > video_duration:
                 seg['endTime'] = video_duration
                 
        return segments

    # ================= 主函数 (Updated) =================

    def process(self, input_path="", split_num=5, part_time=45, video_duration=167, material_id="zhipeng16_test", mode = 1):
        """
        执行智能切分工作流，支持数量和时长约束。
        
        Args:
            input_path: 视频文件路径
            split_num: 期望的最小片段数。如果 Stage1 结果少于此数，则跳过合并。
            part_time: 期望的最大单段时长(秒)。CLIP 合并时不会让片段超过此时长。
            video_duration: 视频总时长(秒)，用于校验。
            material_id: 业务ID，透传给结果。
        """
        # 参数校验
        if not input_path or not os.path.exists(input_path):
            self.log.info(f"❌ Input path invalid: {input_path}")
            return []

        self.log.info(f"\n====== Processing: {os.path.basename(input_path)} ======")
        self.log.info(f"🔧 Constraints: Min Splits={split_num}, Max Part Time={part_time}s")

        # --- Step 1: Base Segmentation (ASR or Visual) ---
        stage1_results = None
        try:
            stage1_results = self.run_asr_stage1(input_path)
        except Exception:
            stage1_results = None
            
        if stage1_results is None:
            stage1_results = self.run_visual_stage1(input_path)
            
        if not stage1_results:
            self.log.info("❌ No segments found in Stage 1.")
            return []
            
        current_split_count = len(stage1_results)
        self.log.info(f"✅ Stage 1 Done. Found {current_split_count} segments.")

        # --- Step 2: Conditional CLIP Refinement ---
        final_results = stage1_results
        
        # 约束逻辑 1：如果切分数量已经很少了 (小于期望的 split_num)，则禁止合并
        if current_split_count < split_num or not mode:
            self.log.info(f"⏩ Skip CLIP Merge: Current count ({current_split_count}) < Expected split_num ({split_num}).")
        else:
            # 只有数量充足时，才进行合并，并传入 part_time 约束
            final_results = self._refine_segments_with_clip(
                video_path=input_path, 
                raw_segments=stage1_results, 
                part_time_limit=part_time  # 传入时长限制
            )
            self.log.info(f"✅ Refinement Done. {len(stage1_results)} -> {len(final_results)} segments.")

        # --- Step 3: Final Format & Metadata ---
        return self._enrich_metadata(final_results, material_id, video_duration)

# ================= 测试调用 =================
if __name__ == "__main__":
    MODELS_DIR = "models/"
    cutter = SmartVideoCutter(model_dir=MODELS_DIR)
    
    # 模拟外部调用
    video_file = '/workspace/work/zhipeng16/datasets/videos/3分钟游戏集锦.mp4'
    video_file = '/workspace/work/zhipeng16/datasets/videos/4分钟游戏教学视频.mp4'
    
    results = cutter.process(
        input_path=video_file,
        split_num=5,           # 期望至少保留5段，如果不够5段就不合并了
        part_time=45,          # 期望每段不超过15秒 (测试用，实际你设的是45)
        video_duration=180,    # 假设视频3分钟
        material_id="GAME_HIGHLIGHT_001"
    )
    
    print(json.dumps(results[:2], indent=2, ensure_ascii=False))