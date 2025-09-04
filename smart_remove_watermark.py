#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2025/9/4 16:25
# @Author  : zhipeng16
# @concat: mintlzp@mail.ustc.edu.cn
# @Site:
# @File: smart_remove_watermark.py.py
# @Software: PyCharm
# describe：(Smart Version) 使用yolo8+iopaint结合，智能移除图片或视频中的水印

import cv2
import os
import time
from collections import Counter
from tqdm import tqdm

import configs
from smart_yolo_utils import SmartYOLOUtils
from smart_iopaint_utils import SmartIOPaintApiUtil, SmartIOPaintCmdUtil
import numpy as np

# --- 全局配置 ---
output_dir = configs.cache_dir  # 输出目录
model_path = f"{configs.models_dir}/last.pt"  # yolo模型路径
device = configs.device  # 设备类型

# 【推荐】是否使用iopaint的api方式去除水印。对于视频处理，强烈建议设为True。
USE_IOPAINT_API = configs.USE_IOPAINT_API

# --- 初始化工具对象 ---
yolo_obj = SmartYOLOUtils(model_path)
iopaint_obj = SmartIOPaintApiUtil(device=device) if USE_IOPAINT_API else SmartIOPaintCmdUtil(device=device)


def process_image(image_path: str):
    """处理单张图片"""
    print(f"开始处理图片: {image_path}")
    if not os.path.exists(image_path):
        print(f"错误: 图片文件不存在 {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图片 {image_path}")
        return

    bboxes = yolo_obj.get_bboxes(image)
    if not bboxes.any():
        print("未检测到水印，跳过处理。")
        return

    mask = iopaint_obj.create_mask(image, bboxes)
    iopaint_obj.erase_watermark(image_path, mask, output_dir)
    print("图片处理完成。")


def process_video(video_path: str, mode: str = 'fixed'):
    """
    处理视频
    :param video_path: 视频文件路径
    :param mode: 'fixed' (固定模式) 或 'realtime' (实时模式)
    """
    print(f"开始处理视频: {video_path} (模式: {mode})")
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在 {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("错误: 无法打开视频文件")
        return

    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 设置输出视频
    output_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_processed.mp4"
    output_path = os.path.join(output_dir, output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    fixed_mask = None

    # --- 固定模式逻辑 ---
    if mode == 'fixed':
        print("固定模式: 正在分析视频前3秒以确定通用mask...")
        detected_bboxes = []
        # 分析前3秒的帧
        for i in range(int(fps * 3)):
            ret, frame = cap.read()
            if not ret:
                break
            bboxes = yolo_obj.get_bboxes(frame)
            if bboxes.any():
                # 将bbox列表转换为可哈希的元组，以便计数
                bbox_tuple = tuple(map(tuple, np.round(bboxes, 2)))
                detected_bboxes.append(bbox_tuple)

        if detected_bboxes:
            # 找到出现次数最多的bbox组合
            most_common_bboxes_tuple = Counter(detected_bboxes).most_common(1)[0][0]
            most_common_bboxes = np.array([list(b) for b in most_common_bboxes_tuple])

            # 创建一个与视频尺寸相同的空白图像来生成mask
            dummy_image = np.zeros((height, width, 3), dtype=np.uint8)
            fixed_mask = iopaint_obj.create_mask(dummy_image, most_common_bboxes)
            print("通用mask已生成。")
        else:
            print("警告: 视频前3秒未检测到任何水印，将不进行处理。")
            cap.release()
            out.release()
            return

        # 将视频指针重置到开头
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # --- 逐帧处理 ---
    with tqdm(total=frame_count, desc="处理进度") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = frame
            if mode == 'fixed' and fixed_mask is not None:
                if USE_IOPAINT_API:
                    processed_frame = iopaint_obj.erase_watermark_from_image_data(frame, fixed_mask)
                else:
                    print("警告: 命令行模式不支持高效的视频处理，结果可能非常慢。")
                    # 此处为命令行模式的低效实现（不推荐）
                    temp_frame_path = f"{configs.cache_dir}/temp_frame.png"
                    cv2.imwrite(temp_frame_path, frame)
                    processed_frame = iopaint_obj.erase_watermark(temp_frame_path, fixed_mask, configs.cache_dir)
                    os.remove(temp_frame_path)

            elif mode == 'realtime':
                bboxes = yolo_obj.get_bboxes(frame)
                if bboxes.any():
                    mask = iopaint_obj.create_mask(frame, bboxes)
                    if USE_IOPAINT_API:
                        processed_frame = iopaint_obj.erase_watermark_from_image_data(frame, mask)
                    else:
                        print("警告: 命令行模式不支持高效的视频处理。")
                        processed_frame = frame  # 实时模式下命令行几乎不可用，直接跳过
                else:
                    processed_frame = frame

            out.write(processed_frame if processed_frame is not None else frame)
            pbar.update(1)

    cap.release()
    out.release()
    print(f"\n视频处理完成，已保存至: {output_path}")


if __name__ == "__main__":
    """
    使用示例
    """
    if USE_IOPAINT_API:
        print("=====【温馨提示】使用iopaint的api方式去除水印，需要先运行iopaint服务：python iopaint_server.py 或使用自定义的IOPaint服务=====\n")

    os.makedirs(output_dir, exist_ok=True)

    # --- 选择一个输入文件进行测试 ---
    # input_path = f"{configs.images_dir}/test1.png"
    input_path = f"/data2/zhipeng16/datasets/output_frames/video/source.mp4"  # 修改为你的视频路径

    # --- 根据文件类型自动选择处理器 ---
    start_time = time.time()
    if input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        process_image(input_path)
    elif input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # 对于视频，选择模式: 'fixed' 或 'realtime'
        process_video(input_path, mode='fixed')
        # process_video(input_path, mode='realtime')
    else:
        print(f"不支持的文件类型: {input_path}")

    end_time = time.time()
    print(f"\n全部完成，总耗时: {end_time - start_time:.2f} 秒")