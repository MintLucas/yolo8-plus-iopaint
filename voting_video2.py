#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/09/16 15:18
# @Author  : zhipeng16
# @Email   : zzzp50@ustc.edu
# @File    : voting_video2.py
# @Usage   : Describe the file's purpose


import cv2
import os
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import time
from datetime import timedelta
from tqdm import tqdm

def extract_frames(video_path, n, output_path):
    """
    从视频中提取帧并保存到指定路径
    :param video_path: 视频文件的路径
    :param n: 截图的时间间隔（秒）
    :param output_path: 保存截图的路径
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return []  # 返回空列表

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    frame_interval = int(fps * n)
    frames = []  # 初始化空列表来存储帧

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # if frame_count == 30:
        #     break  # 只处理前30帧
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_path, f"frame_{frame_count // frame_interval}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"保存帧: {frame_filename}")
            frames.append(frame)  # 将提取的帧添加到列表中

            # 一旦保存的帧数达到 30，立即跳出循环
            if len(frames) >= 30:
                print("达到 30 帧，提前结束")
                break
        frame_count += 1
    cap.release()

    # 打印帧的数量以供调试
    print(f"提取了 {len(frames)} 帧")
    # print(f"frames:{frames}")
    return frames  # 返回提取的帧列表

def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2
    inter_area = max(0, min(x1_max, x2_max) - max(x1, x2)) * max(0, min(y1_max, y2_max) - max(y1, y2))
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area

def detect_watermark_in_frames(frames, model_path, conf=0.2, device='0'):
    if isinstance(model_path, str):
        model = YOLO(model_path)
    else:
        model = model_path
    detection_results = []

    for frame in frames:
        # 对每一帧进行推理
        results = model(frame, conf=conf, device=device)
        # print(f"检测结果: {results}")  # 打印检测结果以供调试

        # 遍历每个检测结果 (每一张图片的多个框)
        for r in results:
            # r.boxes 是一个包含多个框的对象
            # print(f"当前结果的框: {r.boxes}")  # 打印当前结果的框以供调试
            if hasattr(r, 'boxes') and r.boxes is not None:
                # 使用 r.boxes.xywh 提取每个框的位置和尺寸（x, y, width, height）
                boxes = r.boxes.xywh.cpu().numpy()  # 转换为 numpy 数组，便于处理
                # print(f"检测到的框: {boxes}")  # 打印检测到的框以供调试
                detection_results.append(boxes)  # 将检测框位置添加到结果列表
            else:
                detection_results.append([])  # 没有检测框，添加空列表
    # print(f"detection_results长度:{len(detection_results)}")

    return detection_results



# def vote_for_best_box(detection_results, iou_threshold=0.3):
#     box_votes = defaultdict(int)
#     print("开始投票过程")

#     for frame_results in detection_results:
#         if len(frame_results) == 0:
#             continue  # 如果该帧没有检测到任何框，跳过

#         # 遍历该帧中的所有检测框
#         for i in range(len(frame_results)):
#             box1 = frame_results[i]
#             # for j in range(i + 1, len(frame_results)):
#             for j in range(len(frame_results)):
#                 if i == j:
#                     continue  # 跳过相同的框
#                 box2 = frame_results[j]

#                 # 计算 IoU
#                 iou_value = iou(box1, box2)
#                 print(f"计算 IoU: box1 = {box1}, box2 = {box2}, IoU = {iou_value}")

#                 if iou_value > iou_threshold:
#                     box_votes[tuple(box1)] += 1  # 投票增加
#                     print(f"框 {box1} 与 {box2} 有足够的重叠，增加投票")

#     # 返回最多投票的框
#     if box_votes:
#         best_box = max(box_votes, key=box_votes.get)
#         print(f"最优框: {best_box}")
#         return best_box
#     else:
#         print("没有找到有效的框")
#         return None

def vote_for_best_box(detection_results, iou_threshold=0.3):
    """
    聚类
    """
    
    # 收集所有检测框
    all_boxes = []
    for frame_idx, frame_boxes in enumerate(detection_results):
        for box in frame_boxes:
            all_boxes.append({
                'xywh': box,
                'frame_idx': frame_idx,
                'cluster_id': -1  # 初始未分类
            })
    
    if not all_boxes:
        print("没有检测到任何框")
        return None
    
    # 聚类：将IoU大于阈值的框归为同一类
    clusters = []
    for i, box_info in enumerate(all_boxes):
        if box_info['cluster_id'] != -1:
            continue
            
        # 创建新聚类
        cluster_id = len(clusters)
        clusters.append([box_info])
        box_info['cluster_id'] = cluster_id
        
        # 寻找同类框
        for j in range(i + 1, len(all_boxes)):
            other_box = all_boxes[j]
            if other_box['cluster_id'] != -1:
                continue
                
            iou_value = iou(box_info['xywh'], other_box['xywh'])
            if iou_value > iou_threshold:
                other_box['cluster_id'] = cluster_id
                clusters[cluster_id].append(other_box)
    
    # print(f"形成了 {len(clusters)} 个聚类")
    
    # 选择最大的聚类
    if clusters:
        # 按聚类大小排序
        clusters.sort(key=len, reverse=True)
        
        # print("\n聚类大小排名（前5名）:")
        # for i, cluster in enumerate(clusters[:5]):
            # print(f"聚类{i}: {len(cluster)} 个框")
        
        # 选择最大的聚类
        largest_cluster = clusters[0]
        # print(f"\n最大聚类包含 {len(largest_cluster)} 个框")
        
        # 计算聚类中心（平均框）
        cluster_boxes = [box_info['xywh'] for box_info in largest_cluster]
        avg_box = np.mean(cluster_boxes, axis=0)
        
        # print(f"聚类中心框: {avg_box}")
        return avg_box
    else:
        # print("没有形成任何聚类")
        return None



def process_video_for_watermark(video_path, model_path, frame_rate=2, conf=0.2, device='0'):
    # 调用提取帧的函数
    frames = extract_frames(video_path, frame_rate, "frames")  # 提取视频帧

    # 检查是否成功提取帧
    if len(frames) == 0:
        print("没有提取到任何帧！请检查视频路径和视频文件。")
        return None

    # 进行YOLO检测
    detection_results = detect_watermark_in_frames(frames, model_path, conf, device)  
    best_box = vote_for_best_box(detection_results, iou_threshold=0.3)  # 使用更低的 IoU 阈值
    return best_box


def draw_detection_on_frame(frame, best_box):
    """
    将最优框绘制到视频帧上
    :param frame: 当前视频帧
    :param best_box: 最优框 [x, y, w, h]
    :return: 绘制了框的帧
    """
    if best_box is not None:
        x, y, w, h = best_box
        # 绘制绿色框，线宽为2
        cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2 ), int(y + h / 2)), (0, 255, 0), 2)
    return frame


def save_result_video(input_video_path, output_video_path, best_box):
    """
    将最优框标注到原始视频上并保存结果
    """
    
    # 打开输入视频
    video = cv2.VideoCapture(input_video_path)
    if not video.isOpened():
        print(f"无法打开视频文件: {input_video_path}")
        return
    
    # 获取视频信息
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息: {frame_width}x{frame_height}, {fps:.2f}FPS, 总帧数: {total_frames}")
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    # # 使用更高效的编码器
    # fourcc = cv2.VideoWriter_fourcc(*'H264')  # H264编码，更高效
    # # 如果H264不可用，尝试其他编码
    # try:
    #     out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    # except:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    print("使用mp4v编码器")
    
    # 创建进度条
    pbar = tqdm(total=total_frames, desc="处理视频帧", unit="帧")
    
    frame_count = 0
    start_time = cv2.getTickCount()
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        # 绘制检测框
        if best_box is not None:
            frame = draw_detection_on_frame(frame, best_box)
        
        # 写入帧
        out.write(frame)
        frame_count += 1
        
        # 更新进度条
        pbar.update(1)
        
        # 每100帧显示一次预估剩余时间
        if frame_count % 100 == 0:
            current_time = cv2.getTickCount()
            elapsed_time = (current_time - start_time) / cv2.getTickFrequency()
            frames_per_second = frame_count / elapsed_time
            remaining_frames = total_frames - frame_count
            remaining_time = remaining_frames / frames_per_second if frames_per_second > 0 else 0
            
            pbar.set_postfix({
                '速度': f'{frames_per_second:.1f}FPS',
                '剩余时间': f'{remaining_time:.1f}秒'
            })
    
    # 关闭进度条
    pbar.close()
    
    # 释放资源
    video.release()
    out.release()
    
    # 计算总耗时
    end_time = cv2.getTickCount()
    total_time = (end_time - start_time) / cv2.getTickFrequency()
    
    print(f"处理完成！耗时: {total_time:.2f}秒, 平均速度: {total_frames/total_time:.1f}FPS")
    print(f"输出视频: {output_video_path}")


if __name__ == '__main__':
    # 设置视频路径和权重路径
    video_path = "/workspace/work/zhipeng16/datasets/videos/source.mp4"  # 输入你的视频文件路径
    model_path = "/workspace/work/zhipeng16/yolo8-plus-iopaint/models/last.pt"  # 输入你训练好的YOLO模型路径

    # 记录开始时间
    start_time = time.time()
    print(f"开始处理视频: {video_path}")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")


    # 处理视频并得到最优水印框
    best_box = process_video_for_watermark(video_path, model_path, frame_rate=2)

    # 记录检测完成时间
    detect_time = time.time()
    detect_duration = detect_time - start_time
    print(f"水印检测完成，耗时: {timedelta(seconds=int(detect_duration))}")

    # 输出检测结果的视频
    if best_box is not None:
        output_video_path = "/workspace/work/zhipeng16/yolo8-plus-iopaint/test333.mp4"  # 输出视频路径
        print(f"开始生成结果视频: {output_video_path}")
        save_result_video(video_path, output_video_path, best_box)
        print(f"视频生成耗时: {timedelta(seconds=int(time.time() - start_time))}")
    else:
        print("没有检测到有效的水印框，处理失败。")
        print(f"总耗时: {timedelta(seconds=int(time.time() - start_time))}")
