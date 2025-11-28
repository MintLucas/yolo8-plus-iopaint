import cv2
import os
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import time
from datetime import timedelta
from tqdm import tqdm

import cv2
import os
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import time
from datetime import timedelta
from tqdm import tqdm
import os
import numpy as np
import base64
import cv2
import requests

# 初始化 requests session（复用连接，提高接口调用效率）
ocr_session = requests.Session()

def get_four_corner_boxes(frame_shape, scale=0.1):
    """
    生成帧的四个角落区域（每个角落的 1/10 大小）
    返回：[(x_min, y_min, x_max, y_max), ...]（像素坐标，xyxy 格式）
    """
    h, w = frame_shape[:2]  # 帧的高度、宽度（OpenCV 格式：HWC）
    box_w = int(w * scale)  # 角落区域宽度（1/10 视频宽）
    box_h = int(h * scale)  # 角落区域高度（1/10 视频高）
    
    return [
        (0, 0, box_w, box_h),                  # 左上角
        (w - box_w, 0, w, box_h),              # 右上角
        (0, h - box_h, box_w, h),              # 左下角
        (w - box_w, h - box_h, w, h)           # 右下角
    ]

def frame_to_base64(frame):
    """
    将 OpenCV 帧（BGR 格式）转换为 Base64 字符串（OCR 接口要求的输入格式）
    """
    # 转换 BGR 为 RGB（部分 OCR 接口要求 RGB 格式，若接口支持 BGR 可跳过）
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 编码为 JPEG 格式
    ret, buffer = cv2.imencode('.jpg', rgb_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ret:
        raise ValueError("帧编码为 JPEG 失败")
    # 转换为 Base64
    b64 = base64.b64encode(buffer).decode('utf-8')
    return b64

def call_ocr_api(b64_data):
    """
    调用指定的 OCR 接口，返回识别结果
    接口地址：http://ip1.push.weibo.cn:21126/get_ocr
    输入：Base64 编码的图片字符串
    输出：接口返回的 JSON 结果
    """
    ocr_url = 'http://ip1.push.weibo.cn:21126/get_ocr'
    payload = {'data': b64_data}
    try:
        response = ocr_session.post(ocr_url, json=payload, timeout=10)
        response.raise_for_status()  # 抛出 HTTP 错误（如 404、500）
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"OCR 接口调用失败：{str(e)}")
        return [[]]  # 异常时返回默认空结果
def extract_bilibili_box(ocr_result):
    """
    从 OCR 结果中提取含「bilibili」关键词的文本框坐标
    输入：OCR 原始结果（嵌套列表格式）
    输出：匹配的文本框坐标（[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]），未匹配返回 None
    """
    # 目标关键词（兼容大小写和轻微识别误差）
    target_keywords = {'bilibili', 'BiliBili', 'BILIBILI', 'bilibil'}
    
    # 校验 OCR 结果格式（避免格式异常报错）
    if not ocr_result or not isinstance(ocr_result, list):
        print("OCR 结果格式异常，无法提取文本框")
        return None
    
    # 遍历 OCR 结果（你的 OCR 结果是 5 层嵌套，最内层是 [文本框坐标, [文本, 置信度]]）
    # 逐层解析：外层列表 → 内层列表 → [坐标列表, [文本, 置信度]]
    for outer_item in ocr_result:
        if not isinstance(outer_item, list):
            continue
        for inner_item in outer_item:
            # 每个 inner_item 格式：[[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], [文本, 置信度]]
            if len(inner_item) != 2:
                continue  # 跳过格式不匹配的项
            
            text_box = inner_item[0]  # 文本框坐标（[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]）
            text_info = inner_item[1]  # 文本信息（[文本, 置信度]）
            
            # 校验文本信息格式
            if not isinstance(text_info, list) or len(text_info) < 1:
                continue
            
            # 提取文本并匹配关键词
            detected_text = str(text_info[0]).strip()
            if any(keyword in detected_text for keyword in target_keywords):
                adjust_value = -120  # x值减去110
                # 调整第一个坐标的x值（确保不小于0）
                text_box[0][0] = max(0, text_box[0][0] + adjust_value)
                # 调整第四个坐标的x值（确保不小于0）
                text_box[3][0] = max(0, text_box[3][0] + adjust_value)               
                print(f"成功匹配关键词：{detected_text}，对应文本框：{text_box}")
                return text_box  # 返回第一个匹配的文本框（若需所有匹配项，可改为收集到列表中）
    
    # 未匹配到关键词
    print("OCR 结果中未找到 bilibili 相关文本框")
    return None

def textbox_to_yolo(text_box, frame_shape):
    """
    将 OCR 文本框坐标（4个顶点）转换为 YOLO 格式 [x_center, y_center, width, height]
    输入：
        text_box: OCR 提取的文本框 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        frame_shape: 帧的形状 (h, w, c)
    输出：YOLO 格式列表
    """
    # 提取所有 x 和 y 坐标，计算边界框（x_min, y_min, x_max, y_max）
    xs = [point[0] for point in text_box]
    ys = [point[1] for point in text_box]
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    
    # 转换为 YOLO 格式
    h, w = frame_shape[:2]
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    
    return [x_center, y_center, width, height]

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

# def process_video_for_watermark(video_path, model_path, frame_rate=2, conf=0.2, device='0'):
#     # 调用提取帧的函数
#     frames = extract_frames(video_path, frame_rate, "frames")  # 提取视频帧

#     # 检查是否成功提取帧
#     if len(frames) == 0:
#         print("没有提取到任何帧！请检查视频路径和视频文件。")
#         return None

#     # 进行YOLO检测
#     detection_results = detect_watermark_in_frames(frames, model_path, conf, device)  
#     best_box = vote_for_best_box(detection_results, iou_threshold=0.3)  # 使用更低的 IoU 阈值


#     # ========== 兜底策略：YOLO 未检测到，调用 OCR 接口 ==========
#     if best_box is None:
#         print("YOLO 未检测到水印，启动 OCR 兜底策略...")
        
#         # 取第一帧（已提取 30 帧，第一帧效率最高）
#         first_frame = frames[0]
#         frame_shape = first_frame.shape  # (h, w, c)
#         # print(f"帧尺寸: {frame_shape[1]}x{frame_shape[0]}")  # 打印宽x高以供调试 864 * 1920 宽 * 高
#         corner_boxes = get_four_corner_boxes(frame_shape, scale=0.3)  # 四个角落区域
        
#         # 遍历四个角落，逐个调用 OCR 接口
#         for corner_box in corner_boxes:
#             x_min, y_min, x_max, y_max = corner_box
#             # 裁剪角落区域（减少传输数据量，提高 OCR 准确率）
#             corner_frame = first_frame[y_min:y_max, x_min:x_max]
            
#             # 转换为 Base64（OCR 接口要求）
#             b64_data = frame_to_base64(corner_frame)
            
#             # 调用 OCR 接口
#             ocr_result = call_ocr_api(b64_data)
#             print(f"OCR 结果: {ocr_result}")  # 打印 OCR 结果以供调试

#             text_box = extract_bilibili_box(ocr_result)
#             # 检查是否命中关键词
#             if text_box:
#                 # 命中后转换为 YOLO 格式，作为 best_box 返回（只返回第一个命中的区域）
#                 best_box = textbox_to_yolo(text_box, frame_shape)
#                 print(f"兜底策略成功！生成 YOLO 格式 best_box：{best_box}")
#                 break  # 找到一个即可，无需检查其他角落
        
#         if best_box is None:
#             print("OCR 兜底策略未命中任何关键词，返回 None")


#     return best_box


def process_video_for_watermark(video_path, model_path, frame_rate=2, conf=0.2, device='0'):
    # 调用提取帧的函数
    frames = extract_frames(video_path, frame_rate, os.path.dirname(video_path))  # 提取视频帧

    # 检查是否成功提取帧
    if len(frames) == 0:
        print("没有提取到任何帧！请检查视频路径和视频文件。")
        return None

    # 进行YOLO检测
    detection_results = detect_watermark_in_frames(frames, model_path, conf, device)  
    best_box = vote_for_best_box(detection_results, iou_threshold=0.3)  # 使用更低的 IoU 阈值


    # ========== 兜底策略：YOLO 未检测到，调用 OCR 接口 ==========
    if best_box is None:
        print("YOLO 未检测到水印，启动 OCR 兜底策略...")

        ocr_detection_results = [] 
        max_frame_to_process = 10
        process_frames = min(max_frame_to_process, len(frames))  # 最多处理前10帧

        for frame_idx in range(process_frames):
            current_frame = frames[frame_idx]
            frame_shape = current_frame.shape  # (h, w, c)
            corner_boxes = get_four_corner_boxes(frame_shape, scale=0.3)
            
            # 遍历四个角落，逐个调用 OCR 接口
            for corner_box in corner_boxes:
                x_min, y_min, x_max, y_max = corner_box
                # 裁剪角落区域（减少传输数据量，提高 OCR 准确率）
                corner_frame = current_frame[y_min:y_max, x_min:x_max]
                try:
                    # 转换为 Base64（OCR 接口要求）
                    b64_data = frame_to_base64(corner_frame)
                    
                    # 调用 OCR 接口
                    ocr_result = call_ocr_api(b64_data)
                    # print(f"OCR 结果: {ocr_result}")  # 打印 OCR 结果以供调试

                    text_box = extract_bilibili_box(ocr_result)
                    # 检查是否命中关键词
                    if text_box:
                        # 命中后转换为 YOLO 格式，作为 best_box 返回（只返回第一个命中的区域）
                        current_frame_boxes = textbox_to_yolo(text_box, frame_shape)
                        print(f"兜底策略成功！生成 YOLO 格式 current_frame_boxes{current_frame_boxes}")
                        break  # 找到一个即可，无需检查其他角落
                except Exception as e:
                    print(f"第 {frame_idx} 帧，角落区域 {corner_box} 处理失败：{str(e)}")
                    continue
            ocr_detection_results.append(current_frame_boxes)
        ocr_detection_results = [[box] for box in ocr_detection_results]
        # 调用聚类函数，从 OCR 检测结果中筛选最优 box
        if any(len(boxes) > 0 for boxes in ocr_detection_results):  # 若有至少一个帧检测到 box
            best_box = vote_for_best_box(ocr_detection_results, iou_threshold=0.3)
            if best_box is not None:
                print(f"\nOCR 兜底策略 - 聚类后最终 best_box：{best_box}")
            else:
                print("\nOCR 检测到 box，但聚类未形成有效结果，返回 None")
        else:
            print("\nOCR 兜底策略未命中任何关键词，返回 None") 
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
    video_path = "/data2/zhipeng16/git/yolo8-watermark-brand/runs/detect/video_original/test3.mp4"  # 输入你的视频文件路径
    model_path = "/data2/zhipeng16/git/yolo8-watermark-brand/runs/detect/train8/weights/last.pt"  # 输入你训练好的YOLO模型路径

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
        output_video_path = "/data2/zhipeng16/git/yolo8-watermark-brand/runs/detect/video_infer/test333.mp4"  # 输出视频路径
        print(f"开始生成结果视频: {output_video_path}")
        save_result_video(video_path, output_video_path, best_box)
        print(f"视频生成耗时: {timedelta(seconds=int(time.time() - start_time))}")
    else:
        print("没有检测到有效的水印框，处理失败。")
        print(f"总耗时: {timedelta(seconds=int(time.time() - start_time))}")

