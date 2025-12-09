#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/09/18 11:36
# @Author  : zhipeng16
# @Email   : zzzp50@ustc.edu
# @File    : smart_video_detect.py
# @Usage   : Describe the file's purpose
import sys,os
sys.path.append(os.getcwd())
import json
from util.mylogging import get_logger
from smart_video.api_test_oss2 import oss_util
from util.token_util_new import token_fresh
import traceback,time,random
import cv2,base64
import os
import time
from collections import Counter
from tqdm import tqdm
import requests
import configs
import numpy as np
from voting_video_old import process_video_for_watermark
from ultralytics import YOLO
from moviepy.editor import *
from agi_server.iopaint_server_multi import DEVICES_INDICES,BASE_PORT
from smart_video.base_video_class import Base_video_class
# --- 全局配置 ---
# 文件所在的上层目录为项目目录os.path.dirname
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace(os.sep, '/')

# 缓存目录
cache_dir = f"{project_dir}/.cache"

# 模型目录
model_path = f"{project_dir}/models/last.pt"

class Smart_video_detect(Base_video_class):
    def __init__(self, log=get_logger("server_log/smart_video_detect")):
        self.log = log
        super().__init__(self.log)
        self.redis_key = self.__class__.__name__
        self.oss_util = oss_util(self.log)
        self.token_fresh = token_fresh(get_logger("llm_log/smart_video_detect"))
        self.task_status = {}
        from threading import Lock
        self.lock = Lock()
        from queue import Queue
        self.queue = Queue(1500)
        self.sft_yolo = YOLO(model_path)
        self.sft_yolo.to("cuda")
        self.base_machine_ip = "http://10.78.9.45:"
        self.base_machine_ip = "http://ip1.push.weibo.cn:"
        self.lama_inpaints = {
            #: [model,status,sum_fps,mask_local,video_path,deal_index]
            0:self.base_machine_ip+"8500",
            1:self.base_machine_ip+"8501",
            2:self.base_machine_ip+"8502",
            3:self.base_machine_ip+"8503"
        }
        self.lama_inpaints = {k:self.base_machine_ip+str(BASE_PORT+k) for k in range(len(DEVICES_INDICES))}
        self.lama_inpaints = {k:self.base_machine_ip+str(BASE_PORT) for k in range(len(DEVICES_INDICES))}
        self.log.info(f"一共起了{len(DEVICES_INDICES)}个服务, 卡分布{DEVICES_INDICES}, 服务ip: {self.base_machine_ip+str(BASE_PORT)}")
    def dashscope_video_detect(self, input_path = "",  material_id = "zhipeng16_test"):
        """
        使用dashscope进行视频检测和修复
        """
        try:
            material_id = str(material_id)
            self.log.info(f"start_process_task_id_{material_id}")
            self.redis_client.hset(self.redis_key, material_id, 'waiting')
            with self.lock:
                self.redis_client.hset(self.redis_key, material_id, 'downloading')
                self.log.info(f"downloading_process_task_id_{material_id}")
                local_path,file_name = self.oss_util.check_video_path(input_path)
                self.redis_client.hset(self.redis_key, material_id, 'processing_state1')
                
                self.redis_client.hset(self.redis_key, material_id, 'processing_state2')
                yolo_masked = self.get_yolo_masked(local_path)
                self.log.info(f"yolo_masked_process_task_id_{material_id} yolo_masked {yolo_masked}")
                
                start_time = time.time()
                dst_path = "tmp_data/tmp_detect_res/" + file_name + ".mp4"
                if yolo_masked is not None and (not isinstance(yolo_masked, np.ndarray) or yolo_masked.size > 0):
                    dashcope_res = self.deal_local_fast(vpath = local_path, dst_path = dst_path, material_id=material_id)
                else:
                    self.log.info(f"yolo_masked_process_task_id_{material_id} noneed_yolo_masked {yolo_masked}")
                    self.redis_client.hset(self.redis_key, material_id, f'no_need_mask')
                    return input_path
                self.redis_client.hset(self.redis_key, material_id, 'processing_state3')
                end_time = time.time()
                                
                elapsed_time_seconds = end_time -start_time
                elapsed_minutes = int(elapsed_time_seconds // 60)
                elapsed_seconds = int(elapsed_time_seconds % 60)
                # 打印运行时间
                self.log.info(f"运行时间: {elapsed_minutes} 分钟 {elapsed_seconds} 秒")
                
                
                self.redis_client.hset(self.redis_key, material_id, 'processing_state1')
                oss_bucket_object_key = "wces_detect/"+file_name
                self.log.info(f"upload_file_process_task_id_{material_id} oss_bucket_object_key: {oss_bucket_object_key}")
                up_status = self.oss_util.upload_file(local_file_path=dst_path, object_key=oss_bucket_object_key)
                sh_url_path = self.oss_util.get_url(oss_bucket_object_key)
                self.log.info(f"video_split_process_task_id_{material_id} detect_url:{sh_url_path}")
                self.redis_client.hset(self.redis_key, material_id, f'finish_all:{sh_url_path}')
                self.oss_util.delete_file(os.path.dirname(local_path))
                self.oss_util.delete_file(dst_path)
                return sh_url_path
        except Exception as e:
            error_info = traceback.format_exc()
            self.redis_client.hset(self.redis_key, material_id, f'exception')
            self.log.error(f"视频修复异常: {e}\n{error_info}")
            return 0

    def base_video_detect(self, video_url: str,  material_id = "zhipeng_test"):
        """
        发起视频处理任务。

        Args:
            video_url (str): 待处理视频的URL。
            material_id (str): 视频的唯一标识ID。

        Returns:
            dict: 接口返回的JSON数据，如果请求失败则返回None。
        """
        url = 'http://ip1.push.weibo.cn:18196/deal_local_video'
        payload = {
            'uploaded_video_url': video_url,
            'material_id': material_id
        }

        try:
            self.log.info(f"jt16正在发起视频处理任务，material_id: {material_id}")
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()  # 如果状态码不是200，则抛出HTTPError
            result = response.json()
            self.log.info(f"jt16任务发起成功，返回结果: {result}")
            return result
        except requests.exceptions.RequestException as e:
            self.log.error(f"jt16发起任务失败，material_id: {material_id}, 错误: {e}")
            return None
        except json.JSONDecodeError:
            self.log.error(f"jt16解析JSON失败，响应内容: {response.text}")
            return None

        # 2. 查询任务状态函数
    def check_task_status(self, material_id: str):
        """
        查询视频处理任务的状态，并将状态存入Redis。

        Args:
            material_id (str): 视频的唯一标识ID。

        Returns:
            str: 任务的当前状态，如果查询失败则返回None。
        """
        url = 'http://ip1.push.weibo.cn:18196/look_up_complex'
        params = {'dataid': material_id}
        try:
            self.log.info(f"正在查询jt16任务状态，material_id: {material_id}")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            status = data.get('status')
            if not status:
                self.log.warning(f"查询结果中jt16未找到'status'字段，返回数据: {data}")
                return None
                
            self.log.info(f"任务状态查询jt16成功，material_id: {material_id}, 状态: {status}")
            # 将状态存入Redis
            self.redis_client.hset(self.redis_key, material_id,  status)
        except requests.exceptions.RequestException as e:
            self.log.error(f"jt16查询任务状态失败，material_id: {material_id}, 错误: {e}")
            return None
        except json.JSONDecodeError:
            self.log.error(f"jt16解析JSON失败，响应内容: {response.text}")
            return None
        except Exception as e:
            self.log.error(f"jt16查询任务状态时发生未知错误: {e}")
            return None

    def _fallback_split(self, video_path, split_num, video_duration, part_time):
        """
        优化的兜底视频切分函数，在多模态模型调用失败时使用。
        参数：
        - video_path: 视频路径 (保留参数)
        - split_num: 期望切分的片段数量
        - video_duration: 视频总时长
        - part_time: 期望的单个片段时长，单位秒
        """
        #TODO
        pass

    def sub_work(self,cpuid, material_id):

        #model_use = self.lama_inpaints[cpuid]
        nn= 0
        while True:
            index,img = None,None
            try:
                index,img = self.queue.get(timeout = 10)
            except:
                pass
            if img is None:
                self.log.info(f'queue is none,all work finished, material_id:{material_id}  gpu:{cpuid}')
                break
            nn+=1
            if nn%100==0:
                self.log.info(f'queue processing on gpu {cpuid} material_id:{material_id} finsih_num:{nn}')
            erase_data = self.send_inpaint_request_from_data(img, self.mask_by_yolo_center, cpuid)
            if index<len(self.detect_results) and not erase_data is None:
                self.detect_results[index] = erase_data
            else:
                self.log.error(f'queue processing on gpu:{cpuid} deal error:{index} {erase_data==None}')


    def send_inpaint_request_from_data(self, image_data: np.ndarray, mask_data: np.ndarray, cpuid ):
        """
        ✨ (新增) 直接从内存中的图像和掩码数据发送修复请求，并返回处理后的图像数据。
        这是专为视频流处理优化的函数。

        :param image_data: numpy数组格式的原始图像 (BGR格式)
        :param mask_data: numpy数组格式的掩码图像 (单通道灰度图)
        :return: numpy数组格式的处理后图像，如果失败则返回None
        """
        # 将图片和掩码的numpy数组转换为base64字符串
        image_base64 = self.convert_image_data_to_base64(image_data)
        mask_base64 = self.convert_image_data_to_base64(mask_data)
        IOPAINT_SERVER_HOST = self.lama_inpaints[cpuid]
        # 构建请求的JSON body
        json_body = {
            "image": image_base64,
            "mask": mask_base64
        }

        # 发送POST请求
        try:
            response = requests.post(IOPAINT_SERVER_HOST+"/api/v1/inpaint", json=json_body, headers={
            "Content-Type": "application/json"
        }, timeout=30)
        except requests.ConnectionError:
            # 对于视频流处理，只打印错误而不是抛出异常，以避免中断整个视频处理流程
            print(f"\n错误: 无法连接到IOPaint服务: {IOPAINT_SERVER_HOST}。请确保服务已启动。")
            return None
        except Exception as e:
            print(f"请求过程中发生未知错误: {e}")
            return None

        # 检查响应状态码
        if response.status_code == 200:
            # 将返回的二进制图片数据解码为numpy数组
            image_array = np.frombuffer(response.content, np.uint8)
            inpainted_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            return inpainted_image
        else:
            print(f"请求失败，状态码：{response.status_code}, 响应内容: {response.text}")
            return None

    def convert_image_data_to_base64(self, image_data: np.ndarray, ext: str = '.png') -> str:
        """
        ✨ (新增) 将OpenCV图像数据（numpy数组）在内存中编码为base64字符串
        :param image_data: numpy数组格式的图像
        :param ext: 编码格式，默认为'.png'
        :return: base64编码的字符串
        """
        _, buffer = cv2.imencode(ext, image_data)
        return base64.b64encode(buffer).decode('utf-8')

    def deal_local_fast(self,vpath,dst_path,material_id):
        from moviepy.video.io.VideoFileClip import VideoFileClip
        s1 = VideoFileClip(vpath)
        from threading import Thread
        ts = []
        freame_num = int(s1.duration*s1.fps+10)

        self.detect_results = [x for x in range(freame_num)]
        for cpuid,server_ip in self.lama_inpaints.items():
            t = Thread(target=self.sub_work,args=(cpuid,material_id))
            t.start()
            ts.append(t)
        for index,one in enumerate(s1.iter_frames()):
            self.queue.put((index,one))
            if index%100==0:
                per = round(index*100/freame_num,3)
                per = f'{per}%'
                self.log.info(f'masking:{per} put_idx:{index} total_frame_num:{freame_num} qusize:{self.queue.qsize()}')
                self.redis_client.hset(self.redis_key, material_id, f'masking:{per}')
            #print(type(one))
        self.log.info(f'put_idx:{index} nowid:{material_id} total_frame_num:{freame_num} qusize:{self.queue.qsize()}')
        for t in ts:
            t.join()
        self.detect_results= self.detect_results[:index+1]
        self.log.info('finish all')
        clip = ImageSequenceClip(self.detect_results, fps=s1.fps)
        clip.set_audio(s1.audio)
        clip.write_videofile(dst_path)
        self.redis_client.hset(self.redis_key, material_id, 'mask finsih')
        return dst_path

    def create_mask_yolo_center(self, image, bboxes, padding=1):
        """
        根据 [x_center, y_center, width, height] 格式的边界框创建掩码图像。
        
        参数:
        :param image: 原始图像 (numpy array)
        :param bboxes: 边界框列表，格式为 [x_center, y_center, width, height]
        :param padding: 掩码的额外填充
        :return: 掩码图像 (numpy array)
        """
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        for bbox in bboxes:
            if len(bbox) >= 4:
                # 将[x_center, y_center, width, height] 转换为 [x1, y1, x2, y2]
                xc, yc, w, h = map(int, bbox[:4])
                x1 = int(xc - w / 2)
                y1 = int(yc - h / 2)
                x2 = int(xc + w / 2)
                y2 = int(yc + h / 2)

                # 根据原函数的 padding 逻辑调整坐标
                x1 = np.clip(x1 - padding, 0, width)
                y1 = np.clip(y1 - padding, 0, height)
                x2 = np.clip(x2 + padding, 0, width)
                y2 = np.clip(y2 + padding, 0, height)
                
                # 在掩码上绘制矩形
                mask[y1:y2, x1:x2] = 255
                
        return mask
    def get_yolo_masked(self, video_path: str, mode: str = 'fixed'):
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
        fixed_mask = None
        bboxes = process_video_for_watermark(video_path=video_path, model_path=self.sft_yolo, conf = 0.2)
        if bboxes is None or (isinstance(bboxes, np.ndarray) and bboxes.size == 0):
            self.log.info(f"{video_path}视频不存在水印")
            return ""
        dummy_image = np.zeros((height, width, 3), dtype=np.uint8)
        fixed_mask = self.create_mask_yolo_center(dummy_image, [bboxes])
        self.mask_by_yolo_center = fixed_mask

        return fixed_mask
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
                        os.remove(temp_frame_path) # 实时模式下命令行几乎不可用，直接跳过

                out.write(processed_frame if processed_frame is not None else frame)
                pbar.update(1)

        cap.release()
        out.release()
        print(f"\n视频处理完成，已保存至: {output_path}")

    def gen_video_detect(self, source_input):
        video_input = source_input.get("uploaded_video_url", "")
        split_num = source_input.get("split_num", 2)
        part_time = source_input.get("part_time", 45)
        mode = source_input.get("mode", 1)
        video_duration = source_input.get("video_duration", 167)
        task_id = source_input.get("material_id", "zhipeng_test")
        if mode == 1:
            res = self.dashscope_video_detect(video_input,  material_id=task_id)
        else:
            res = self.base_video_detect(video_input,  material_id=task_id)
        return res



if __name__ == '__main__':
    smart_video_split = Smart_video_detect()
    test_url = "https://wb-channel-aiclip-media.oss-cn-beijing.aliyuncs.com/cph/yt_dlp/4/59014/2025-09-23/v_cfe37bd91d6af648323ca73f3ffd8a92.mp4?x-oss-date=20251125T092513Z&x-oss-expires=604800&x-oss-signature-version=OSS4-HMAC-SHA256&x-oss-credential=LTAI5tHj9VxWxHdfk1rWYrdj%2F20251125%2Fcn-beijing%2Foss%2Faliyun_v4_request&x-oss-signature=f1bfa194248fd016e8f510c50440cd82d108c5eaf1eda9d64a40b6054c8b5c78"
    smart_video_split.dashscope_video_detect(test_url)
    sys.exit()

