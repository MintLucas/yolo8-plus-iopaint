#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2025/9/4 16:25
# @Author  : zhipeng16
# @concat: mintlzp@mail.ustc.edu.cn
# @Site: 
# @File: smart_iopaint_utils.py
# @Software: PyCharm
# @Usage:


import uuid
import cv2
import os
import subprocess
import numpy as np

import configs
from iopaint_api_utils import InpaintAPI


class SmartBaseIOPaint:
    """
    iopaint的工具类 - 基类
    """

    def __init__(self, device="cpu"):
        self.device = device

    def create_mask(self, image, bboxes, padding=1):
        """
        根据边界框创建掩码图像
        :param image: 原始图像 (numpy array)
        :param bboxes: 边界框列表
        :param padding: 掩码的额外填充
        :return: 掩码图像 (numpy array)
        """
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        for bbox in bboxes:
            if len(bbox) >= 4:
                x1, y1, x2, y2 = map(int, bbox[:4])
                # 这里可以根据需要调整padding逻辑
                # x1 = np.clip(x1 - padding - 130, 0, width) # 旧的padding逻辑
                # y1 = np.clip(y1 - padding, 0, height)
                # x2 = np.clip(x2 + padding, 0, width)
                # y2 = np.clip(y2 + padding + 60, 0, height)
                x1 = np.clip(x1 - padding - 130, 0, width)
                y1 = np.clip(y1 - padding, 0, height)
                x2 = np.clip(x2 + padding, 0, width)
                y2 = np.clip(y2 + padding + 60, 0, height)
                mask[y1:y2, x1:x2] = 255
        return mask

    def erase_watermark(self, image_path, mask, output_dir):
        """
        (保留方法) 从文件路径擦除水印
        """
        raise NotImplementedError


class SmartIOPaintCmdUtil(SmartBaseIOPaint):
    """
    (不推荐用于视频) 命令行方式运行iopaint的工具类
    注意：此方法对于视频逐帧处理效率极低，因为它涉及频繁的磁盘IO。
    """

    def erase_watermark(self, image_path, mask, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        image_name = os.path.basename(image_path)
        temp_mask_path = f"{configs.cache_dir}/{uuid.uuid4()}{image_name}"
        cv2.imwrite(temp_mask_path, mask)

        command = [
            "python", "-m", "iopaint", "run",
            "--model=lama",
            f"--device={self.device}",
            f"--image={image_path}",
            f"--mask={temp_mask_path}",
            f"--output={output_dir}"
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)  # 增加capture_output来隐藏iopaint的输出
            output_path = f"{output_dir}/{image_name}"
            print(f"水印已移除： {image_path} => {output_path}")
        finally:
            os.remove(temp_mask_path) if os.path.exists(temp_mask_path) else None


class SmartIOPaintApiUtil(SmartBaseIOPaint):
    """
    调用api方式运行iopaint的工具类，为视频流处理进行了优化。
    """

    def __init__(self, device="cpu"):
        super().__init__(device)
        self.inpaint_api = InpaintAPI()

    def erase_watermark(self, image_path, mask, output_dir):
        """
        (用于图片文件) 从文件路径擦除水印
        """
        os.makedirs(output_dir, exist_ok=True)
        image_name = os.path.basename(image_path)
        temp_mask_path = f"{configs.cache_dir}/{uuid.uuid4()}{image_name}"
        cv2.imwrite(temp_mask_path, mask)

        output_path = os.path.join(output_dir, image_name)
        try:
            self.inpaint_api.send_inpaint_request(image_path, temp_mask_path, output_path)
            print(f"水印已移除： {image_path} => {output_path}")
        finally:
            os.remove(temp_mask_path) if os.path.exists(temp_mask_path) else None
        return cv2.imread(output_path)

    def erase_watermark_from_image_data(self, image_data: np.ndarray, mask_data: np.ndarray):
        """
        (✨新增方法，用于视频帧)直接从内存中的图像和掩码数据中擦除水印。
        这避免了为每一帧读写磁盘，大大提高了视频处理速度。

        :param image_data: 图像的numpy数组
        :param mask_data: 掩码的numpy数组
        :return: 处理后的图像的numpy数组，如果失败则返回None
        """
        try:
            # 假设inpaint_api.send_inpaint_request_from_data方法已实现，
            # 它可以接收numpy数组，在内存中编码并发送API请求，然后解码返回的图像数据。
            # 如果iopaint_api_utils不支持，需要在那边添加此功能。
            # 这里我们模拟其行为：
            return self.inpaint_api.send_inpaint_request_from_data(image_data, mask_data)
        except Exception as e:
            print(f"Error during in-memory inpainting: {e}")
            return None