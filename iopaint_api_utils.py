#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author：samge
# date：2024-06-03 14:50
# describe：iopaint的工具类 - 通过api调用封装为类

import os
import requests
import base64
import cv2
import numpy as np

import configs


# IOPaint的服务地址，除了在本项目中执行 python iopaint_server.py 启动iopaint服务外，也可以选择对接单独部署的iopaint服务
IOPAINT_SERVER_HOST = configs.IOPAINT_SERVER_HOST


class InpaintAPI:

    def __init__(self):
        self.api_inpaint = f"{IOPAINT_SERVER_HOST}/api/v1/inpaint"
        self.headers = {
            "Content-Type": "application/json"
        }
        self.timeout = 30

    def convert_image_to_base64(self, image_path: str) -> str:
        """将图片文件转换为base64字符串"""
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string

    def convert_image_data_to_base64(self, image_data: np.ndarray, ext: str = '.png') -> str:
        """
        ✨ (新增) 将OpenCV图像数据（numpy数组）在内存中编码为base64字符串
        :param image_data: numpy数组格式的图像
        :param ext: 编码格式，默认为'.png'
        :return: base64编码的字符串
        """
        _, buffer = cv2.imencode(ext, image_data)
        return base64.b64encode(buffer).decode('utf-8')

    def send_inpaint_request(self, image_path: str, mask_path: str, output_path: str):
        """发送POST请求到inpaint API，并保存返回的图片"""
        # 保证输出路径存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 将图片和标记转换为base64字符串
        image_base64 = self.convert_image_to_base64(image_path)
        mark_base64 = self.convert_image_to_base64(mask_path)

        # 构建请求的JSON body
        json_body = {
            "image": image_base64,
            "mask": mark_base64
        }

        # 发送POST请求
        try:
            response = requests.post(self.api_inpaint, json=json_body, headers=self.headers, timeout=self.timeout)
        except requests.ConnectionError:
            msg = "\n"
            msg += "=" * 100
            msg += f"\nFailed to connect to the server. please check if the IOPaint service has started properly：{IOPAINT_SERVER_HOST}.\n"
            if '127.0.0.1' in IOPAINT_SERVER_HOST or 'localhost' in IOPAINT_SERVER_HOST:
                msg += "did you forget to execute 'python iopaint_server.py' to start the iopaint service?\n"
            msg += "=" * 100
            raise ValueError(msg)
        except Exception as e:
            raise e
        
        # 检查响应状态码
        if response.status_code == 200:
            # 将返回的二进制图片数据保存到文件
            with open(output_path, "wb") as f:
                f.write(response.content)
        else:
            print(f"请求失败，状态码：{response.status_code}, 响应内容: {response.text}")

    def send_inpaint_request_from_data(self, image_data: np.ndarray, mask_data: np.ndarray):
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

        # 构建请求的JSON body
        json_body = {
            "image": image_base64,
            "mask": mask_base64
        }

        # 发送POST请求
        try:
            response = requests.post(self.api_inpaint, json=json_body, headers=self.headers, timeout=self.timeout)
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


if __name__ == "__main__":
    # 使用示例
    # 1. 从文件处理
    print("--- 示例1: 从文件处理 ---")
    image_path = f"{configs.images_dir}/test.png"
    # 假设你有一个mask文件
    mask_image_path = f"{configs.cache_dir}/test_mask.png" 
    output_path = f"{configs.cache_dir}/output_from_file.png"
    
    # 创建一个虚拟的mask文件用于测试
    if os.path.exists(image_path):
        img_for_mask = cv2.imread(image_path)
        h, w, _ = img_for_mask.shape
        mock_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mock_mask, (w//4, h//4), (w*3//4, h*3//4), 255, -1)
        cv2.imwrite(mask_image_path, mock_mask)

        inpaint_api = InpaintAPI()
        inpaint_api.send_inpaint_request(image_path, mask_image_path, output_path)
        print(f"文件处理完成，输出至: {output_path}")

    # 2. 从内存数据处理
    print("\n--- 示例2: 从内存数据处理 ---")
    if os.path.exists(image_path):
        # 读取图片和mask到内存
        image_data = cv2.imread(image_path)
        mask_data = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
        
        inpaint_api = InpaintAPI()
        processed_image_data = inpaint_api.send_inpaint_request_from_data(image_data, mask_data)

        if processed_image_data is not None:
            output_data_path = f"{configs.cache_dir}/output_from_data.png"
            cv2.imwrite(output_data_path, processed_image_data)
            print(f"内存数据处理完成，输出至: {output_data_path}")
        else:
            print("内存数据处理失败。")
