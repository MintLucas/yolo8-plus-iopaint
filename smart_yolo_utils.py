#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2025/9/4 16:28
# @Author  : zhipeng16
# @concat: mintlzp@mail.ustc.edu.cn
# @Site:
# @File: smart_yolo_utils.py
# @Software: PyCharm
# describe：(Smart Version) 使用yolo提取图片中目标边框的工具

from ultralytics import YOLO

import configs


class SmartYOLOUtils:
    """
    智能YOLO工具类，用于检测图像中的目标。
    与原版YOLOUtils功能一致，为保持项目结构清晰而创建。
    """
    def __init__(self, model_path):
        """
        初始化模型
        :param model_path: YOLO模型的路径
        """
        self.model = YOLO(model_path)

    def get_bboxes(self, image, conf=0.1):
        """
        从给定的图像中获取目标的边界框
        :param image: 输入图像 (numpy array)
        :param conf: 置信度阈值
        :return: 检测到的边界框列表
        """
        results = self.model(image, conf=conf, verbose=False) # 设置verbose=False以减少控制台输出
        # 注意：这里需要根据你的模型输出进行适配，有些可能是results[0].boxes.xyxy
        if len(results) > 0 and len(results[0].boxes) > 0:
            bboxes = results[0].boxes.cpu().data.numpy()
            return bboxes
        return []


if __name__ == "__main__":
    # 将.pt模型转为.onnx模型，需要安装依赖：onnx==1.16.1、onnx-simplifier==0.4.36、onnxsim==0.4.36、onnxslim==0.1.28、onnxruntime_gpu==1.18.0
    model = YOLO(f"{configs.models_dir}/last.pt")
    model.export(format='onnx', imgsz=640, dynamic=True, simplify=True) # 建议使用一个固定的imgsz