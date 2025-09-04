#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author：samge
# date：2024-06-03 14:59
# describe：使用yolo提取图片中目标边框的工具

from ultralytics import YOLO

import configs


class YOLOUtils:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def get_bboxes(self, image, conf=0.1):
        results = self.model(image, conf=conf)
        # bboxes = results[0].boxes[0].cpu().data.numpy()
                # 注意：这里需要根据你的模型输出进行适配，有些可能是results[0].boxes.xyxy
        if len(results) > 0 and len(results[0].boxes) > 0:
            bboxes = results[0].boxes.cpu().data.numpy()
            return bboxes
        return []
        return bboxes


if __name__ == "__main__":
    # 将.pt模型转为.onnx模型，需要安装依赖：onnx==1.16.1、onnx-simplifier==0.4.36、onnxsim==0.4.36、onnxslim==0.1.28、onnxruntime_gpu==1.18.0
    model = YOLO(f"{configs.models_dir}/last.pt")
    model.export(format='onnx', imgsz=288, dynamic=True, simplify=True)
