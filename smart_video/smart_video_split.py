#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/09/15 14:53
# @Author  : zhipeng16
# @Email   : zzzp50@ustc.edu
# @File    : smart_video_split.py
# @Usage   : Describe the file's purpose
import sys,os
sys.path.append(os.getcwd())
import json
from util.mylogging import get_logger
from smart_video.api_test_oss2 import oss_util
from util.token_util_new import token_fresh
import traceback
class Smart_video_split:
    def __init__(self, log=get_logger("smart_video_split")):
        self.log = log
        self.oss_util = oss_util(self.log)
        self.token_fresh = token_fresh()
        self.task_status = {}
        from threading import Lock
        self.lock = Lock()

    def dashscope_video_split(self, input_path = "", video_duration = 167, material_id = "zhipeng16_test"):
        """
        使用dashscope进行视频切割
        """
        try:
            self.log.info(f"start_process_task_id_{material_id}")
            self.task_status[material_id] = 'waiting'
            with self.lock:
                self.task_status[material_id] = 'downloading'
                local_path,file_name = self.oss_util.check_video_path(input_path)
                oss_bucket_object_key = "wces/"+file_name
                self.task_status[material_id] = 'processing_state1'
                up_status = self.oss_util.upload_file(local_file_path=local_path, object_key=oss_bucket_object_key)
                sh_url_path = self.oss_util.get_url(oss_bucket_object_key)
                self.task_status[material_id] = 'processing_state2'
                dashcope_res = self.token_fresh.call_model_zp_video_split(videoUrl = sh_url_path, duration=video_duration)
                task_id = dashcope_res.get("response_data", {}).get("RequestId", "")
                self.task_status[material_id] = 'processing_state3'
                dashcope_status = self.token_fresh.query_video_split_task_status(task_id)
                dashcope_video_split_res = json.loads(dashcope_status.get("response_data", {}).get("Data", {}).get("Result", "{}"))
                dashcope_video_split_list = dashcope_video_split_res.get("splitVideoPartResults", [])
                self.task_status[material_id] = f'finish_all:{json.dumps(dashcope_video_split_list, ensure_ascii=False)}'
                self.oss_util.delete_file(local_path)
                return dashcope_video_split_list
        except Exception as e:
            error_info = traceback.format_exc()
            self.task_status[material_id] = f'error:{e}'
            self.log.error(f"视频切割异常: {e}\n{error_info}")
            return 0

    def base_video_split(self, video_input: str, split_num: int = 2, part_time: int = 45, mode: str = 'auto', video_duration:int = 167, material_id = "zhipeng_test"):
        """
        智能视频切分函数。

        Args:
            video_input (str): 视频路径，可以是本地路径或URL。
            split_num (int, optional): 期望的拆分段数。
            part_time (int, optional): 每段视频的期望时长（秒）。
            mode (str, optional): 处理模式。

        Returns:
            dict: 包含视频切分结果的字典，如果处理失败则返回None。
        """
        local_video_path = video_input
        temp_dir = None

        try:
            prompt_file_path = os.path.join(os.getcwd(), 'config', 'split_video_prompt.md')
            if not os.path.exists(prompt_file_path):
                self.log.error(f"错误: 找不到prompt文件: {prompt_file_path}")
                return None

            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                prompt_content = f.read().strip()

            # 3. 将视频和prompt送入多模态大模型进行处理
            # 在这里，我们调用一个模拟函数来代替实际的模型调用
            # 实际开发中，这里会是调用一个API接口，如 model.chat(...)


            use_medias =[{'type': 'video_url', 'video_url': {'url': video_input}}]
            user_prompt = {
                "split_num": split_num,
                "part_time": part_time,
            }
            self.task_status[material_id] = f'processing_state1'
            result = self.token_fresh.call_model_zp(system_prompt=prompt_content, user_prompt=json.dumps(user_prompt),
                                                        user_prompt_media=use_medias,
                                                        model_type='volcengine:Doubao-Seed-1.6-flash')
            result = json.loads(result.strip("```json").strip("```"))

            if not result:
                self.log.error("模型调用失败，采用兜底策略")
                self.task_status[material_id] = f'processing_state2'
                result = self._fallback_split(local_video_path, split_num, video_duration)

            self.log.info("处理完成，返回结果。")
            self.task_status[material_id] = f'finish_all:{json.dumps(result["splitVideoPartResults"], ensure_ascii=False)}'
            return result

        except Exception as e:
            self.log.error(f"处理过程中发生错误: {e}")
            result = self._fallback_split(local_video_path, split_num, video_duration)
            return result


    def _fallback_split(self, video_path, split_num, video_duration):
        """
        兜底视频切分函数，在多模态模型调用失败时使用。
        采用经验性、非均匀的方式切分视频。
        """
        self.log.info("大模型请求失败，正在执行兜底切分策略...")
        # duration = _get_video_duration(video_path)
        duration = video_duration
        if not duration:
            return {"splitVideoPartResults": []}

        results = []

        # 设定基于经验的切分比例，可以自行调整
        # 示例：高光可能出现在 10% 处，40% 处，75% 处
        split_points = [0.1, 0.4, 0.75]

        for i, point in enumerate(split_points):
            # 确保切分数量不超过 split_num
            if len(results) >= split_num:
                break

            # 假设每个高光片段持续 15-25 秒
            start_time = duration * point
            end_time = min(start_time + 20, duration)  # 假设每个高光时长为20秒，不超过总时长

            # 确保片段有意义，例如时长大于5秒
            if end_time - start_time > 5:
                theme = f"自动切分 {i + 1}：视频{int(start_time)}s附近"
                results.append({
                    "beginTime": round(start_time, 2),
                    "endTime": round(end_time, 2),
                    "theme": theme,
                    "type": "highlight",
                    "by": "fallback"
                })

        self.log.info("兜底切分完成。")
        return {"splitVideoPartResults": results}



    def gen_video_split(self, source_input):
        video_input = source_input.get("uploaded_video_url", "")
        split_num = source_input.get("split_num", 2)
        part_time = source_input.get("part_time", 45)
        mode = source_input.get("mode", 0)
        video_duration = source_input.get("video_duration", 167)
        task_id = source_input.get("material_id", "zhipeng_test")
        if mode == 1:
            res = self.dashscope_video_split(video_input, video_duration, material_id=task_id)
        else:
            res = self.base_video_split(video_input, split_num, part_time, mode, video_duration, task_id)
        return res



if __name__ == '__main__':
    smart_video_split = Smart_video_split()
    test_url = "https://weibo-ml-wces-push.oss-cn-shanghai.aliyuncs.com/wces/video.mp4?Expires=1758100623&OSSAccessKeyId=TMP.3KktyZb2nsSSVfK2cy8fu5yt8x26F6DJqwJwg9j1Lrh6J8iV8aEsZQo8rfshq2KXnZCxmJJhgLEsSeFVqsDKoTa6tpT1AF&Signature=d3mcacidzGYeibYxoWD8rpfbs4c%3D"
    smart_video_split.dashscope_video_split(test_url, video_duration=30)
    sys.exit()

