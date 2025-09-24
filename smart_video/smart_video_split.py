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
import traceback,time,random,redis
from smart_video.base_video_class import Base_video_class
from smart_video.test_video_segments import filter_and_combine_video_segments

class Smart_video_split(Base_video_class):
    def __init__(self, log=get_logger("smart_video_split")):
        self.log = log
        super().__init__(self.log)
        self.redis_key = self.__class__.__name__
        self.oss_util = oss_util(self.log)
        self.token_fresh = token_fresh()
        from threading import Lock
        self.lock = Lock()

    def dashscope_video_split(self, input_path = "", split_num=5, part_time = 45, video_duration = 167, material_id = "zhipeng16_test"):
        """
        使用dashscope进行视频切割
        """
        try:
            material_id = str(material_id)
            self.log.info(f"start_process_task_id_{material_id}")
            self.redis_client.hset(self.redis_key, material_id, 'waiting')
            with self.lock:
                self.redis_client.hset(self.redis_key, material_id, 'downloading')
                self.log.info(f"downloading_process_task_id_{material_id}")
                local_path,file_name = self.oss_util.check_video_path(input_path)
                oss_bucket_object_key = "wces/"+file_name
                self.redis_client.hset(self.redis_key, material_id, 'processing_state1')
                self.log.info(f"upload_file_process_task_id_{material_id} oss_bucket_object_key: {oss_bucket_object_key}")
                up_status = self.oss_util.upload_file(local_file_path=local_path, object_key=oss_bucket_object_key)
                sh_url_path = self.oss_util.get_url(oss_bucket_object_key)
                self.log.info(f"get_url_process_task_id_{material_id} url:{sh_url_path}")
                self.redis_client.hset(self.redis_key, material_id, 'processing_state2')
                start_time = time.time()
                dashcope_res = self.token_fresh.call_model_zp_video_split(videoUrl = sh_url_path, duration=video_duration)
                self.log.info(f"video_split_process_task_id_{material_id} url:{sh_url_path}")
                task_id = dashcope_res.get("response_data", {}).get("RequestId", "")
                self.redis_client.hset(self.redis_key, material_id, 'processing_state3')
                dashcope_status = self.token_fresh.query_video_split_task_status(task_id)
                self.log.info(f"query_video_split_process_task_id_{material_id} dashcope_status:{dashcope_status}")
                dashcope_video_split_res = json.loads(dashcope_status.get("response_data", {}).get("Data", {}).get("Result", "{}"))
                dashcope_video_split_list = dashcope_video_split_res.get("splitVideoPartResults", [])
                end_time = time.time()
                elapsed_time_seconds = end_time -start_time
                elapsed_minutes = int(elapsed_time_seconds // 60)
                elapsed_seconds = int(elapsed_time_seconds % 60)
                # 打印运行时间
                self.log.info(f"dash_split运行时间: {elapsed_minutes} 分钟 {elapsed_seconds} 秒")
                dashcope_video_split_list = filter_and_combine_video_segments(dashcope_video_split_list, split_num, part_time)
                self.redis_client.hset(self.redis_key, material_id, f'finish_all:{json.dumps(dashcope_video_split_list, ensure_ascii=False)}')
                self.oss_util.delete_file(local_path)
                return dashcope_video_split_list
        except Exception as e:
            error_info = traceback.format_exc()
            self.redis_client.hset(self.redis_key, material_id, f'error:{e}')
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
            self.redis_client.hset(self.redis_key, material_id, f'processing_state1')
            result = self.token_fresh.call_model_zp(system_prompt=prompt_content, user_prompt=json.dumps(user_prompt),
                                                        user_prompt_media=use_medias,
                                                        model_type='volcengine:Doubao-Seed-1.6-flash')
            result = json.loads(result.strip("```json").strip("```"))

            if not result:
                self.log.error("模型调用失败，采用兜底策略")
                self.redis_client.hset(self.redis_key, material_id, f'processing_state2')
                result = self._fallback_split(local_video_path, split_num, video_duration, part_time)

            self.log.info("处理完成，返回结果。")
            self.redis_client.hset(self.redis_key, material_id, f'finish_all:{json.dumps(result["splitVideoPartResults"], ensure_ascii=False)}')
            return result

        except Exception as e:
            self.log.error(f"处理过程中发生错误,走finnaly兜底逻辑: {e}")
            result = self._fallback_split(local_video_path, split_num, video_duration, part_time)
            self.redis_client.hset(self.redis_key, material_id, f'finish_all:{json.dumps(result["splitVideoPartResults"], ensure_ascii=False)}')
            return result


    def _fallback_split(self, video_path, split_num, video_duration, part_time):
        """
        优化的兜底视频切分函数，在多模态模型调用失败时使用。
        参数：
        - video_path: 视频路径 (保留参数)
        - split_num: 期望切分的片段数量
        - video_duration: 视频总时长
        - part_time: 期望的单个片段时长，单位秒
        """
        self.log.info(f"大模型请求失败，正在执行兜底切分，期望片段时长：{part_time}s...")
        
        duration = video_duration
        # 确保 part_time 参数有效，设置一个最小有效值
        min_segment_duration = max(5, part_time / 2)

        if not duration or split_num <= 0 or part_time <= 0:
            return {"splitVideoPartResults": []}

        results = []
        processed_time = 0

        # 循环生成指定数量的片段
        while len(results) < split_num and processed_time < duration:
            
            # 片段时长在 [part_time/2, part_time] 范围内随机浮动
            # 这确保了时长可控，但又不会过于呆板
            segment_duration = random.uniform(min_segment_duration, part_time)

            # 模拟高光点的位置，在已处理时间之后随机选择一个位置作为起点
            # 这里的随机步长可以与 part_time 相关，例如 part_time/4 到 part_time
            # 确保不会过于密集
            start_time_candidate = processed_time + random.uniform(part_time / 4, part_time)
            
            # 确保片段不会超出视频总时长
            if start_time_candidate + segment_duration > duration:
                # 如果剩余时长不够，尝试将剩余部分作为一个片段
                remaining_duration = duration - start_time_candidate
                if remaining_duration > 5: # 确保片段有意义
                    end_time = duration
                    theme = f"自动切分 {len(results) + 1}：视频末尾"
                    results.append({
                        "beginTime": round(start_time_candidate, 2),
                        "endTime": round(end_time, 2),
                        "theme": theme,
                        "type": "highlight",
                        "by": "fallback"
                    })
                break # 循环结束

            start_time = start_time_candidate
            end_time = start_time + segment_duration
            
            theme = f"自动切分 {len(results) + 1}：视频{int(start_time)}s附近"
            results.append({
                "beginTime": round(start_time, 2),
                "endTime": round(end_time, 2),
                "theme": theme,
                "type": "highlight",
                "by": "fallback"
            })
            
            # 更新已处理时间，避免下一个片段与之重叠
            # 在片段之间留出随机的“过渡/非高光”时间
            processed_time = end_time + random.uniform(5, 15)

        self.log.info(f"兜底切分完成，共生成 {len(results)} 个片段。")
        return {"splitVideoPartResults": results}



    def gen_video_split(self, source_input):
        video_input = source_input.get("uploaded_video_url", "")
        split_num = source_input.get("split_num", 2)
        part_time = source_input.get("part_time", 45)
        mode = source_input.get("mode", 0)
        video_duration = source_input.get("video_duration", 167)
        task_id = source_input.get("material_id", "zhipeng_test")
        if mode == 1:
            res = self.dashscope_video_split(video_input, split_num, part_time, video_duration, material_id=task_id)
        else:
            res = self.base_video_split(video_input, split_num, part_time, mode, video_duration, task_id)
        return res



if __name__ == '__main__':
    smart_video_split = Smart_video_split()
    test_url = "https://weibo-ml-wces-push.oss-cn-shanghai.aliyuncs.com/wces/video.mp4?Expires=1758100623&OSSAccessKeyId=TMP.3KktyZb2nsSSVfK2cy8fu5yt8x26F6DJqwJwg9j1Lrh6J8iV8aEsZQo8rfshq2KXnZCxmJJhgLEsSeFVqsDKoTa6tpT1AF&Signature=d3mcacidzGYeibYxoWD8rpfbs4c%3D"
    smart_video_split.dashscope_video_split(test_url, video_duration=30)
    sys.exit()

