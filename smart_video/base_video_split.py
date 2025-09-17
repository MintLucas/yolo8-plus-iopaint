#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2025/9/8 11:30
# @Author  : zhipeng16
# @concat: mintlzp@mail.ustc.edu.cn
# @Site: 
# @File: split_video_util.py
# @Software: PyCharm
# @Usage:
import os
import requests
from urllib.parse import urlparse
import json
import tempfile
import shutil
from util.token_util_new import token_fresh
token = token_fresh()


def _get_video_duration(video_path):
    """
    使用 ffprobe 获取视频时长。
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration_data = json.loads(result.stdout)
        duration = float(duration_data['format']['duration'])
        return duration
    except (FileNotFoundError, subprocess.CalledProcessError, KeyError, json.JSONDecodeError) as e:
        token.log.info(f"无法获取视频时长: {e}")
        return None


def _fallback_split(video_path, split_num, video_duration):
    """
    兜底视频切分函数，在多模态模型调用失败时使用。
    采用经验性、非均匀的方式切分视频。
    """
    token.log.info("大模型请求失败，正在执行兜底切分策略...")
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

    token.log.info("兜底切分完成。")
    return {"splitVideoPartResults": results}

def _is_url(path):
    """
    检查路径是否为URL。
    """
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def _download_video(url, download_dir):
    """
    从URL下载视频到指定目录。
    """
    try:
        local_path = os.path.join(download_dir, os.path.basename(urlparse(url).path))
        token.log.info(f"开始下载视频: {url} -> {local_path}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        token.log.info("视频下载完成。")
        return local_path
    except requests.exceptions.RequestException as e:
        token.log.info(f"下载视频失败: {e}")
        return None


def _call_multimodal_model(video_path, prompt, split_num, part_time):
    """
    模拟调用多模态大模型，进行视频切分。
    注意：这是模拟函数，不进行实际模型调用。
    在真实项目中，这里会替换为对模型API的调用，并传入相应的参数。
    """
    token.log.info(f"正在模拟调用多模态大模型...")
    token.log.info(f"视频路径: {video_path}")
    token.log.info(f"处理模式: {prompt}")
    token.log.info(f"期望段数: {split_num}")
    token.log.info(f"每段期望时间: {part_time}")

    # 模拟模型返回的JSON数据
    mock_result = {
        "splitVideoPartResults": [
            {
                "beginTime": 0.1,
                "endTime": 7.31,
                "theme": "相信大家都想在这个夏天练出这种和这种薄肌身材，今天就详细分享我自己从这样到这种薄肌平常训练动作。",
                "type": "common",
                "by": "multimodal"
            },
            {
                "beginTime": 7.31,
                "endTime": 13.98,
                "theme": "首先我们追求的不是这种大块肌肉，所以我们就没必要特意去健身房练，在家随便准备一片空地就行。",
                "type": "common",
                "by": "multimodal"
            },
            {
                "beginTime": 13.98,
                "endTime": 52.58,
                "theme": "首先是我的胸肌训练方法...",
                "type": "common",
                "by": "multimodal"
            },
            {
                "beginTime": 52.58,
                "endTime": 56.34,
                "theme": "最后就是背部训练...",
                "type": "common",
                "by": "multimodal"
            },
            {
                "beginTime": 56.34,
                "endTime": 61.3,
                "theme": "由于背部肌群比较大...",
                "type": "common",
                "by": "multimodal"
            },
            {
                "beginTime": 61.3,
                "endTime": 67.27,
                "theme": "只要你每天坚持...",
                "type": "common",
                "by": "multimodal"
            }
        ]
    }
    return mock_result


def base_video_split(video_input: str, split_num: int = 2, part_time: int = 45, mode: str = 'auto', video_duration:int = 167):
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
        # 1. 判断视频路径是本地还是URL
        # if _is_url(video_input):
        #     temp_dir = tempfile.mkdtemp()
        #     local_video_path = _download_video(video_input, temp_dir)
        #     if not local_video_path:
        #         token.log.info("视频下载失败，无法继续处理。")
        #         return None

        # 2. 读取prompt文件
        prompt_file_path = os.path.join(os.getcwd(), 'prompt_conf', 'split_video_prompt.md')
        if not os.path.exists(prompt_file_path):
            token.log.error(f"错误: 找不到prompt文件: {prompt_file_path}")
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
        result = token.call_model_zp(system_prompt=prompt_content, user_prompt=json.dumps(user_prompt),
                                                     user_prompt_media=use_medias,
                                                     model_type='volcengine:Doubao-Seed-1.6-flash')
        result = json.loads(result.strip("```json").strip("```"))

        if not result:
            token.log.error("模型调用失败，采用兜底策略")
            result = _fallback_split(local_video_path, split_num, video_duration)

        token.log.info("处理完成，返回结果。")
        return result

    except Exception as e:
        token.log.error(f"处理过程中发生错误: {e}")
        result = _fallback_split(local_video_path, split_num, video_duration)
        return result
    # finally:
        # 4. 如果是下载的临时文件，则清理
        # if temp_dir and os.path.exists(temp_dir):
        #     print(f"清理临时文件目录: {temp_dir}")
        #     shutil.rmtree(temp_dir)


# 示例调用
if __name__ == '__main__':
    # 确保 prompt_conf 目录和文件存在
    os.makedirs('./prompt_conf', exist_ok=True)
    # 模拟本地视频路径
    # local_video = "local_video_path/sample.mp4"
    # print("--- 场景1: 输入本地视频路径 ---")
    # local_result = smart_video_split(video_input=local_video, split_num=5, part_time=None)
    # print("本地处理结果:", json.dumps(local_result, indent=4, ensure_ascii=False))
    # print("\n" + "=" * 50 + "\n")

    # 模拟URL视频路径
    url_video = "https://aiclip.weibo.com/redirect?key=cph%2Fyt_dlp%2F7%2F86347%2F2025-09-08%2Fv_bb9e583ff072f40a2378e28d05c87eed.mp4"
    url_video = "https://aiclip.weibo.com/redirect?key=cph%2Fyt_dlp%2F1%2F30041%2F2025-09-08%2Fv_a1c112cbd215bc5cd233303d47420301.mp4"
    url_video = "https://aiclip.weibo.com/redirect?key=cph%2Fyt_dlp%2F8%2F1178%2F2025-09-08%2Fv_fc93fab5019f34d01cf22bfa253ae0e7.mp4"
    url_video = "https://wb-channel-aiclip-media.oss-cn-beijing.aliyuncs.com/cph/yt_dlp/3/97073/2025-09-05/v_d2276be08d62e0fea72d5fe4384f2dc4.mp4?x-oss-date=20250905T025501Z&x-oss-expires=604800&x-oss-signature-version=OSS4-HMAC-SHA256&x-oss-credential=LTAI5tHj9VxWxHdfk1rWYrdj%2F20250905%2Fcn-beijing%2Foss%2Faliyun_v4_request&x-oss-signature=d5b3955f92dc1296d8094146fbdfd110e47b367dba41ba8ff683f4491b41334a"
    print("--- 场景2: 输入URL视频路径 ---")
    url_result = smart_video_split(video_input=url_video, split_num=2, part_time=45)
    print("URL处理结果:", json.dumps(url_result, indent=4, ensure_ascii=False))