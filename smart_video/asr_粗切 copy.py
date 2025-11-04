#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2025 sina.com.cn, Inc. All Rights Reserved
#
########################################################################
"""
File: asr_粗切.py
Author: yining8(yining8@staff.sina.com.cn)
Date: 2025/09/11 18:25:05
"""
import os, re
from funasr import AutoModel
import subprocess
import requests
def asr_ms_to_hmsms(ms):
    """将ms转化成hh:mm:ss.ms"""
    return f"{ms//3600000:02d}:{ms%3600000//60000:02d}:{ms%60000//1000:02d}.{ms%1000:03d}"

def download_video(url, save_path='./data/'):
    # 确保保存路径存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 获取文件名
    filename = url.split('/')[-1]
    filename_re = re.search('(.*)\.mp4\?', filename) # Y3tEFiiklx08puyOXady01041200BeL10E010.mp4?label=mp4_hd&template=852x480.25.0&Expires=1755504324&ssig=YwuvXtCFRC&KID=unistore,video
    if filename_re:
        filename = filename_re[1]+'.mp4'

    print(f'filename: {filename}')
    filepath = os.path.join(save_path, filename)
    print(f'filepath: {filepath}')
    if os.path.exists(filepath):
        print(f"func download_video: 视频已存在 {filepath}")
        return filepath
    # 发起请求
    response = requests.get(url, stream=True)
    # 检查请求是否成功
    if response.status_code == 200:
        # 以二进制模式写入文件
        with open(filepath, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"视频已下载到 {filepath}")
        return filepath
    else:
        print(f"下载失败，状态码：{response.status_code}")
        return ""


def asr_collect_split_para(asr_text, asr_raw_text, asr_punc_array, asr_timestamp):
    """根据asr的分句，给出分段文本+起始|终止时间"""
    para_list = []
    para = ""
    ti = 0
    dur_start, dur_end = -1, -1
    for i in range(len(asr_raw_text)):
        if len(para) == 0:
            dur_start = asr_timestamp[i][0]
        ti += 1
        para += asr_raw_text[i]
        punc_signal = int(asr_punc_array[i])
        if punc_signal != 1: # 需要停顿
            para += asr_text[ti] # 加对应标点
            ti += 1
            if (punc_signal==3 and len(para)>20) or punc_signal==4: # 切分段落
                dur_end = asr_timestamp[i][1]
                tmp_dic = {
                        "text": para,
                        "start_ms": dur_start,
                        "end_ms": dur_end,
                        "start_str": asr_ms_to_hmsms(dur_start),
                        "end_str": asr_ms_to_hmsms(dur_end),
                        "duration": dur_end - dur_start
                        }
                para_list.append(tmp_dic)
                para = ""
    return para_list
    
# 将模型下载路径设置为你想要的路径
# os.environ['MODELSCOPE_CACHE'] = '/data2/zhipeng16/git/yolo8-plus-iopaint/models/asr_models/'
from funasr import AutoModel
# 初始化模型
cur_dir='/workspace/work/zhipeng16/yolo8-plus-iopaint/models/'
# 模型 ID，用于自动下载，它们是 FunASR 官方定义的
asr_model_id = "speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
vad_model_id = "speech_fsmn_vad_zh-cn-16k-common-pytorch"
punc_model_id = "ct-punc"

asr_model = AutoModel(
    model=cur_dir+asr_model_id,
    vad_model=cur_dir+vad_model_id, # VAD用来切静音
    punc_model=cur_dir+punc_model_id, # 加标点
    device="cuda:2",
    disable_update=True,
    en_post_proc = True
    
)

# asr_model = AutoModel(
#     model="speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn",   # 逐字转录
#     vad_model="speech_fsmn_vad_zh-cn-16k-common-pytorch", # VAD用来切静音
#     punc_model="ct-punc", # 加标点
#     device="cuda"
#     )

video_url = "https://aiclipcdn.weibo.com/ai_media/temp/c1cb62a72c364ecfb7f695fde0d7fd4d_iUBYcPYQlx08nuw3XGWQ01041200nGSB0E010.mp4"
video_url = "https://weibo-ml-wces-push.oss-cn-shanghai.aliyuncs.com/wces/video_13m.mp4?x-oss-signature-version=OSS4-HMAC-SHA256&x-oss-date=20250922T065139Z&x-oss-expires=431999&x-oss-credential=LTAI5tNLPgvob9Kim83vQ3Ch%2F20250922%2Fcn-shanghai%2Foss%2Faliyun_v4_request&x-oss-signature=a0592ccf79422d8350bbc1c876cd3c07b3450c8e049a8c3917dddbc401420ce8"
print(f'video_url: {video_url}')
video_path = download_video(video_url, save_path=cur_dir+"FunASR_part/data")
print(f'video_path: {video_path}')
video_name = video_url.split('/')[-1].split('.')[0]
wav_path = os.path.splitext(video_path)[0]+".wav"

# 抽音频（16kHz 单声道）
if not os.path.exists(wav_path):
    print('提取音频文件')
    subprocess.run(["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000", wav_path])

# asr粗切
asr_result = asr_model.generate(wav_path, return_raw_text=True)[0]
asr_text = asr_result["text"] # 分句
asr_raw_text = asr_result["raw_text"].split(' ') # 无标点
asr_raw_text = [x for x in asr_raw_text if len(x)!=0]
asr_timestamp = asr_result["timestamp"]
asr_punc_array = asr_result["punc_array"] # 1不需要切分，2逗号，3切分，4一定要切分
para_list = asr_collect_split_para(asr_text, asr_raw_text, asr_punc_array, asr_timestamp) # 切分整合段落
for dic in para_list:
    print(dic)
