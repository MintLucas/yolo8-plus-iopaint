#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2025/6/18 15:02
# @Author  : zhipeng16
# @concat: mintlzp@mail.ustc.edu.cn
# @Site:
# @File: token_util_new.py
# @Software: PyCharm
# @Usage:

import json
import sys
import time
import traceback
import sys, os

sys.path.append(os.getcwd())
from util import mylogging
import requests
from urllib.parse import urlparse
import os

debug_module = "onlin"
question_source = "116913455"
push_source = "356732087"
role_source = "704786492"
huoshan_models_list = [
    "Doubao-Seed-1.6-thinking-250615",
    "Doubao-Seed-1.6-thinking",
    "Doubao-Seed-1.6-250615",
    "Doubao-Seed-1.6-flash-250615",
    "Doubao-Seed-1.6-flash",
    "Doubao-Seed-1.6",
    "DeepSeek-R1-250528",
    "Doubao-1.5-vision-pro-250328",
    "Doubao-1.5-vision-lite-250315",
    "Doubao-1.5-thinking-pro-250415",
    "Doubao-1.5-thinking-pro",
    "Doubao-1.5-thinking-pro-vision",
    "Doubao-1.5-vision-pro",
    "Doubao-1.5-vision-lite",
    "Doubao-1.5-pro-32k",
    "Doubao-1.5-pro-256k",
    "Doubao-1.5-lite-32k",
    "Doubao-1.5-vision-pro-32k",
    "deepseek-r1",
    "deepseek-v3",
    "Doubao-1.5-thinking-vision-pro-250428",
    "DeepSeek-V3-250324"
]
models_dict = {"video-huoshan": "volcengine:Doubao-Seedance-1.0-pro-250528",
               "video-huoshan-fast": "volcengine:Doubao-Seedance-1.0-pro-fast-251015",
               "video-split": "dashscope:SplitVideoParts",
               "image": "volcengine:Doubao-Seedream-4.0-250828",
               "text-huoshan": "volcengine:Doubao-Seed-1.6-flash",
               "text-ds": "volcengine:DeepSeek-V3-250324",
               "text-gemini": "google-cloud:gemini-2.0-flash",
               "text-qwen": "dashscope:qwen-vl-max"}
video_prompt_test = "生成一段 10 秒健身教学视频，分镜节奏如下：0-3 秒用文字标题 + 运动场景快速引入，呈现‘夏天练出薄肌身材’的主题；3-7 秒展示居家训练场景（如空地），搭配画外音说明‘无需去健身房，在家即可训练’；7-10 秒特写演示欢聚俯卧撑动作，突出胸部发力细节，字幕标注‘每天 50-100 个，练胸关键动作’。整体画面简洁，节奏紧凑，以实用训练指导为主，结尾可快速闪过‘坚持即见效’的提示文字 --ratio 16:9 --duration 10"
video_duration = "--duration 10"
test_video_url = "https://wb-channel-aiclip-media.oss-cn-beijing.aliyuncs.com/cph/yt_dlp/3/97073/2025-09-05/v_d2276be08d62e0fea72d5fe4384f2dc4.mp4?x-oss-date=20250905T025501Z&x-oss-expires=604800&x-oss-signature-version=OSS4-HMAC-SHA256&x-oss-credential=LTAI5tHj9VxWxHdfk1rWYrdj%2F20250905%2Fcn-beijing%2Foss%2Faliyun_v4_request&x-oss-signature=d5b3955f92dc1296d8094146fbdfd110e47b367dba41ba8ff683f4491b41334a"


class token_fresh:
    def __init__(self, log=mylogging.get_logger("token_util_new"), session=requests.session()):
        self.log = log
        self.session = session
        self.last_fresh_time = time.time()
        self.token_cache = {}
        self.gpt4_address = 'http://i.aigc.weibo.com/completion'

        # 'param':{'type':'dashscope','model_id':'xingchen-base','appkey': '704786492'}
        # 角色号1
        self.source = '704786492'
        self.source = "2552521548"
        # push
        self.source = "356732087"
        # 不知道哪个
        self.source = "1315463960"
        # 出题
        self.source = "116913455"

    def get_token(self, source="1315463960", uid=None):
        now = time.time()
        key = (source, uid)
        if not self.token_cache.get(key) or now - self.token_cache[key][0] >= 8 * 3600:
            headers = {}
            url = f'http://ip1.push.weibo.cn:9033/get_uid_token?source={source}&ips=1' if not uid else \
                f'http://ip1.push.weibo.cn:9033/get_uid_token?source={source}&uid={uid}&ips=1'
            try:
                r = self.session.get(url, timeout=3)
                headers = {'Authorization': r.json()['data']}
            except:
                import traceback
                err_msg = traceback.format_exc()
                self.log.error(f'call token err:{err_msg}')
            self.log.info(f'call token:source:{source} {headers}')
            if headers:
                # self.token = headers
                self.token_cache[key] = [now, headers]

        return self.token_cache[key][1]

    def call_model_local(self, parm, model_ext, is_stream=False, source="356732087"):
        '''本地调试使用'''
        if source == "356732087":
            ser_url = "http://ip1.push.weibo.cn:6990/gpt4"
        else:
            ser_url = 'http://ip1.push.weibo.cn:11024/gpt4'
        d = {'qu': 'oj', 'param': parm, 'direct': 1}
        parm['appkey'] = source
        if is_stream:
            d['stream'] = 'true'
        # print(parm)
        # d= {'qu':'oj','stream':'true','param':{'message': '猪八戒是个什么样的人？请你模仿他的口吻说几句话', 'appkey': '356732087', 'type': 'dashscope', 'model_id': 'deepseek-v3','smart_schedule':'1'},'direct':1}
        if 'dashscope' in parm.get("type", ""):
            d['api_ext'] = {
        "request_compatible_mode": "dashscope"
    }
        d['model_ext'] = model_ext

        cs = None
        js = {}

        try:
            if not is_stream:
                r = self.session.post(ser_url, json=d, timeout=300)
                cs = r.content.decode()
                js = r.json()
            else:
                r = self.session.post(ser_url, json=d, timeout=300, stream=True)
                cnts = []
                for li in r.iter_lines(chunk_size=10):
                    li = li.decode()
                    # sm_js = {}

                    from jsonpath import jsonpath
                    try:
                        sm_js = json.loads(li)
                        cnt = jsonpath(sm_js, '$.response_data.output.choices[0].message.content')[0]
                        cnts.append(cnt)
                    except:
                        pass
                self.log.info(f'stream ret:{cnts}')
                if cnts:
                    cnts = ''.join(cnts)
                    js = {'response_data': {'choices': [{'message': {'content': cnts}}]
                                            }
                          }
        except:
            er_msg = traceback.format_exc()
            self.log.error(f'model err:{cs} {er_msg}')
        return js

    def call_model(self, parm, model_ext, timeout=300, try_time=2, source="356732087"):
        # 线上使用
        import json
        token = self.get_token()
        post_d = {'url': self.gpt4_address,
                  'headers': token,
                  'params': parm,
                  'timeout': timeout
                  }
        parm['use_ext_first'] = '1'
        payload = json.dumps(model_ext, ensure_ascii=False).encode('utf-8')
        post_d['data'] = payload
        js = {}
        cs = None
        begin = time.time()
        for i in range(try_time):
            try:
                r = self.session.post(**post_d)
                cs = r.content.decode()
                js = r.json()
                self.log.info(f'call_model_zp:{js}')
                break
            except:
                import traceback
                er_msg = traceback.format_exc()
                self.log.error(f'err call model:{cs} {er_msg}')
        end = time.time()
        self.log.info(f'model call spend:{end - begin}')
        # print(cs)
        return js

    def decode_llm_it_output(self, ans, model_type = ''):
        import jsonpath
        # jsonpath(ans,'$.response_data.choices[0].message.content')[0]
        if 'qwen' in model_type:
            messages = ans.get("response_data", {}).get('output', {}).get("choices", [{"message": {}}])[0]
            message = messages.get("message", {"content": ""}).get("content", [{'text':''}])[0].get('text')
        else:
            messages = ans.get("response_data", {}).get("choices", [{"message": {}}])[0]
            message = messages.get("message", {"content": ""}).get("content", "")
        return message

    def call_model_google(self, user_prompt, system_prompt, model_type=models_dict["text-gemini"], source="356732087",
                      data={}):
        from jsonpath import jsonpath
        if not user_prompt and not system_prompt:
            return ""
        pic_ch = []
        if data.get('images'):
            pic_ch += [{
                "inlineData": {"mimeType": "image/*", "data": x}
            } for x in data['images'][:7]]
        if data.get('videos'):
            pic_ch += [{
                "inlineData": {"mimeType": "video/*", "data": x}
            } for x in data.get('videos', [])[:2]]
        self.log.info(f'img or video num:{len(pic_ch)}')
        pic_ch.append({'text': user_prompt})
        # 线上使用
        model_ext = {
            'model_ext':{
                'contents':[
                    {
                        'role':'user',
                        'parts':pic_ch
                    }
                ]
            }

        }

        utype, model_id = model_type.split(":")
        params = {
            'model_id': model_id,
            'appkey': source,
            'type': utype,
            'use_ext_first': '1',
            'message': user_prompt
        }

        llm_res_json = self.call_model(params, model_ext, source=source)
        llm_res = ""
        try:
            self.log.info(f"call_model_zp_video :{model_type}\n{llm_res_json}")
            try:
                llm_res = jsonpath(llm_res_json, '$.response_data.candidates[0].content.parts[0].text')[0]
            except:
                llm_res = ""
            # llm_res = self.decode_llm_it_output(llm_res_json)
            if not llm_res:
                llm_res_json = self.call_model_local(params, model_ext, source=source)
                self.log.info(f"llm_res_none_call_model_zp_video_local :{model_type}\n{llm_res_json}")
                llm_res = jsonpath(llm_res_json, '$.response_data.candidates[0].content.parts[0].text')[0]

        except:
            self.log.error(f"call_model_zpl_error: input:{llm_res_json}, " + traceback.format_exc())
            print(traceback.format_exc())
        return llm_res

    def call_model_zp(self, user_prompt, system_prompt, model_type=models_dict["text-ds"], source="356732087",
                      user_prompt_media=""):
        if not user_prompt and not system_prompt:
            return ""
        # 线上使用
        
        model_ext = {
            "model_ext": {
                # 'disable_analysis': True,
                "temperature": 0.8,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }
        }
        if 'qwen' in model_type:
            model_ext = {"model_ext":{
                    "model": 'qwen-vl-max',
                    "input": {
                        # 'disable_analysis': True,
                        "temperature": 0.8,
                        "messages": [
                            {"role": "user", "content": user_prompt}
                        ]
                    }
                }}
        if user_prompt_media:
            model_ext['model_ext']['messages'].append({"role": "user", "content": user_prompt_media})
        
        utype, model_id = model_type.split(":")
        params = {
            'model_id': model_id,
            'appkey': source,
            'type': utype,
            'use_ext_first': '1',
            'message': 'no_use',
            'smart_schedule':'1'
        }

        llm_res_json = self.call_model(params, model_ext, source=source)
        llm_res = ""
        try:
            self.log.info(f"call_model_zp_video :{model_type}\n{llm_res_json}")
            llm_res = self.decode_llm_it_output(llm_res_json, model_type)
            if not llm_res:
                llm_res_json = self.call_model_local(params, model_ext, source=source)
                self.log.info(f"llm_res_none_call_model_zp_video_local :{model_type}\n{llm_res_json}")
                llm_res = self.decode_llm_it_output(llm_res_json, model_type)
        except:
            self.log.error(f"call_model_zpl_error: input:{llm_res_json}, " + traceback.format_exc())
            print(traceback.format_exc())
        return llm_res

    def call_model_zp_img(self, user_prompt="一只可爱的猫猫拿起桌上的一只毛球", system_prompt="",
                          model_type="volcengine:Doubao-Seedance-1.0-pro-250528", source="356732087"
                          , imgs=[], response_format='b64_json'):
        if not user_prompt and not system_prompt:
            return ""
        # 线上使用
        model_ext = {
            "model_ext": {
                # 'disable_analysis': True,
                "temperature": 0.8,
                "req_key": "high_aes_general_v30l_zt2i",
                'prompt': user_prompt,
                'width': 1131,
                'height': 852,
                "response_format": response_format,
                "watermark": True,
                "sequential_image_generation": "disabled",
                "size": "1152x864"
            }
        }
        # if imgs:
        #     model_ext["model_ext"]["content"].extend(imgs)
        utype, model_id = model_type.split(":")
        params = {
            'model_id': model_id,
            'appkey': source,
            'type': utype,
            'use_ext_first': '1',
            'message': 'no_use'
        }

        llm_res_json = self.call_model(params, model_ext,
                                       source=source) if debug_module == "online" else self.call_model_local(params,
                                                                                                             model_ext,
                                                                                                             source=source)
        try:
            res_data = llm_res_json['response_data']['data']
        except Exception as e:
            self.log.error(f"error_in_zp_img: {llm_res_json}")
            self.log.error(traceback.format_exc())
            return ""
        if response_format == "url":
            img_data = res_data.get("url", "")
        else:
            img_data = res_data[0].get("b64_json", "") if "Seedream" in model_type else \
            res_data.get("binary_data_base64", [""])[0]
        return img_data

    def re_call_all_models(self, user_prompt, system_prompt, es_doc):
        model_ext = {
            "model_ext": {
                # 'disable_analysis': True,
                "temperature": 0.8,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(json.dumps(user_prompt), ensure_ascii=False, indent=4)}
                ]
            }
        }
        models = [['volcengine', 'deepseek-r1'], ['weibo', 'deepseek-r1'], ['weibo', 'deepseek-v3'],
                  ['volcengine', 'Doubao-1.5-pro-32k'], ['volcengine', 'deepseek-v3'], ['dashscope', 'qwen-long'],
                  ['azure', 'gpt-4o'], ['azure', 'gpt-4o-mini'], ['azure', 'gpt-4.1']]
        ret = {}
        from jsonpath import jsonpath
        groups = []
        for utype, model_id in models:
            params = {
                'model_id': model_id,
                'appkey': self.source,
                'type': utype,
                'use_ext_first': '1',
                'message': 'none'
            }
            try:
                js = self.token_help.call_model(params, model_ext)
                ans = jsonpath(js, '$.response_data.choices[0].message.content')[0]
                # self.log.info(f'ans debug:{ans}')
                ans = re.sub('^[^\[\}\]\{]+', '', ans)
                ans = re.sub('[^\[\}\]\{]+$', '', ans)
                self.log.info(f'ans debug:{ans}')
                groups = json.loads(ans)
                self.log.info(f'get mode ret succ:{utype} {model_id}')
                es_doc['model_company'] = params['type']
                es_doc['model_id'] = params['model_id']
                break
            except:
                er_msg = traceback.format_exc()
                self.log.error(f'model err:{utype} {model_id} {er_msg}')
        return groups

    def img_to_model_ext(self, imgs=[], role=[], type='url'):
        import base64
        res = []
        for i in range(len(imgs)):
            tmp = {
                "type": "image_url",
                "image_url": {
                    "url": imgs[
                        i] if type == 'url' else f"data:image/jpeg;base64,{base64.b64encode(open(imgs[i], 'rb').read()).decode('utf-8')}"
                },
                "role": 'first_frame' if i + 1 > len(role) else role[i]
            }
            res.append(tmp)
        return res

    def call_model_zp_video(self,
                            user_prompt="多个镜头。一名侦探进入一间光线昏暗的房间。他检查桌上的线索，手里拿起桌上的某个物品。镜头转向他正在思索。 --ratio 16:9 --duration 10",
                            system_prompt="", model_type="volcengine:Doubao-Seedance-1.0-pro-250528",
                            source="356732087",
                            imgs=[]):
        if not user_prompt and not system_prompt:
            return ""
        # 线上使用
        model_ext = {
            "model_ext": {
                # 'disable_analysis': True,
                "temperature": 0.8,
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ]
            }
        }
        if imgs:
            model_ext["model_ext"]["content"].extend(imgs)
        utype, model_id = model_type.split(":")
        params = {
            'model_id': model_id,
            'appkey': self.source,
            'type': utype,
            'use_ext_first': '1',
            'message': 'none'
        }

        llm_res_json = self.call_model(params, model_ext,
                                       source=source) if debug_module == "online" else self.call_model_local(params,
                                                                                                             model_ext,
                                                                                                             source=source)

        return llm_res_json

    def call_model_zp_video_split(self, videoUrl=test_video_url, duration=167, model_type=models_dict['video-split'],
                                  source="356732087"):
        # 线上使用
        model_ext = {
            "model_ext": {
                "Action": "SplitVideoParts",
                "VideoUrl": videoUrl
            },
            "api_ext": {
                "file-duration": duration
            }
        }
        utype, model_id = model_type.split(":")
        params = {
            'model_id': model_id,
            'appkey': self.source,
            'type': utype,
            'use_ext_first': '1',
            'message': 'none'
        }
        self.log.info(f"llm_res_input params:{params}, model_ext {model_ext}, source {source}")
        llm_res_json = self.call_model(params, model_ext,
                                       source=source) if debug_module == "online" else self.call_model_local(params,
                                                                                                             model_ext,
                                                                                                             source=source)
        self.log.info(f"llm_res_output_json:{llm_res_json}")
        return llm_res_json

    def query_video_task_status(self, task_id, model_type="volcengine:Doubao-Seedance-1.0-pro-250528",
                                single_file_name="defaut"):
        """
        查询视频生成任务的状态并轮询直到任务完成或失败

        参数:
            ark_api_key (str): ARK 平台的 API 密钥
            task_id (str): 视频生成任务的 ID (如 cgt-2025****)

        返回:
            dict: 任务成功或失败的详细结果
        """

        model_ext = {
            "model_ext": {
                "id": task_id
            },
            "api_ext": {"request_method": "get_video"}
        }
        utype, model_id = model_type.split(":")
        params = {
            'model_id': model_id,
            'appkey': self.source,
            'type': utype,
            'use_ext_first': '1',
            'message': 'none'
        }

        self.log.info("----- 开始轮询任务状态 -----")
        while True:
            try:
                # 发送 GET 请求查询任务状态
                task_result = self.call_model(params, model_ext) if debug_module == "online" else self.call_model_local(
                    params, model_ext)
                # 获取任务状态
                status = task_result.get("response_data", {}).get("status", "")
                if not status:
                    self.log.info("无法获取任务状态，重试中...")
                    time.sleep(10)
                    continue

                # 根据状态处理
                if status == "succeeded":
                    self.log.info("----- 任务成功完成 -----")
                    save_path = self.download_video_from_data(task_result, single_file_name)
                    return save_path
                elif status == "failed":
                    self.log.info("----- 任务执行失败 -----")
                    error_info = task_result.get("error", "无详细错误信息")
                    self.log.info(f"错误详情: {error_info}")
                    return task_result
                else:
                    # 任务未完成（如 pending、running 等状态），继续轮询
                    self.log.info(f"当前状态: {status}，10秒后重试...")
                    time.sleep(10)

            except requests.exceptions.RequestException as e:
                # 处理网络请求异常（如超时、连接失败等）
                self.log.error(f"请求异常: {str(e)}，5秒后重试...")
                time.sleep(5)

    def query_video_split_task_status(self, task_id="87A6590C-DD23-5336-9274-DC885E3AF878",
                                      model_type=models_dict["video-split"]):
        """
        查询视频生成任务的状态并轮询直到任务完成或失败

        参数:
            ark_api_key (str): ARK 平台的 API 密钥
            task_id (str): 视频生成任务的 ID (如 cgt-2025****)

        返回:
            dict: 任务成功或失败的详细结果
        """

        model_ext = {
            "model_ext": {
                "Action": "GetAsyncJobResult",
                "JobId": task_id
            }
        }
        utype, model_id = model_type.split(":")
        model_id = "GetAsyncJobResult"
        params = {
            'model_id': model_id,
            'appkey': self.source,
            'type': utype,
            'use_ext_first': '1',
            'message': 'none'
        }
        self.log.info("-----split_task 开始轮询任务状态 -----")
        while True:
            try:
                # 发送 GET 请求查询任务状态
                task_result = self.call_model(params, model_ext) if debug_module == "online" else self.call_model_local(
                    params, model_ext)
                # 获取任务状态
                status = task_result.get("response_data", {}).get("Data", {}).get("Status", "")
                code = task_result.get("code", "")
                if code != 200:
                    self.log.info(f"split_task_code != 200,{task_result['msg']}")
                    return task_result
                if not status:
                    self.log.info("split_task无法获取任务状态，重试中...")
                    time.sleep(10)
                    continue

                # 根据状态处理
                if status == "PROCESS_SUCCESS":
                    self.log.info("----- split_task任务成功完成 -----")
                    self.log.info(f"llm_res_output_split_task_result:{task_result}")
                    # save_path = self.download_video_from_data(task_result)
                    return task_result
                elif status == "PROCESS_FAILED":
                    self.log.info("----- split_task任务执行失败 -----")
                    error_info = task_result.get("response_data", {}).get("Data", {}).get("ErrorMessage", "无详细错误信息")
                    self.log.info(f"split_task错误详情: {error_info}")
                    return task_result
                else:
                    # 任务未完成（如 pending、running 等状态），继续轮询
                    self.log.info(f"split_task当前状态: {status}，10秒后重试...")
                    time.sleep(10)

            except requests.exceptions.RequestException as e:
                # 处理网络请求异常（如超时、连接失败等）
                self.log.error(f"split_task请求异常: {str(e)}，5秒后重试...")
                time.sleep(5)

    def download_video_from_data(self, data, save_dir="tmp/tmp_video_res"):
        """
        从数据中解析视频URL并下载

        参数:
            data: 包含视频信息的字典
            save_dir: 视频保存目录
        """
        try:
            # 解析数据结构，提取video_url
            response_data = data.get("response_data", {})
            content = response_data.get("content", {})
            video_url = content.get("video_url")

            if not video_url:
                self.log.info("未找到视频URL")
                return

            # 从URL中提取文件名（取最后一段作为文件名）
            parsed_url = urlparse(video_url)
            file_name = os.path.basename(parsed_url.path)
            save_path = save_dir + f"_{file_name}"
            if not file_name.endswith((".mp4", ".mov", ".avi")):
                save_path += ".mp4"  # 确保文件后缀正确
            # save_path = os.path.join(save_dir, file_name)

            # 下载视频
            self.log.info(f"开始下载视频: {video_url}")
            self.log.info(f"保存路径: {save_path}")
            full_file_path = os.path.dirname(save_path)
            save_dir_final = os.path.dirname(full_file_path)
            # 创建保存目录
            os.makedirs(save_dir_final, exist_ok=True)

            # 发送请求并流式下载
            with requests.get(video_url, stream=True, timeout=30) as r:
                r.raise_for_status()  # 检查请求是否成功
                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):  # 分块写入
                        if chunk:
                            f.write(chunk)

            self.log.info(f"视频下载完成: {save_path}")
            return save_path
        except Exception as e:
            self.log.error(f"下载失败: {str(e)}")

    def test_role(self):
        tf = token_fresh()
        user_prompt = """当前有一个《莲花楼》中李相夷的微博角
        色账号，该账号需要根据李相夷的生平经历、人物性格、说话风格和网友进行聊天。\n现在李相夷和一位昵称为“沐茗Fruit”网友有如下对话内容：“06月23日16时 李相夷：准备考试辛苦啦，和我说说复习情况\n06月23日16时 沐茗Fruit：怎
        么，花花你要给我出出复习的注意？[doge]\n06月23日16时 李相夷：乐意呀，你哪科较薄弱呢？\n06月23日16时 沐茗Fruit：我……\n06月23日16时 李相夷：别着急，慢慢说，薄弱科目是啥？\n06月23日16时 沐茗Fruit：其实最主要是考专>业科目\n06月23日16时 李相夷：专业科目内容繁杂，你复习得咋样啦？\n06月23日16时 沐茗Fruit：还好\n06月23日16时 李相夷：看来复习挺顺利嘛，有没有哪个知识点特难搞？\n06月23日16时 沐茗Fruit：有好多……花花我太难了[苦涩]”。\n根据以上内容，现在时间是06月23日16时，请代入李相夷的视角，以轻松愉快的语气加入对话，分享相似经历或提出有趣的问题。\n回复要求：\n- 保持闲聊的轻松氛围，不要过于严肃。\n- 回复内容需与对方分享的内容相关，可适度>延伸话题。\n- 不超过30字，保持对话的连贯性。\n- 使用简单易懂的语言，避免专业术语。\n  - 用户【处境、情绪、意图】信息：用户：\n- **{处境}**：准备专业科目考试遇难题\n- **{情绪}**：因难题多感到困扰和苦涩\n- **{意图
        }**：向对方倾诉备考的艰难\n\n虚拟角色：\n- **{回复策略}**：倾听并给予安慰，引导说具体难题\n- **{建议回复}**：别着急，说说哪些知识点难搞定？  - 直接输出回复本身，不要做任何解释说明，不要用引号，不包含角色名。\n- 重点回答对方说的最后一句话：06月23日16时 ：有好多……花花我太难了[苦涩]。\n- 回复内容不要包含具体日期和时间。\n"""
        model_test = ["weibo:deepseek-r1", "volcengine:DeepSeek-R1-250528"]
        res1 = tf.call_model_zp(user_prompt, "", model_test[0])
        print(res1)
        res2 = tf.call_model_zp(user_prompt, "", model_test[1])
        print(res2)

    def video_smart_split(self):
        res1 = self.call_model_zp_video_split()
        task_id = res1.get("response_data", {}).get("RequestId", "")
        print(res1)
        res2 = self.query_video_split_task_status(task_id)


def test_video(tf, prompt="落霞与孤鹜齐飞，秋水共长天一色", modelType=models_dict["video-huoshan"]):
    video_id = tf.call_model_zp_video(prompt, model_type=modelType)
    # video_id = {}
    save_path = tf.query_video_task_status(video_id.get("response_data", {}).get("id", "cgt-20250724193620-njqsn"))
    print(save_path)

if __name__ == '__main__':
    tf = token_fresh()
    # user_prompt = """
    # 生成一张可爱的小猫咪图片
    # """
    # test = tf.call_model_zp_img(user_prompt, "", model_type="volcengine:volc-txt2img-21", source="116913455")
    # test = tf.call_model_zp_img(user_prompt, "", "volcengine:Doubao-Seedream-4.0-250828", source="116913455")
    #
    # tf.video_smart_split()

    # tf.test_role()

    # sys.exit()
    # print(video_id)
    user_prompt = """当前有一个《莲花楼》中李相夷的微博角
色账号，该账号需要根据李相夷的生平经历、人物性格、说话风格和网友进行聊天。\n现在李相夷和一位昵称为“沐茗Fruit”网友有如下对话内容：“06月23日16时 李相夷：准备考试辛苦啦，和我说说复习情况\n06月23日16时 沐茗Fruit：怎
么，花花你要给我出出复习的注意？[doge]\n06月23日16时 李相夷：乐意呀，你哪科较薄弱呢？\n06月23日16时 沐茗Fruit：我……\n06月23日16时 李相夷：别着急，慢慢说，薄弱科目是啥？\n06月23日16时 沐茗Fruit：其实最主要是考专>业科目\n06月23日16时 李相夷：专业科目内容繁杂，你复习得咋样啦？\n06月23日16时 沐茗Fruit：还好\n06月23日16时 李相夷：看来复习挺顺利嘛，有没有哪个知识点特难搞？\n06月23日16时 沐茗Fruit：有好多……花花我太难了[苦涩]”。\n根据以上内容，现在时间是06月23日16时，请代入李相夷的视角，以轻松愉快的语气加入对话，分享相似经历或提出有趣的问题。\n回复要求：\n- 保持闲聊的轻松氛围，不要过于严肃。\n- 回复内容需与对方分享的内容相关，可适度>延伸话题。\n- 不超过30字，保持对话的连贯性。\n- 使用简单易懂的语言，避免专业术语。\n  - 用户【处境、情绪、意图】信息：用户：\n- **{处境}**：准备专业科目考试遇难题\n- **{情绪}**：因难题多感到困扰和苦涩\n- **{意图
}**：向对方倾诉备考的艰难\n\n虚拟角色：\n- **{回复策略}**：倾听并给予安慰，引导说具体难题\n- **{建议回复}**：别着急，说说哪些知识点难搞定？  - 直接输出回复本身，不要做任何解释说明，不要用引号，不包含角色名。\n- 重点回答对方说的最后一句话：06月23日16时 ：有好多……花花我太难了[苦涩]。\n- 回复内容不要包含具体日期和时间。\n"""
    model_test = ["volcengine:DeepSeek-V3-250324", "weibo:deepseek-r1", "dashscope:deepseek-r1",
                  "volcengine:DeepSeek-R1-250528",models_dict['text-gemini']]
    source = "116913455"
    # res1 = tf.call_model_google(user_prompt, "", model_test[-1])
    res2 = tf.call_model_zp(user_prompt, "", models_dict['text-qwen'])
    import datetime

    next_month = datetime.datetime.now() + datetime.timedelta(month_clu=1)
    print(next_month)

