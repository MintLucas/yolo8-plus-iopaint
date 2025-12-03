#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/11/24 11:44
# @Author  : zhipeng16
# @Email   : zzzp50@ustc.edu
# @File    : cat_server_help.py
# @Usage   : Describe the file's purpose
import sys,os
sys.path.append(os.getcwd())
import json
from util.mylogging import get_logger
import requests
from util.token_util_new import token_fresh, models_dict
from util.api_client_new import Api_client
from util.url_util import url_to_base64
class Cat_Server_help:
    def __init__(self, log = get_logger("server_log/Cat_Server_help")):
        self.log = log
        self.session = requests.session()
        self.tf = token_fresh()
        self.prompt_path = "config/prompt/cat_rec_star/create_blog_v2_mode0.md"
        self.supply_key = ["blog_text", "user_info", "history_blog", "reference_blog"]
        self.show__batch_server = Api_client(session=self.session)
    def sync_llm_result_post(self, task_id, text_content):
        # 正式地址
        url = 'http://i.answer.media.weibo.com/admin/task/receivetext'
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}

        # 构建表单数据
        data = {
            "id": task_id,
            "text": text_content
        }

        # 记录同步前的数据日志
        self.log.info(f"开始同步LLM结果 | task_id: {task_id} | 待同步内容: {text_content[:50]}")  # 限制内容长度，避免日志过长
        self.log.debug(f"同步请求详情 | URL: {url} | 请求头: {headers} | 请求数据: {data}")  # 调试级别日志记录详细参数

        response_content = None
        try:
            # 使用data参数发送URL编码的表单数据
            r = self.session.post(url, headers=headers, data=data)
            response_content = r.content.decode()
            # 记录接口返回结果
            self.log.info(f"LLM结果同步完成 | task_id: {task_id} | 接口响应状态码: {r.status_code} | 响应内容: {response_content[:500]}")  # 限制响应内容长度
        except Exception as e:
            er_msg = traceback.format_exc()
            self.log.error(f"LLM结果同步失败 | task_id: {task_id} | 错误信息: {er_msg}")

        return "success" if response_content else "failed"  # 优化返回值，体现实际同步状态

    def run(self, para_dict):
        with open(self.prompt_path, encoding='utf-8') as f:
            base_prompt = f.read()

        """
        1. 当前博文内容：筛选出的优质原创博文，即为你需要转发的原博文内容。
2. 当前博文博主信息：需要转发的原博文博主信息，主要包括领域，昵称。
3. 历史转发博文内容：“跨域星推官”账号过去六个月内高互动的转发博文。
4. 微博站内各领域高互动博文：当前原博文博主所在的领域的近 1个月高互动的博文内容。
        """
        supply_dict = {key:value for key,value in para_dict.items() if key in self.supply_key}
        # base_prompt = base_prompt.format(**para_dict)
        if para_dict['blog_img']:
            use_medias = []
            for one_url in para_dict['blog_img']:
                if "http" in one_url:
                    b64_img = url_to_base64(one_url)
                    use_medias.append({'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64_img}'}})
            result = self.tf.call_model_zp(json.dumps(para_dict), base_prompt, models_dict["text-huoshan"],user_prompt_media=use_medias)
        else:
            result = self.tf.call_model_zp(json.dumps(para_dict), base_prompt, models_dict["text-huoshan"])
        if not result:
            res = {}
        else:
            res = json.loads(result.strip("```json").strip("```"))
        return res
    
if __name__ == '__main__':
    server_help = Cat_Server_help()
    header = server_help.tf.get_token()
    sys.exit()
