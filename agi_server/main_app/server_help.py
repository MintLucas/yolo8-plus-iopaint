#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/11/06 14:58
# @Author  : zhipeng16
# @Email   : zzzp50@ustc.edu
# @File    : server_help.py
# @Usage   : Describe the file's purpose
from util.mylogging import get_logger
import requests
class Server_Help:
    def __init__(self, log = get_logger("server_log/Server_Help")):
        self.log = log
        self.session = requests.session()
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
        self.log.info(f"开始同步LLM结果 | task_id: {task_id} | 待同步内容: {text_content[:200]}")  # 限制内容长度，避免日志过长
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