#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/11/06 14:58
# @Author  : zhipeng16
# @Email   : zzzp50@ustc.edu
# @File    : server_help.py
# @Usage   : Describe the file's purpose
import util.mylogging
import requests
class Server_Help:
    def __init__(self, log = mylogging.get_logger("log/server_log/Server_Help")):
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

        cs = None
        try:
            # 使用data参数发送URL编码的表单数据
            r = self.session.post(url, headers=headers, data=data)
            cs = r.content.decode()
        except:
            er_msg = traceback.format_exc()
            self.log.error(er_msg)

        self.log.info(f'task_id:{task_id}, text_content:{text_content}, sync_llm_result:{cs}')
        return "success"