#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/12/3 14:34
# @Author  : zhipeng16
# @concat: mintlzp@mail.ustc.edu.cn
# @Site: 
# @File: external_server_help.py
# @Software: PyCharm
# @Usage:
from util import mylogging
from util.token_util_new import token_fresh
import requests
import traceback
import re
class servel_help():
    def __init__(self, log = mylogging.get_logger('servel_help')):
        self.log = log
        from util.sinawatch import ApiClient
        self.api = ApiClient('http://iconnect.monitor.sina.com.cn', '2019052715', '17cc26DOfNTnk9bZlgwNvoiKk39T8Z')
        self.token_help = token_fresh()
        self.session = requests.session()

    def get_images_url_from_baidu(self, keyword, page_num):
        res_list = []
        try:
            header = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36'}
            # 请求的 url
            url = 'https://image.baidu.com/search/acjson?'
            n = 0
            for pn in range(0, 30 * page_num, 30):
                # 请求参数
                param = {'tn': 'resultjson_com',
                         'logid': '7603311155072595725',
                         'ipn': 'rj',
                         'ct': 201326592,
                         'is': '',
                         'fp': 'result',
                         'queryWord': keyword,
                         'cl': 2,
                         'lm': -1,
                         'ie': 'utf-8',
                         'oe': 'utf-8',
                         'adpicid': '',
                         'st': -1,
                         'z': '',
                         'ic': '',
                         'hd': '',
                         'latest': '',
                         'copyright': '',
                         'word': keyword,
                         's': '',
                         'se': '',
                         'tab': '',
                         'width': '',
                         'height': '',
                         'face': 0,
                         'istype': 2,
                         'qc': '',
                         'nc': '1',
                         'fr': '',
                         'expermode': '',
                         'force': '',
                         'cg': '',  # 这个参数没公开，但是不可少
                         'pn': pn,  # 显示：30-60-90
                         'rn': '30',  # 每页显示 30 条
                         'gsm': '1e',
                         '1618827096642': ''
                         }
                request = requests.get(url=url, headers=header, params=param)
                if request.status_code == 200:
                    print('Request success.')
                request.encoding = 'utf-8'
                # 正则方式提取图片链接
                html = request.text
                image_url_list = re.findall('"thumbURL":"(.*?)",', html, re.S)
                n += 1
                res_list.append(image_url_list)
                self.log.info(f"检索关键词: {keyword}, 成功获取第{n}页，{len(image_url_list)}张图片url")
            return res_list
        except Exception as e:
            error_str = traceback.format_exc()
            self.log.error(traceback.format_exc())
            if res_list:
                return res_list
            else:
                return str(error_str)

if __name__ == '__main__':
    s_h = servel_help()
    s_h.send_alert("AI出题流程完成", "test")
    print(1)
