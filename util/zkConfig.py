#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/4/25 16:35
# @Author  : zhipeng16
# @concat: mintlzp@mail.ustc.edu.cn
# @Site: 
# @File: zkConfig.py
# @Software: PyCharm
# @Usage:
# -*- coding: utf-8 -*-
# 读取zk配置的json到本地

from kazoo.client import KazooClient
import json

zk = KazooClient(
    hosts='zk1.public.data.sina.com.cn:12183,zk2.public.data.sina.com.cn:12183,zk3.public.data.sina.com.cn:12183,zk4.public.data.sina.com.cn:12183,zk5.public.data.sina.com.cn: 12183')
zk.start()
zk.logger.setLevel(10)
# sse = P.zk.create("/push/aigc_ip/smart_address_test","see".encode("utf-8"))
# rt = zk.set('/push/aigc_ip/test',"jiangtao16".encode("utf-8"))
def read(tag_name):
    rt = zk.get('/push/aigc_ip/' + tag_name)
    old_address = json.loads(rt[0].decode("utf-8"))

    json_str = json.dumps(old_address, ensure_ascii=False)
    print(json_str)
    with open('prompt_config/' + tag_name + '.json', 'w', encoding="utf8") as f:
        json.dump(old_address, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    #read("smart_address")
    read("role_info")
    read("default_address")