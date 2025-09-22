#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/09/22 11:46
# @Author  : zhipeng16
# @Email   : zzzp50@ustc.edu
# @File    : base_video_class.py
# @Usage   : Describe the file's purpose

import sys,os,redis
sys.path.append(os.getcwd())
from util.mylogging import get_logger


class Base_video_class:
    def __init__(self, log=get_logger("base_video_class")):
        self.redis_client = redis.Redis(host="rm57984.eos.grid.sina.com.cn", port=57984, decode_responses=True, db=1)

if __name__ == '__main__':
    base_video_class = Base_video_class()
    sys.exit()
