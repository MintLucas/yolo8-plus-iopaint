#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/4/26 10:17
# @Author  : zhipeng16
# @concat: mintlzp@mail.ustc.edu.cn
# @Site: 
# @File: erternal_server_dataCaculate.py
# @Software: PyCharm
# @Usage:
from util import mylogging
from fastapi import FastAPI, Request, UploadFile
import traceback, re, os, requests, time, json
import datetime
import uuid
# from util.es_util import ES
import random
from queue import Queue
from smart_video.smart_video_split import Smart_video_split
class CVServer:
    def __init__(self, log = mylogging.get_logger("CVServer")):
        self.log = log
        self.session = requests.session()
        self.smart_video = Smart_video_split(self.log)

    
    
    def getErrorTrace(self):
        import traceback
        self.log.error(traceback.format_exc())


import asyncio
if __name__ == '__main__':

    et_s = CVServer()
    et_s.getErrorTrace()