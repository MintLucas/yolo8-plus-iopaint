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
from external_server_help import servel_help
import random
from queue import Queue
class ExternalServer:
    def __init__(self, log = mylogging.get_logger("ExternalServer")):
        self.log = log
        self.session = requests.session()
        self.servel_help = servel_help(self.log)

    def check_feat(self, par, featName="needInfo", featType=list):
        try:
            if not par.get(featName):
                return 0, '{} must give'.format(featName)
            feat = par.get(featName, '[]')
            if not isinstance(feat, featType):
                if isinstance(feat, str):
                    feat = json.loads(feat)
                if not isinstance(feat, featType):
                    return 0, '{} type error'.format(featName)
            return 1, feat
        except Exception as e:
            self.log.error(traceback.format_exc())
            return 0, '{} type error {}'.format(featName, e)

    def getErrorTrace(self):
        import traceback
        self.log.error(traceback.format_exc())


import asyncio
if __name__ == '__main__':

    et_s = ExternalServer()
    et_s.getErrorTrace()