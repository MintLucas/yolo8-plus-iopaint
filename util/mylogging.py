#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/09/18 11:18
# @Author  : zhipeng16
# @Email   : zzzp50@ustc.edu
# @File    : mylogging.py
# @Usage   : Describe the file's purpose
import logging
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler
import os

logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("elasticsearch").setLevel(logging.ERROR)
import datetime
import sys
import time


class SafeRotatingFileHandler(TimedRotatingFileHandler):
    def __init__(self, filename, when='h', interval=1, backupCount=0, encoding=None, delay=False, utc=False):
        TimedRotatingFileHandler.__init__(self, filename, when, interval, backupCount, encoding, delay, utc)

    """
    Override doRollover
    lines commanded by "##" is changed by cc
    """

    def doRollover(self):
        """
        do a rollover; in this case, a date/time stamp is appended to the filename
        when the rollover happens.  However, you want the file to be named for the
        start of the interval, not the current time.  If there is a backup count,
        then we have to get a list of matching filenames, sort them and remove
        the one with the oldest suffix.

        Override,   1. if dfn not exist then do rename
                    2. _open with "a" model
        """
        if self.stream:
            self.stream.close()
            self.stream = None
        # get the time that this sequence started at and make it a TimeTuple
        currentTime = int(time.time())
        dstNow = time.localtime(currentTime)[-1]
        t = self.rolloverAt - self.interval
        if self.utc:
            timeTuple = time.gmtime(t)
        else:
            timeTuple = time.localtime(t)
            dstThen = timeTuple[-1]
            if dstNow != dstThen:
                if dstNow:
                    addend = 3600
                else:
                    addend = -3600
                timeTuple = time.localtime(t + addend)
        dfn = self.baseFilename + "." + time.strftime(self.suffix, timeTuple)
        ##        if os.path.exists(dfn):
        ##            os.remove(dfn)

        # Issue 18940: A file may not have been created if delay is True.
        ##        if os.path.exists(self.baseFilename):
        if not os.path.exists(dfn) and os.path.exists(self.baseFilename):
            os.rename(self.baseFilename, dfn)
        if self.backupCount > 0:
            for s in self.getFilesToDelete():
                os.remove(s)
        if not self.delay:
            self.mode = "a"
            self.stream = self._open()
        newRolloverAt = self.computeRollover(currentTime)
        while newRolloverAt <= currentTime:
            newRolloverAt = newRolloverAt + self.interval
        # If DST changes and midnight or weekly rollover, adjust for this.
        if (self.when == 'MIDNIGHT' or self.when.startswith('W')) and not self.utc:
            dstAtRollover = time.localtime(newRolloverAt)[-1]
            if dstNow != dstAtRollover:
                if not dstNow:  # DST kicks in before next rollover, so we need to deduct an hour
                    addend = -3600
                else:  # DST bows out before next rollover, so we need to add an hour
                    addend = 3600
                newRolloverAt += addend
        self.rolloverAt = newRolloverAt


def get_logger(file_name):
    # 配置日志格式
    log_fmt = '%(asctime)s\tFile \"%(filename)s\",line %(lineno)s\t%(levelname)s: %(message)s'
    formatter = logging.Formatter(log_fmt)

    # 确定日志保存路径
    log_path = './log'
    if 'RESULT_DIR' in os.environ:
        log_path = '%s/log' % os.environ['RESULT_DIR']
        # 1. 创建普通日志处理器，用于记录 INFO 和 WARNING，不记录 ERROR
    log_file_path = os.path.join(log_path, f"{file_name}.log")
    log_dir = os.path.dirname(log_file_path)
    
    os.makedirs(log_dir, exist_ok=True)
    
    log_file_handler = RotatingFileHandler(
        filename=log_file_path,
        mode='a',
        maxBytes=1024 * 1024 * 100,
        backupCount=3
    )
    log_file_handler.setFormatter(formatter)
    # 核心改动：添加一个 lambda 过滤器来排除 ERROR 级别的日志
    log_file_handler.addFilter(lambda record: record.levelno < logging.ERROR)

    # 2. 创建错误日志处理器，用于单独记录 ERROR 级别及以上的信息
    error_file_path = os.path.join(log_path, f"{file_name}.err")
    error_file_handler = RotatingFileHandler(
        filename=error_file_path,
        mode='a',
        maxBytes=1024 * 1024 * 100,
        backupCount=1
    )
    error_file_handler.setFormatter(formatter)
    error_file_handler.setLevel(logging.ERROR)

    # 获取或创建日志器实例
    log = logging.getLogger(file_name)
    log.setLevel(logging.INFO)

    # 防止重复添加处理器
    if not log.handlers:
        log.addHandler(log_file_handler)
        log.addHandler(error_file_handler)

    log.propagate = False

    return log

def get_es_logger(file_name):
    day_now = datetime.datetime.now()
    day_now = day_now.strftime('%Y%m%d')
    log_fmt = '%(asctime)s\tFile:%(filename)s_%(lineno)s\t%(levelname)s: %(message)s'
    # log_fmt = '{"log_level":"%(levelname)s","message":%(message)s}'
    # log_fmt = '{}'
    formatter = logging.Formatter(log_fmt)
    log_path = './log_es'
    if 'RESULT_DIR' in os.environ:
        log_path = '%s/log_es' % os.environ['RESULT_DIR']
    
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_file_handler = SafeRotatingFileHandler(filename="%s/%s.log.%s" % (log_path,file_name, day_now), when="MIDNIGHT",
                                                backupCount=7)
    log_file_handler.setFormatter(formatter)
    log_file_handler.propagate = False
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(file_name)
    #log.addHandler(log_file_handler)
    log.propagate = False
    return log
class LevelFilter(object):
    """
  This is a filter which keep only a specific level
  @http://stackoverflow.com/questions/8162419/python-logging-specific-level-only
  """

    def __init__(self, level):
        self.__level = level

    def filter(self, logRecord):
        return logRecord.levelno <= self.__level

def get_logger_mul(file_name,log_path=None):
    from cloghandler import ConcurrentRotatingFileHandler
    f_name_source = file_name
    log_dir = log_path if log_path else './log'
    if not  os.path.exists(log_dir):
        os.makedirs(log_dir)
    file_name=f'{log_dir}/{file_name}'
    logger = logging.getLogger(file_name)
    logger.setLevel(logging.INFO)

    # ================================
    # Standard Error Handler: gt WARN
    # ================================
    LOG_FILENAME = os.path.abspath(file_name+ ".err")
    handler = ConcurrentRotatingFileHandler(LOG_FILENAME, "a", 100 * 1024 * 1024, 5)
    handler.suffix = "%Y-%m-%d.log"
    handler.setLevel(logging.WARN)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(module)s:%(lineno)d] [%(process)d] %(message)s"
    )
    handler.setFormatter(formatter)
    if [str(x) for x in logger.handlers if f'{f_name_source}.err' in str(x)]:
        pass
    else:
        logger.addHandler(handler)


    # ================================
    # Standard Output Handler: INFO ONLY
    # ================================
    # handler = logging.StreamHandler(sys.stdout)
    LOG_FILENAME = os.path.abspath(file_name + ".info")
    handler = ConcurrentRotatingFileHandler(LOG_FILENAME, "a", 100 * 1024 * 1024, 5)

    handler.suffix = "%Y-%m-%d.log"
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(module)s:%(lineno)d] [%(process)d] %(message)s"
    )
    handler.setFormatter(formatter)
    handler.addFilter(LevelFilter(logging.INFO))
    #print(logger.handlers)
    #if not logger.handlers: # 加上这个逻辑 日志就不打印了
    #    logger.addHandler(handler)
    #logger.addHandler(handler)# 走这个逻辑 日志打印两遍
    if [str(x) for x in logger.handlers if f'{f_name_source}.info' in str(x)]:
        pass
    else:
        logger.addHandler(handler)
    #print([str(x) for x in logger.handlers])
    return logger

def get_md_logger(file_name):

    logger = logging.getLogger('RolesModel')
    logger.setLevel(logging.INFO)
    file_handler = TimedRotatingFileHandler('/data0/weibo_bigdata_push/xudong/llm/onlineLog/log.weibo_status_' + file_name, when='H', interval=1)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

if __name__ == "__main__":
    get_logger('server_log/cat_server_log')