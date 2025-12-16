#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/11/24 14:08
# @Author  : zhipeng16
# @Email   : zzzp50@ustc.edu
# @File    : cat_main_server.py
# @Usage   : Describe the file's purpose
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import uuid  # 用于生成唯一task_id
import time
import asyncio
import json
# --------------------------
# 导入视频生成核心逻辑和工具类
# --------------------------
import sys
import os
sys.path.append(os.getcwd())
from util.fast_api_middleware_logging import LogRequestsMiddleware
from agi_server.cat_rec_star.cat_server_help import Cat_Server_help
import traceback
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)


server_help = Cat_Server_help()

from util.token_util_new import token_fresh, models_dict  # 导入模型工具
from util.oss_util import oss_util  # 导入模型调用工具类

# 1. 初始化FastAPI应用
app = FastAPI(
    title="问题-视频路径生成API",
    description="接收问题列表，返回对应视频路径",
    version="1.0.0"
)
# app.add_middleware(LogRequestsMiddleware, log_folder="server_question_multi")

# 2. 定义数据模型（保持不变）
class SingleQuestion(BaseModel):
    question_id: int
    question: str
    choose: List[str]
    answer: List[str]

class ApiRequest(BaseModel):
    task_id: int = 100
    blog_text: str = ""
    uid: str = ""
    nick: str = ""
    blog_img: list = []
    blog_video: list = []
    blog_video_asr: list = []
    user_info: str = ""
    history_blog: list = []
    reference_blog: list = []
    
class ResultItem(BaseModel):
    question_id: str
    question: str
    video_path: str

class ApiResponse(BaseModel):
    code: int = 0
    result: dict
    task_id: str  # 改为字符串类型，用uuid生成唯一ID

tf = token_fresh()  # 初始化token工具
VIDEO_MODEL_TYPE = models_dict["video-huoshan"]  # 视频模型类型（与video_generator一致）
TASK_RECORD: Dict[str, Dict] = {}  # 存储任务状态，方便测试时查询


import json  # 新增json模块导入

@app.post(path="/create_repost_blog", response_model=ApiResponse, summary="跨域星推官博文生成", tags=["cat_rec_star"])
async def create_repost_blog(request: ApiRequest):
    # 新增：记录完整传入数据体日志
    server_help.log.info(f"收到请求：传入数据体={json.dumps(request.dict(), ensure_ascii=False)}")  # 完整记录传入数据

    # 生成唯一任务ID（方便测试追踪）
    task_id_server = str(uuid.uuid4())
    start_time = time.time()
    # 记录任务初始状态
    TASK_RECORD[task_id_server] = {
        "status": "processing",
        "start_time": start_time
    }
    para_dict = request.dict()
    task_id = para_dict.get("task_id", -1)
    try:
        res = server_help.run(para_dict)
        # 构建返回结果对象
        result = ApiResponse(
            code=200,
            result=res,
            task_id=str(para_dict["task_id"])  # 返回唯一任务ID
        )
        
        # 新增：记录完整传出数据体日志
        server_help.log.info(f"返回结果：传出数据体={json.dumps(result.dict(), ensure_ascii=False)}")  # 完整记录返回数据
        return result

    except Exception as e:
        # 捕获异常，更新任务失败状态
        error_msg = str(e)
        server_help.log.error(f"{traceback.format_exc()}")
        TASK_RECORD[task_id].update({
            "status": "failed",
            "error_msg": error_msg,
            "end_time": time.time()
        })
        # 抛出异常，告知测试时的错误原因
        raise HTTPException(
            status_code=-1,
            detail=f"生成失败，task_id: {task_id}, 错误: {error_msg}"
        )


# --------------------------
# 新增：任务状态查询接口（测试时可查看生成进度和结果）
# --------------------------
@app.get(
    path="/task-status/{task_id}",
    summary="查询视频生成任务状态（测试用）"
)
def get_task_status(task_id: str):
    if task_id not in TASK_RECORD:
        raise HTTPException(status_code=404, detail=f"任务ID {task_id} 不存在")
    return {
        "task_id": task_id,
        **TASK_RECORD[task_id]
    }

# 运行说明（保持不变）
# 依赖安装：pip install fastapi uvicorn
# 运行命令：uvicorn question_video_api:app --reload
# 接口文档：http://127.0.0.1:8000/docs

if __name__ == '__main__':


    import uvicorn

    uvicorn.run(app='cat_main_server:app', host='0.0.0.0', port=14546, workers=3)