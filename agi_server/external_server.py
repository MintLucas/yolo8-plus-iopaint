#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 202/9/08 10:02
# @Author  : zhipeng16
# @concat: mintlzp@mail.ustc.edu.cn
# @Site: 
# @File: external_server.py.py
# @Software: PyCharm
# @Usage:
# -*- coding: utf-8 -*-
import os,sys
#添加当前运行脚本的路径解决
sys.path.append(os.getcwd())
import json
from agi_server.erternal_server_dataCaculate import ExternalServer
from functools import wraps
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi import FastAPI, Request, UploadFile
import traceback, re, os, requests, time,copy
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse,JSONResponse
import inspect
import asyncio
import types
from fastapi import Form, File, HTTPException
from typing import Optional, List, Dict, Any
from util.fast_api_middleware import LogRequestsMiddleware

app = FastAPI(docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory="agi_server/static"), name="static")
app.add_middleware(LogRequestsMiddleware)

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )


def log_request(func):
    @wraps(func)
    async def wrapper(request: Request, *args, **kwargs):
        current_function_name = func.__name__   # 获取当前函数名
        par = await request.json()
        ExS.log.info(f'{current_function_name}_recv:{par}')
        return await func(request, *args, **kwargs)
    return wrapper



@app.get('/get_images_url_from_baidu')
async def get_images_url_from_baidu(request: Request):
    par = request.query_params.items()

    par = {x: y for x, y in par}
    ExS.log.info(f'recv:{par}')
    end = {'code':0}
    state, needInfo = ExS.check_feat(par, 'keyword', str)

    if state == 0:
        end['err_msg'] = needInfo
        end['code'] = -1
        return end
    spider_res = ExS.servel_help.get_images_url_from_baidu(par.get("keyword", "微博大眼仔"), par.get("page_num", 1))

    if isinstance(spider_res, str):
        end['err_msg'] = spider_res
        end['code'] = -1
    else:
        end['result'] = spider_res
    current_function_name = inspect.stack()[0][3] # 获取当前函数名
    # end['code'] = int(needInfo.get("id", ""))
    ExS.log.info(f'{current_function_name}_send:{end}')
    return end


@app.post("/get_llm_completion")
async def get_llm_completion(
    request: Request
):
    """
    接收前端提交的 JSON 数据和可选文件，模拟生成 LLM 结果。
    根据文件类型进行不同处理。
    """

    # 从 FormData 中获取名为 'request_data' 的 JSON 字符串，并解析它
    par = await request.json()
    end = {'code': 0}
    state, request_data = ExS.check_feat(par, 'info', dict)
    extendInfo = par.get("extend_info", {})
    if state == 0:
        end['err_msg'] = request_data
        end['code'] = -1
        return end

    # 从解析后的字典中获取各项数据
    task_type = request_data.get('task_type', "")
    model_name = request_data.get('model_name', "weibo:deepseek-v3")
    max_tokens = request_data.get('max_tokens', "")
    user_prompt = request_data.get('user_prompt', """默写静夜思""")
    system_prompt = request_data.get('system_prompt', """
    你现在是一个智能的AI助手。
```""")
    # 获取其他模型控制参数
    id = request_data.get('id', "")
    # 启动异步任务处理
    async def process_call_model():
        try:

            llm_res = ExS.servel_help.token_help.call_model_zp(user_prompt, system_prompt, model_name)
            save_res = "success"
            res = ExS.servel_help.sync_llm_result_post(id, llm_res)
        except Exception as e:
            # 处理异步任务中的异常
            error_message = f"异步任务处理失败: {str(e)}"
            ExS.log.error(error_message)
            save_res = "异步任务处理失败"
            # 可以选择发送错误邮件或记录到日志等
    # 创建并启动异步任务，但不等待它完成
    asyncio.create_task(process_call_model())
    end['result'] = "success"
    end['id'] = id
    return end

@app.post("/get_llm_completion_direct")
async def get_llm_completion_direct(
    request: Request
):
    """
    接收前端提交的 JSON 数据和可选文件，模拟生成 LLM 结果。
    根据文件类型进行不同处理。
    """

    # 从 FormData 中获取名为 'request_data' 的 JSON 字符串，并解析它
    par = await request.json()
    end = {'code': 0}
    state, request_data = ExS.check_feat(par, 'info', dict)
    extendInfo = par.get("extend_info", {})
    if state == 0:
        end['err_msg'] = request_data
        end['code'] = -1
        return end

    # 从解析后的字典中获取各项数据
    task_type = request_data.get('task_type', "")
    model_name = request_data.get('model_name', "weibo:deepseek-v3")
    max_tokens = request_data.get('max_tokens', "")
    user_prompt = request_data.get('user_prompt', "默写静夜思")
    system_prompt = request_data.get('system_prompt', "")
    # 获取其他模型控制参数
    temperature = request_data.get('temperature', "")
    id = request_data.get('id', "")
    frequency_penalty = request_data.get('frequency_penalty', "") # 演示额外参数
    llm_res = ExS.servel_help.token_help.call_model_zp(user_prompt, system_prompt, model_name)
    end['result'] = llm_res
    end['id'] = id
    current_function_name = inspect.stack()[0][3]  # 获取当前函数名
    ExS.log.info(f'{current_function_name}_return:{end}')
    return end


@app.post("/test_api_post") # <-- 这个就是前端调用的接口名称
async def test_api_post(
    # 这些参数直接对应前端 FormData.append() 的键名
    task_type: str = Form(...),          # 对应 formData.append('task_type', currentTask)
    model_name: str = Form(...),         # 对应 formData.append('model_name', selectedModel)
    max_tokens: int = Form(...),         # 对应 formData.append('max_tokens', outputLength)
    user_prompt: str = Form(...),        # 对应 formData.append('user_prompt', userPrompt)
    system_prompt: str = Form(...),      # 对应 formData.append('system_prompt', systemPrompt)
    excel_file: Optional[UploadFile] = File(None) # 对应 formData.append('excel_file', excelFile)
):
    """
    接收前端提交的数据，包括 LLM 相关的参数和可选的 Excel 文件。
    并模拟生成 LLM 结果。
    """
    # ... 后端处理逻辑（读取文件、调用LLM模型等）
    # LLM 结果生成后，通过字典形式返回，例如 {"result": "你的LLM生成内容"}
    llm_result = "这里是你的 LLM 实际生成的结果。"
    return {"result": llm_result}

# 模拟的模型数据。在实际应用中，这可能来自数据库、配置文件或某个模型管理服务。
AVAILABLE_MODELS = [
    {"id": "model_a", "name": "通用模型 A"},
    {"id": "model_b", "name": "专业模型 B"},
    {"id": "model_c", "name": "精简模型 C"},
    {"id": "model_d", "name": "测试模型 D"} # 增加一个模型用于演示
]

@app.get("/get_all_model_type", response_model=List[dict]) # 明确返回类型是字典列表
async def get_all_model_type():
    """
    获取所有可用的模型类型列表。
    """
    print("Received request for /get_all_model_type")
    # 通常，你只会返回模型的ID和名称
    return AVAILABLE_MODELS

from util import mylogging

logger = mylogging.get_logger('external_server')
ExS = ExternalServer(logger)

if __name__ == '__main__':


    import uvicorn

    uvicorn.run(app='external_server:app', host='0.0.0.0', port=2222, workers=1)





