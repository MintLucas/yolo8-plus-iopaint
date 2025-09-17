#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/09/17 15:07
# @Author  : zhipeng16
# @Email   : zzzp50@ustc.edu
# @File    : cv_server.py
# @Usage   : Describe the file's purpose

import sys,os
sys.path.append(os.getcwd())
from util.mylogging import get_logger
from fastapi import FastAPI,Request,Response,Depends

#添加当前运行脚本的路径解决
sys.path.append(os.getcwd())
import json
from agi_server.cv_server_dataCaculate import CVServer
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



@app.get('/deal_local_video')
async def get_images_url_from_baidu(request: Request):
    par = request.query_params.items()
    end = {'code':0}

    return end

@app.post("/split_video")
async def split_video(request: Request,response: Response):
    try:
        data = await request.json()
        url = data['uploaded_video_url']
        material_id = data['material_id']
    except Exception as e:
        trace_info = traceback.format_exc()
        ExS.log.error(
            f"请求处理失败: {e}",
            extra={'request_body': request.body(), 'trace_info': trace_info}
        )
        return {'code':-2,'err_msg':'bad request:'+str(e)}
    end = {'code':0}
    from threading import Thread
    def sub(data):
        ExS.smart_video.gen_video_split(data)
    t = Thread(target=sub, args=(data))
    t.start()
    end['url'] =url
    end['dataid'] = material_id
    return end

from util import mylogging

logger = mylogging.get_logger('CVServer')
ExS = CVServer(logger)

if __name__ == '__main__':


    import uvicorn

    uvicorn.run(app='external_server:app', host='0.0.0.0', port=2222, workers=1)


