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


from util import mylogging

logger = mylogging.get_logger('external_server')
ExS = ExternalServer(logger)

if __name__ == '__main__':


    import uvicorn

    uvicorn.run(app='external_server:app', host='0.0.0.0', port=2222, workers=1)





