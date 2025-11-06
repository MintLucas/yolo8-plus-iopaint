#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2025/8/13 19:14
# @Author  : zhipeng16
# @concat: mintlzp@mail.ustc.edu.cn
# @Site: 
# @File: fast_api_middleware_streaming.py
# @Software: PyCharm
# @Usage:
import time
import json
import io
import asyncio
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import Message
from typing import Dict, Any
from util.mylogging import get_logger

class LogRequestsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, log_folder = "default_log"):
        super().__init__(app)
        # 传入日志文件夹名称初始化日志器，get_logger需适配路径逻辑
        self.logger = get_logger(f"server_log/{log_folder}_middleware_fast_api")

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # 读取并记录请求体 (入参)
        req_body = await self.get_request_body(request)
        try:
            body_json = json.loads(req_body.decode())
            self.logger.info(f"\n--- 请求开始 ---")
            self.logger.info(f"请求: {request.method} {request.url.path}")
            self.logger.info(f"请求体: {json.dumps(body_json, indent=2, ensure_ascii=False)}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.logger.info(f"\n--- 请求开始 ---")
            self.logger.info(f"请求: {request.method} {request.url.path}")
            self.logger.info(f"请求体（非JSON）: {req_body.decode()}")

        # 处理请求并获取响应
        response = await call_next(request)

        # 捕获并记录响应体
        response_body = b''
        async for chunk in response.body_iterator:
            response_body += chunk

        # 解析响应体日志
        try:
            response_json = json.loads(response_body.decode('utf-8'))
            log_body = json.dumps(response_json, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, UnicodeDecodeError):
            log_body = response_body.decode('utf-8')

        # 记录响应信息
        process_time = time.time() - start_time
        self.logger.info(f"--- 响应结束 ---")
        self.logger.info(f"状态码: {response.status_code} | 处理时间: {process_time:.4f}s")
        self.logger.info(f"响应体: {log_body}")

        # 重新构建响应
        return Response(content=response_body, status_code=response.status_code, headers=dict(response.headers))

    async def get_request_body(self, request: Request):
        async def set_body(request, body):
            async def receive() -> Message:
                return {'type': 'http.request', 'body': body}
            request._receive = receive
        body = await request.body()
        await set_body(request, body)
        return body