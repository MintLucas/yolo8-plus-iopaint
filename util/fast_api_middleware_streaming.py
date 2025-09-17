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

logger = get_logger("fast_api_middleware_streaming")

class LogRequestsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 1. 记录请求开始时间
        start_time = time.time()

        # 2. 读取并记录请求体 (入参)
        req_body = await self.get_request_body(request)
        try:
            body_json = json.loads(req_body.decode())
            logger.info(f"\n--- 请求开始 ---")
            logger.info(f"请求: {request.method} {request.url.path}")
            logger.info(f"请求体: {json.dumps(body_json, indent=2, ensure_ascii=False)}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.info(f"\n--- 请求开始 ---")
            logger.info(f"请求: {request.method} {request.url.path}")
            logger.info(f"请求体（非JSON）: {req_body.decode()}")

        # 3. 处理请求并获取响应
        response = await call_next(request)

        # 4. 读取并记录响应体 (出参)
        # 注意：这里我们通过重新构建响应来读取响应体
        if isinstance(response, StreamingResponse):
            # 对于流式响应，我们不能直接读取，这里可以只记录状态码
            response_body = f"<Streaming Response>"
        else:
            response_body = await self.get_response_body(response)
            try:
                # 尝试解析JSON格式的响应体
                response_json = json.loads(response_body)
                response_body = json.dumps(response_json, indent=2, ensure_ascii=False)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # 如果不是JSON，就直接作为字符串记录
                pass

        # 5. 记录响应信息和处理时间
        process_time = time.time() - start_time
        logger.info(f"--- 响应结束 ---")
        logger.info(f"状态码: {response.status_code} | 处理时间: {process_time:.4f}s")
        logger.info(f"响应体: {response_body}")

        return response

    async def get_request_body(self, request: Request):
        # 内部方法，用于处理请求流
        async def set_body(request, body):
            async def receive() -> Message:
                return {'type': 'http.request', 'body': body}

            request._receive = receive

        body = await request.body()
        await set_body(request, body)
        return body

    async def get_response_body(self, response: Response):
        """
        从响应中读取响应体，并返回一个新的响应对象。
        """
        # 读取响应体
        response_body = b''
        async for chunk in response.body_iterator:
            response_body += chunk

        # 创建一个新的响应，以确保原始响应流不会被破坏
        return response_body.decode('utf-8')

