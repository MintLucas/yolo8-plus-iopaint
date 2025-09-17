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

logger = get_logger("middleware_fast_api")


class LogRequestsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # 读取并记录请求体 (入参)
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

        # 处理请求并获取响应
        response = await call_next(request)

        # 核心改动：捕获并记录响应体
        response_body = b''
        async for chunk in response.body_iterator:
            response_body += chunk

        # 在这里，response_body 已经包含了完整的响应体数据
        # 尝试解析并打印 JSON 格式
        try:
            response_json = json.loads(response_body.decode('utf-8'))
            log_body = json.dumps(response_json, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # 如果不是 JSON，直接打印原始内容
            log_body = response_body.decode('utf-8')

        # 记录响应信息和处理时间
        process_time = time.time() - start_time
        logger.info(f"--- 响应结束 ---")
        logger.info(f"状态码: {response.status_code} | 处理时间: {process_time:.4f}s")
        logger.info(f"响应体: {log_body}")

        # 重新构建响应，并将其返回给客户端
        # 这也是关键步骤，因为原始的响应流已经被消耗了
        return Response(content=response_body, status_code=response.status_code, headers=dict(response.headers))

    async def get_request_body(self, request: Request):
        async def set_body(request, body):
            async def receive() -> Message:
                return {'type': 'http.request', 'body': body}

            request._receive = receive

        body = await request.body()
        await set_body(request, body)
        return body
