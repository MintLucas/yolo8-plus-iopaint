"""
问题视频路径生成API
功能：接收问题列表，调用核心逻辑生成视频并返回真实路径

# Usage: uvicorn question_video_api:app --reload
# lsof -i :8000
# kill -9 进程ID

"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import uuid  # 用于生成唯一task_id
import time

# --------------------------
# 导入视频生成核心逻辑和工具类
# --------------------------
import sys
import os
sys.path.append(os.getcwd())
from video_generator import process_api_questions_generate_videos  # 导入核心生成函数
from picture_generator import process_api_questions_generate_pictures  # 导入API定义

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

from util.token_util_new import token_fresh, models_dict  # 导入模型工具
from util.oss_util import oss_util  # 导入模型调用工具类

# 1. 初始化FastAPI应用
app = FastAPI(
    title="问题-视频路径生成API",
    description="接收问题列表，返回对应视频路径",
    version="1.0.0"
)

# 2. 定义数据模型（保持不变）
class SingleQuestion(BaseModel):
    question_id: int
    question: str
    choose: List[str]
    answer: List[str]

class ApiRequest(BaseModel):
    question_list: List[SingleQuestion]
    mode_type: str = "pic"
    style_type: str = "q"

class ResultItem(BaseModel):
    question_id: str
    question: str
    video_path: str

class ApiResponse(BaseModel):
    code: int = 0
    result: List[ResultItem]
    task_id: str  # 改为字符串类型，用uuid生成唯一ID

tf = token_fresh()  # 初始化token工具
VIDEO_MODEL_TYPE = models_dict["video-huoshan"]  # 视频模型类型（与video_generator一致）
TASK_RECORD: Dict[str, Dict] = {}  # 存储任务状态，方便测试时查询


# 核心API接口（POST方法）- 已修改为调用真实生成逻辑
@app.post(path="/generate-video-path", response_model=ApiResponse, summary="生成问题对应的视频路径", tags=["视频路径生成"])
def generate_video_path_api(request: ApiRequest):
    # 生成唯一任务ID（方便测试追踪）
    task_id = str(uuid.uuid4())
    start_time = time.time()
    # 记录任务初始状态
    TASK_RECORD[task_id] = {
        "status": "processing",
        "start_time": start_time,
        "total_questions": len(request.question_list)
    }
    if request.mode_type == "video":
        try:
            # 将Pydantic模型列表转为字典列表（适配process_api_questions_generate_videos的参数要求）
            question_list_dict = [q.dict() for q in request.question_list]

            print(question_list_dict)
            
            # 调用核心逻辑，生成视频
            video_results = process_api_questions_generate_videos(
                questions=question_list_dict,
                tf=tf,
                video_model_type=VIDEO_MODEL_TYPE
            )

            # 更新任务状态（统计成功/失败）
            success_count = len([r for r in video_results if r["status"] == "success"])
            TASK_RECORD[task_id].update({
                "status": "completed",
                "end_time": time.time(),
                "duration": round(time.time() - start_time, 2),
                "success_count": success_count,
                "failed_count": len(video_results) - success_count
            })

            # 转换结果格式，适配API响应模型
            response_result = []
            for res in video_results:
                response_result.append(ResultItem(
                    question_id=res["question_id"],
                    question=res["question"],
                    video_path=res["video_path"]  # 真实视频路径（非模拟）
                ))

            return ApiResponse(
                code=0,
                result=response_result,
                task_id=task_id  # 返回唯一任务ID
            )

        except Exception as e:
            # 捕获异常，更新任务失败状态
            error_msg = str(e)
            TASK_RECORD[task_id].update({
                "status": "failed",
                "error_msg": error_msg,
                "end_time": time.time()
            })
            # 抛出异常，告知测试时的错误原因
            raise HTTPException(
                status_code=500,
                detail=f"视频生成失败，task_id: {task_id}, 错误: {error_msg}"
            )
    else:
        try:
            # 将Pydantic模型列表转为字典列表（适配process_api_questions_generate_videos的参数要求）
            question_list_dict = [q.dict() for q in request.question_list]

            print(question_list_dict)
            
            # 调用核心逻辑，生成视频
            video_results = process_api_questions_generate_pictures(
                questions=question_list_dict,
                tf=tf,
                video_model_type=VIDEO_MODEL_TYPE
            )

            # 更新任务状态（统计成功/失败）
            success_count = len([r for r in video_results if r["status"] == "success"])
            TASK_RECORD[task_id].update({
                "status": "completed",
                "end_time": time.time(),
                "duration": round(time.time() - start_time, 2),
                "success_count": success_count,
                "failed_count": len(video_results) - success_count
            })

            # 转换结果格式，适配API响应模型
            response_result = []
            for res in video_results:
                response_result.append(ResultItem(
                    question_id=res["question_id"],
                    question=res["question"],
                    video_path=res["video_path"]  # 真实视频路径（非模拟）
                ))

            return ApiResponse(
                code=0,
                result=response_result,
                task_id=task_id  # 返回唯一任务ID
            )

        except Exception as e:
            # 捕获异常，更新任务失败状态
            error_msg = str(e)
            TASK_RECORD[task_id].update({
                "status": "failed",
                "error_msg": error_msg,
                "end_time": time.time()
            })
            # 抛出异常，告知测试时的错误原因
            raise HTTPException(
                status_code=500,
                detail=f"视频生成失败，task_id: {task_id}, 错误: {error_msg}"
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

    uvicorn.run(app='question_video_api:app', host='0.0.0.0', port=2222, workers=1)