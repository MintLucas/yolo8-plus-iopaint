# run_online_emb_by_llm.py
import argparse
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

# 从我们的模型文件中导入LLMModel类
from model_loader import LLMModel

# --- 全局变量 ---
# 这个变量将在服务启动时被实例化
llm_service: LLMModel = None

# --- FastAPI App 初始化 ---
app = FastAPI(
    title="LLM API Service 🚀",
    description="一个解耦了模型加载的、生产级的FastAPI服务。",
    version="1.1.0"
)

# --- Pydantic 模型定义 API 输入输出 ---
class ChatRequest(BaseModel):
    prompt: str
    history: List[Dict[str, str]] = []
    max_length: int = 8192
    temperature: float = 0.7
    top_p: float = 0.8

class ChatResponse(BaseModel):
    response: str
    history: List[Dict[str, str]]

# --- FastAPI 生命周期事件 ---
@app.on_event("startup")
async def startup_event():
    """
    应用启动时执行的事件。
    负责解析命令行参数并实例化LLMModel服务。
    """
    global llm_service
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='baichuan2', type=str)
    parser.add_argument('--model_path', required=True, type=str)
    parser.add_argument('--device', default='0', type=str)
    parser.add_argument('--dtype', default='bf16', type=str)
    parser.add_argument('--dim', default='5120', type=str)
    # 添加host和port参数，尽管它们主要由uvicorn使用
    parser.add_argument('--host', default='0.0.0.0', type=str)
    parser.add_argument('--port', default=8000, type=int)
    args = parser.parse_args()

    print("--- API Service is starting up... ---")
    print("--- Initializing model instance... ---")
    llm_service = LLMModel(
        model_path=args.model_path,
        model_type=args.model_type,
        device=args.device,
        dtype=args.dtype,
        dim=args.dim
    )
    print("--- ✅ API Service is ready to accept requests. ---")

# --- API 端点 ---
@app.get("/", summary="服务健康检查")
def read_root():
    return {"status": "ok", "message": "LLM API is running."}

@app.post("/chat", response_model=ChatResponse, summary="生成聊天回复")
async def chat_endpoint(request: ChatRequest):
    """
    接收提示和历史记录，然后从LLM生成响应。
    """
    if not llm_service:
        raise HTTPException(status_code=503, detail="Model service is not available or still loading.")

    try:
        response, history = llm_service.chat(
            prompt=request.prompt,
            history=request.history,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p
        )
        return ChatResponse(response=response, history=history)

    except Exception as e:
        print(f"Error during model inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    # 解析命令行参数以将host和port传递给Uvicorn
    # 完整的参数解析在startup_event中进行，这里只关心uvicorn本身需要的参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0', type=str)
    parser.add_argument('--port', default=8000, type=int)
    # 使用parse_known_args忽略应用内部使用的其他参数
    cli_args, _ = parser.parse_known_args()
    
    uvicorn.run(app, host=cli_args.host, port=cli_args.port)