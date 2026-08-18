"""
FastAPI + vLLM logprobs 服务

用途
====
部署在 GPU 服务器上，为 IB-LLM-Collab 提供本地 LLM logprobs 能力，
用于计算 H(Y|T)、I(T;Y) 等信息论量。

为什么不用 vLLM 自带 OpenAI server？
===================================
自建 FastAPI 方便后续扩展自定义接口，例如：
- /logprobs：给定 prompt + response，返回 response 每个 token 的 logprob
- /compute_h_y_given_t：直接返回 H(Y|T) bits
- /batch_logprobs：批量计算，减少 HTTP 往返

启动方式
========
    bash start_server.sh

或直接：
    python vllm_server.py --host 0.0.0.0 --port 8000

接口
====
1. GET /health
2. POST /generate
3. POST /logprobs
4. POST /compute_h_y_given_t
5. POST /batch_logprobs
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams


# ─────────────────────────────────────────────────────────────────────────────
# 默认配置
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_BASE_PATH = "/workspace/work/zhipeng16/git/Multi_agent_image_tagging/model/Qwen"
DEFAULT_MODEL_PATH = f"{DEFAULT_BASE_PATH}/Qwen2.5-VL-3B-Instruct"
DEFAULT_GPU_COUNT = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").count(",") + 1


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI 数据结构
# ─────────────────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    logprobs: int | None = None


class GenerateResponse(BaseModel):
    text: str
    token_ids: list[int]
    token_logprobs: list[float | None] | None = None
    top_logprobs: list[dict[str, float]] | None = None


class LogprobsRequest(BaseModel):
    """给定 prompt 和目标 response，计算 response 在 prompt 条件下的 token logprobs。"""
    prompt: str = Field(..., description="条件上下文，例如 T 或解码 prompt")
    response: str = Field(..., description="目标文本，例如 gold answer Y")
    temperature: float = 0.0
    top_logprobs: int = 0


class TokenLogprob(BaseModel):
    token: str
    token_id: int
    logprob: float | None


class LogprobsResponse(BaseModel):
    prompt: str
    response: str
    tokens: list[TokenLogprob]
    total_logprob: float
    n_tokens: int
    nll_nats: float
    nll_bits: float
    avg_nll_bits_per_token: float


class BatchLogprobsRequest(BaseModel):
    items: list[LogprobsRequest]


class BatchLogprobsResponse(BaseModel):
    results: list[LogprobsResponse]
    mean_nll_bits: float
    mean_nll_bits_per_sample: float


class HYGivenTRequest(BaseModel):
    """计算 H(Y|T) 的单样本近似：-log P(Y|T)，单位 bits。"""
    t_message: str
    y_answer: str
    instruction: str = "Given the message, output the final answer only."
    temperature: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 服务状态
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ServerState:
    llm: LLM | None = None
    model_path: str = DEFAULT_MODEL_PATH
    tokenizer: Any | None = None


state = ServerState()
app = FastAPI(title="IB-LLM vLLM Logprobs Server", version="0.1.0")


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _require_llm() -> LLM:
    if state.llm is None:
        raise HTTPException(status_code=503, detail="LLM not initialized")
    return state.llm


def _decode_token(token_id: int) -> str:
    if state.tokenizer is None:
        return str(token_id)
    try:
        return state.tokenizer.decode([token_id])
    except Exception:
        return str(token_id)


def _build_answer_prompt(t_message: str, instruction: str) -> str:
    return f"{instruction}\n\nMessage:\n{t_message}\n\nFinal answer:"


def _logprobs_for_prompt_response(
    prompt: str,
    response: str,
    temperature: float = 0.0,
    top_logprobs: int = 0,
) -> LogprobsResponse:
    """
    计算 response 在 prompt 条件下的逐 token logprob。

    返回：
        logprob_i = ln P(x_i | prompt, x_<i)

        nll_bits =
            -sum(logprob_i) / ln(2)

    注意：
        vLLM 返回的是自然对数 logprob，不是 log2。
    """

    llm = _require_llm()
    tokenizer = state.tokenizer

    if tokenizer is None:
        raise HTTPException(
            status_code=500,
            detail="Tokenizer not initialized",
        )

    if not prompt:
        raise HTTPException(
            status_code=400,
            detail="prompt cannot be empty",
        )

    if not response:
        raise HTTPException(
            status_code=400,
            detail="response cannot be empty",
        )

    full_text = prompt + response
    prompt_char_end = len(prompt)

    # 使用 offset_mapping 获取每个 token 对应的原文字符区间
    try:
        encoded = tokenizer(
            full_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Tokenizer does not support offset_mapping: {e}",
        )

    full_ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])

    if not full_ids:
        raise HTTPException(
            status_code=400,
            detail="full_text has no tokens",
        )

    # 选择与 response 有字符重叠的 token。
    #
    # token 区间为 [start, end)
    # 只要 end > prompt_char_end，就认为它属于 response 部分。
    #
    # 这样即使 prompt 和 response 边界发生 BPE 合并，
    # 也不会再错误地按照 len(prompt_ids) 进行切分。
    response_positions = [
        pos
        for pos, (start, end) in enumerate(offsets)
        if end > prompt_char_end
    ]

    if not response_positions:
        raise HTTPException(
            status_code=400,
            detail="response has no tokens after tokenization",
        )

    response_ids = [
        full_ids[pos]
        for pos in response_positions
    ]

    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=temperature,
        prompt_logprobs=max(1, top_logprobs + 1),
    )

    outputs = llm.generate(
        [full_text],
        sampling_params,
        use_tqdm=False,
    )

    if not outputs:
        raise HTTPException(
            status_code=500,
            detail="vLLM returned no output",
        )

    out = outputs[0]
    prompt_logprobs = out.prompt_logprobs

    if prompt_logprobs is None:
        raise HTTPException(
            status_code=500,
            detail="vLLM did not return prompt_logprobs",
        )

    tokens: list[TokenLogprob] = []
    total_logprob = 0.0

    for pos, token_id in zip(response_positions, response_ids):
        if pos >= len(prompt_logprobs):
            raise HTTPException(
                status_code=500,
                detail=(
                    f"prompt_logprobs index out of range: "
                    f"pos={pos}, length={len(prompt_logprobs)}"
                ),
            )

        lp_dict = prompt_logprobs[pos]

        if lp_dict is None or token_id not in lp_dict:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Missing logprob for response token at position {pos}. "
                    "Increase prompt_logprobs/top_logprobs."
                ),
            )

        lp_obj = lp_dict[token_id]
        logprob_value = float(
            getattr(lp_obj, "logprob", lp_obj)
        )

        total_logprob += logprob_value

        tokens.append(
            TokenLogprob(
                token=_decode_token(token_id),
                token_id=int(token_id),
                logprob=logprob_value,
            )
        )

    # vLLM 的 logprob 是自然对数 ln(P)
    nll_nats = -total_logprob

    # 转为 bits：
    # log2(P) = ln(P) / ln(2)
    nll_bits = nll_nats / math.log(2)

    avg_nll_bits = nll_bits / len(tokens)

    return LogprobsResponse(
        prompt=prompt,
        response=response,
        tokens=tokens,
        total_logprob=total_logprob,
        n_tokens=len(tokens),
        nll_nats=nll_nats,
        nll_bits=nll_bits,
        avg_nll_bits_per_token=avg_nll_bits,
    )


# ─────────────────────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if state.llm is not None else "not_initialized",
        "model_path": state.model_path,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    llm = _require_llm()
    sampling_params = SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        logprobs=req.logprobs,
    )
    outputs = llm.generate([req.prompt], sampling_params, use_tqdm=False)
    out = outputs[0].outputs[0]

    token_logprobs = None
    if out.logprobs is not None:
        token_logprobs = []
        for i, token_id in enumerate(out.token_ids):
            lp_dict = out.logprobs[i] if i < len(out.logprobs) else None
            if lp_dict is not None and token_id in lp_dict:
                lp_obj = lp_dict[token_id]
                token_logprobs.append(float(getattr(lp_obj, "logprob", lp_obj)))
            else:
                token_logprobs.append(None)

    return GenerateResponse(
        text=out.text,
        token_ids=[int(x) for x in out.token_ids],
        token_logprobs=token_logprobs,
        top_logprobs=None,
    )


@app.post("/logprobs", response_model=LogprobsResponse)
def logprobs(req: LogprobsRequest) -> LogprobsResponse:
    return _logprobs_for_prompt_response(
        prompt=req.prompt,
        response=req.response,
        temperature=req.temperature,
        top_logprobs=req.top_logprobs,
    )


@app.post("/compute_h_y_given_t", response_model=LogprobsResponse)
def compute_h_y_given_t(req: HYGivenTRequest) -> LogprobsResponse:
    prompt = _build_answer_prompt(req.t_message, req.instruction)
    return _logprobs_for_prompt_response(
        prompt=prompt,
        response=req.y_answer,
        temperature=req.temperature,
        top_logprobs=0,
    )


@app.post("/batch_logprobs", response_model=BatchLogprobsResponse)
def batch_logprobs(req: BatchLogprobsRequest) -> BatchLogprobsResponse:
    results = [
        _logprobs_for_prompt_response(
            prompt=item.prompt,
            response=item.response,
            temperature=item.temperature,
            top_logprobs=item.top_logprobs,
        )
        for item in req.items
    ]
    mean_nll_bits = sum(r.nll_bits for r in results) / max(len(results), 1)
    return BatchLogprobsResponse(
        results=results,
        mean_nll_bits=mean_nll_bits,
        mean_nll_bits_per_sample=mean_nll_bits,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 启动
# ─────────────────────────────────────────────────────────────────────────────

def init_model(
    model_path: str,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    max_model_len: int,
    enforce_eager: bool,
) -> None:
    print(f"[llm_server] initializing vLLM model: {model_path}")
    print(f"[llm_server] tensor_parallel_size={tensor_parallel_size}, gpu_memory_utilization={gpu_memory_utilization}")
    state.model_path = model_path
    state.llm = LLM(
        model=model_path,
        runner="generate",
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=enforce_eager,
        max_model_len=max_model_len,
        mm_processor_kwargs={
            "use_fast": True,
            "disable_video": True,
        },
    )
    state.tokenizer = state.llm.get_tokenizer()
    print("[llm_server] model initialized")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FastAPI + vLLM logprobs server")
    parser.add_argument("--model", type=str, default=os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--host", type=str, default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--gpu-memory-utilization", type=float, default=float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.85")))
    parser.add_argument("--tensor-parallel-size", type=int, default=int(os.environ.get("TENSOR_PARALLEL_SIZE", str(DEFAULT_GPU_COUNT))))
    parser.add_argument("--max-model-len", type=int, default=int(os.environ.get("MAX_MODEL_LEN", "10000")))
    parser.add_argument("--enforce-eager", action="store_true", default=os.environ.get("ENFORCE_EAGER", "1") == "1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_model(
        model_path=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
