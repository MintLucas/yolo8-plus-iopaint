#!/usr/bin/env bash
set -euo pipefail

# FastAPI + vLLM logprobs server 启动脚本
# 放在 agi_server/llm_server 下，不影响旧 gpu_server/keep_alive.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Python 绝对路径（与 gpu_server/run_keep_alive.sh 保持一致）
PYTHON="${PYTHON:-/workspace/work/moniforge3/envs/lyf50_vllm/bin/python}"

export MODEL_PATH="${MODEL_PATH:-/workspace/work/zhipeng16/git/Multi_agent_image_tagging/model/Qwen/Qwen2.5-VL-3B-Instruct}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-10000}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-1}"

mkdir -p logs
LOG_FILE="logs/vllm_server_$(date +%Y%m%d_%H%M%S).log"

echo "[start_server] MODEL_PATH=$MODEL_PATH"
echo "[start_server] HOST=$HOST PORT=$PORT"
echo "[start_server] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[start_server] TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE"
echo "[start_server] GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
echo "[start_server] MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "[start_server] ENFORCE_EAGER=$ENFORCE_EAGER"
echo "[start_server] LOG_FILE=$LOG_FILE"

ARGS=(
  --model "$MODEL_PATH"
  --host "$HOST"
  --port "$PORT"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
)

if [[ "$ENFORCE_EAGER" == "1" || "$ENFORCE_EAGER" == "true" || "$ENFORCE_EAGER" == "True" ]]; then
  ARGS+=(--enforce-eager)
fi

# -u: unbuffered stdout/stderr，便于 tail -f 实时看日志
"$PYTHON" -u vllm_server.py "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"
