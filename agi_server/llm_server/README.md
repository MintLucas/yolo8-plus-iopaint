# FastAPI + vLLM Logprobs Server

该目录是新增的独立 LLM logprobs 服务，不影响旧的 `agi_server/gpu_server/keep_alive.py` 部署。

## 作用

为 `IB-LLM-Collab` 提供本地 GPU 模型的 logprobs 能力，用于计算：

```text
H(Y|T) ≈ -log P(Y | T)
I(T;Y) = H(Y) - H(Y|T)
```

Principia 商业接口继续用于生成文本；本服务只负责 logprobs / 条件负对数似然。

## 为什么使用 FastAPI

FastAPI 对 vLLM 推理速度影响很小，通常是毫秒级 HTTP 开销，瓶颈仍然是 GPU 推理。

选择 FastAPI 的原因：

- 便于调试：启动后访问 `/docs` 有 Swagger UI
- 便于扩展：后续可以加批量接口、特定 H(Y|T) 接口、缓存等
- 不影响旧服务：独立目录、独立端口、独立启动脚本

## 文件

| 文件 | 说明 |
|---|---|
| `vllm_server.py` | FastAPI 服务主体，启动时加载 vLLM 模型 |
| `start_server.sh` | 启动脚本 |
| `test_client.py` | 简单测试客户端 |

## 启动

```bash
cd /workspace/work/zhipeng16/git/yolo8-plus-iopaint/agi_server/llm_server
bash start_server.sh
```

默认配置：

```bash
MODEL_PATH=/workspace/work/zhipeng16/git/Multi_agent_image_tagging/model/Qwen/Qwen2.5-VL-3B-Instruct
HOST=0.0.0.0
PORT=8000
CUDA_VISIBLE_DEVICES=0,1,2,3
TENSOR_PARALLEL_SIZE=4
GPU_MEMORY_UTILIZATION=0.85
MAX_MODEL_LEN=10000
ENFORCE_EAGER=1
```

可通过环境变量覆盖：

```bash
PORT=8001 CUDA_VISIBLE_DEVICES=0,1 TENSOR_PARALLEL_SIZE=2 bash start_server.sh
```

## 测试

```bash
python test_client.py --url http://127.0.0.1:8000
```

## API

### `GET /health`

检查服务状态。

```bash
curl http://127.0.0.1:8000/health
```

### `POST /generate`

生成文本，可选返回生成 token 的 logprob。

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Question: What is the capital of France?\nAnswer:",
    "max_tokens": 16,
    "temperature": 0.0,
    "logprobs": 1
  }'
```

### `POST /logprobs`

给定 `prompt` 和目标 `response`，计算 response 在 prompt 条件下的 token logprobs。

```bash
curl -X POST http://127.0.0.1:8000/logprobs \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Question: What is the capital of France?\nAnswer:",
    "response": " Paris",
    "top_logprobs": 5
  }'
```

返回字段：

```json
{
  "total_logprob": -0.123,
  "n_tokens": 1,
  "nll_nats": 0.123,
  "nll_bits": 0.177,
  "avg_nll_bits_per_token": 0.177,
  "tokens": [
    {"token": " Paris", "token_id": 1234, "logprob": -0.123}
  ]
}
```

### `POST /compute_h_y_given_t`

更贴近 IB 实验的接口：给定 T 和 Y，直接计算 `-log P(Y|T)`，单位 bits。

```bash
curl -X POST http://127.0.0.1:8000/compute_h_y_given_t \
  -H 'Content-Type: application/json' \
  -d '{
    "t_message": "France capital = Paris.",
    "y_answer": " Paris"
  }'
```

### `POST /batch_logprobs`

批量计算多个 prompt/response 的 logprobs。

```json
{
  "items": [
    {"prompt": "...", "response": "..."},
    {"prompt": "...", "response": "..."}
  ]
}
```

## 注意事项

1. `response` 前面最好保留必要空格，例如英文答案常写成 `" Paris"`，这会影响 tokenizer 切分和 logprob。
2. 本服务用 `prompt_logprobs` 计算目标 response 的条件 logprob：输入 `prompt + response`，截取 response 部分 token 的 logprob。
3. 如果返回缺失 token logprob，服务会报错，避免 silently 产生错误 H(Y|T)。
4. 后续若需要更高吞吐，应优先使用 `/batch_logprobs`，减少 HTTP 往返。
