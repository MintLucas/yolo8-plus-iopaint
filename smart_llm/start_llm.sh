#!/bin/bash
set -x

# ==============================================================================
#  LLM API Service Launcher
#  Description: This script starts the LLM FastAPI service in the background.
#  Usage: ./run_online_emb_by_llm.sh [model_name] [dimension] [dtype] [port] [cuda_device]
#  Example: ./run_online_emb_by_llm.sh baichuan2 5120 bf16 8000 0
# ==============================================================================

# --- Default Configuration ---
model='baichuan2'
dim='5120'
dtype='bf16'
port='8000'
cuda_device='0'

# --- Argument Parsing ---
# Override defaults with command-line arguments if provided
if [[ $# -ge 1 ]]; then
    model=$1
fi
if [[ $# -ge 2 ]]; then
    dim=$2
fi
if [[ $# -ge 3 ]]; then
    dtype=$3
fi
if [[ $# -ge 4 ]]; then
    port=$4
fi
if [[ $# -ge 5 ]]; then
    cuda_device=$5
fi

# --- Set Paths ---
# Get the directory where the script is located
script_dir=$(cd $(dirname "$0") && pwd)

# Define and create the log directory
log_dir="${script_dir}/logs"
mkdir -p ${log_dir}

# Define model paths (keeping original logic for model locations)
model_path="/data3/push_recall/models/Baichuan2-13B-Chat" # Default
if [[ ${model} == "baichuan2" ]]; then
    model_path="/data3/push_recall/models/Baichuan2-13B-Chat"
elif [[ ${model} == "chatglm3" ]]; then
    model_path="/data3/push_recall/models/chatglm3-6b"
elif [[ ${model} == "chatglm" ]]; then
    model_path="/data3/push_recall/models/chatglm-6b"
fi

# --- Execute Python Service ---
# Using the python interpreter from your specified conda environment
python_executable="/data3/env/miniconda3/envs/glm/bin/python"
python_script="${script_dir}/server_llm.py"
log_file="${log_dir}/llm_service_${model}_${port}.log"

# Run the FastAPI server in the background using nohup
# All stdout and stderr will be redirected to the log file
nohup CUDA_VISIBLE_DEVICES=${cuda_device} ${python_executable} ${python_script} \
    --model_type ${model} \
    --model_path ${model_path} \
    --dim ${dim} \
    --dtype ${dtype} \
    --port ${port} \
    --host "0.0.0.0" > ${log_file} 2>&1 &

# --- Output Information ---
echo "✅ LLM service is starting..."
echo "   - Model: ${model}"
echo "   - Port: ${port}"
echo "   - GPU: ${cuda_device}"
echo "   - PID: $!"
echo "   - Logs: tail -f ${log_file}"