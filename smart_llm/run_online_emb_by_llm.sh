#!/bin/bash
set -x
cur_dt=`date +%Y%m%d`
del_dt=`date +%Y%m%d -d "${cur_dt} - 7 days"`

dim='5120'
model='baichuan2'
model_type='model1'
dtype='bf16'

if [[ $# -eq 1 ]]; then
    model_type=$1
elif [[ $# -eq 2 ]]; then
    model_type=$1
    dim=$2
elif [[ $# -eq 3 ]]; then
    model=$1
    model_type=$2
    dim=$3
elif [[ $# -eq 3 ]]; then
    model=$1
    model_type=$2
    dim=$3
    dtype=$4
fi

model_path="/data3/push_recall/models/Baichuan2-13B-Chat"
log_path="/data3/push_recall/content_emb/${model}_code/logs"
if [[ ${model} == "baichuan2" ]]; then
    model_path="/data3/push_recall/models/Baichuan2-13B-Chat"
    log_path="/data3/push_recall/content_emb/${model}_code/logs"
elif [[ ${model} == "chatglm3" ]]; then
    model_path="/data3/push_recall/models/chatglm3-6b"
    log_path="/data3/push_recall/content_emb/${model}_${dim}_code/logs"
elif [[ ${model} == "chatglm" ]]; then
    model_path="/data3/push_recall/models/chatglm-6b"
    log_path="/data3/push_recall/content_emb/emb_${dim}_code/logs"
fi
checkpoint_path="/data3/push_recall/content_emb/result/content_emb_${model}_${dim}_${model_type}"
local_file_prefix="/data3/push_recall/content_emb/${model}_${dim}/push_content_emb_${model}_${dim}"
local_file="${local_file_prefix}_${cur_dt}"

wait
# 执行emb推理任务
cd llm_content_emb
CUDA_VISIBLE_DEVICES=1 /data3/env/miniconda3/envs/glm/bin/python run_online_emb_by_llm.py --local_file ${local_file} --model_type ${model} --model_path ${model_path} --checkpoint_path ${checkpoint_path} --dim ${dim} --dtype ${dtype} --dt ${cur_dt} --del_dt ${del_dt}

wait
# emb文件传输到191
rsync -avz --progress ${local_file} 10.54.25.191::pir/content_emb/${model}_${dim}_emb/

wait
# emb文件清理
del_file="${local_file_prefix}_${del_dt}"
if [ -f ${del_file} ]; then
    rm -rf ${del_file}
fi

wait
# emb推理日志清理
log="${log_path}/run_online_emb_${dim}_${model}_${del_dt}"
if [ -f ${log} ]; then
    rm -rf ${log}
fi
