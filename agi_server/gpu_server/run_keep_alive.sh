#只能2 、 4卡，由于vllm的限制；确保不走 MPS
export CUDA_MPS_PIPE_DIRECTORY=None
export CUDA_VISIBLE_DEVICES=0,1,2,3
/workspace/work/moniforge3/envs/lyf50_vllm/bin/python keep_alive.py