#!/bin/bash
# clone_env.sh
# 用法: ./clone_env.sh [旧环境名] [新环境名]

GREEN='\033[0;32m'
NC='\033[0m'

# 1. 检查参数
if [ $# -ne 2 ]; then
    echo "用法: $0 <被克隆的环境名> <新环境名>"
    echo "示例: $0 ven3.8 ven3.8_backup"
    exit 1
fi

OLD_ENV=$1
NEW_ENV=$2

echo -e "正在将环境 ${GREEN}$OLD_ENV${NC} 克隆为 ${GREEN}$NEW_ENV${NC} ..."

# 2. 执行克隆
# 这里的 source 是为了确保 conda 命令可用（如果你没执行第一个脚本）
CONDA_BASE="/workspace/work/moniforge3"
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda create -n "$NEW_ENV" --clone "$OLD_ENV"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}克隆成功！${NC}"
    echo "激活命令: conda activate $NEW_ENV"
else
    echo "克隆失败，请检查原环境是否存在。"
fi