#!/bin/bash
# create_py_env.sh
# 用法: ./create_py_env.sh [Python版本] [环境名]

GREEN='\033[0;32m'
NC='\033[0m'

# 1. 交互式输入（如果未提供参数）
if [ $# -lt 2 ]; then
    read -p "请输入 Python 版本 (例如 3.8 或 3.10): " PY_VER
    read -p "请输入新环境名称: " ENV_NAME
else
    PY_VER=$1
    ENV_NAME=$2
fi

# 2. 确保 conda 可用
CONDA_BASE="/workspace/work/moniforge3"
[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ] && source "$CONDA_BASE/etc/profile.d/conda.sh"

echo -e "准备创建环境: ${GREEN}$ENV_NAME${NC} (Python $PY_VER)"

# 3. 创建环境
# -y 表示自动确认 yes
conda create -n "$ENV_NAME" python="$PY_VER" -y

if [ $? -eq 0 ]; then
    echo -e "${GREEN}环境创建完毕！${NC}"
    echo "激活命令: conda activate $ENV_NAME"
else
    echo "创建失败，请检查版本号或网络。"
fi