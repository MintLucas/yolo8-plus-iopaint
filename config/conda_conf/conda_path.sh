#!/bin/bash
# fix_conda_path.sh
# 作用：将 Conda 初始化脚本写入 .bashrc，解决找不到 conda 命令的问题

# 1. 根据你提供的路径推导 Base 目录
# 注意：你给的路径里写的是 moniforge3 (可能是 miniforge3 的笔误？这里严格按你给的写)
CONDA_BASE="/workspace/work/moniforge3"
CONDA_INIT_SCRIPT="$CONDA_BASE/etc/profile.d/conda.sh"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "正在检查 Conda 初始化脚本..."

# 2. 检查文件是否存在
if [ -f "$CONDA_INIT_SCRIPT" ]; then
    echo "找到初始化脚本: $CONDA_INIT_SCRIPT"
    
    # 3. 检查是否已经配置过
    if grep -q "$CONDA_INIT_SCRIPT" ~/.bashrc; then
        echo -e "${GREEN}检测到 .bashrc 中已存在相关配置，无需重复添加。${NC}"
    else
        echo "正在将配置写入 ~/.bashrc ..."
        echo "" >> ~/.bashrc
        echo "# Conda Init" >> ~/.bashrc
        echo "source $CONDA_INIT_SCRIPT" >> ~/.bashrc
        echo -e "${GREEN}写入成功！${NC}"
    fi

    echo -e "----------------------------------------"
    echo -e "${GREEN}请务必执行以下命令使配置立即生效：${NC}"
    echo -e "source ~/.bashrc"
    echo -e "----------------------------------------"
else
    echo -e "${RED}错误：在 $CONDA_BASE 下未找到 etc/profile.d/conda.sh${NC}"
    echo "请检查路径是否拼写正确（例如是否为 miniforge3 ？）"
fi