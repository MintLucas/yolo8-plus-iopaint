#!/bin/bash
# install_tmux.sh
# 作用：自动识别系统并安装 tmux

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

if [ "$EUID" -ne 0 ]; then 
  echo -e "${RED}请使用 sudo 或 root 权限运行此脚本。${NC}"
  exit 1
fi

echo -e "${GREEN}=== 开始安装 Tmux ===${NC}"

if command -v tmux &> /dev/null; then
    echo -e "${GREEN}检测到 Tmux 已经安装：${NC}"
    tmux -V
    exit 0
fi

# 识别系统并安装
if [ -f /etc/redhat-release ]; then
    echo "检测到系统: CentOS/RHEL"
    yum install -y tmux
elif [ -f /etc/lsb-release ] || [ -f /etc/debian_version ]; then
    echo "检测到系统: Ubuntu/Debian"
    apt-get update
    apt-get install -y tmux
else
    echo -e "${RED}未识别的操作系统，请手动安装 tmux。${NC}"
    exit 1
fi

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Tmux 安装成功！${NC}"
    tmux -V
else
    echo -e "${RED}安装失败。${NC}"
fi