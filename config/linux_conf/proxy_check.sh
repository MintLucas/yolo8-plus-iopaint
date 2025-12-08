#!/bin/bash
# check_github.sh
# 作用：诊断 GitHub 连接质量和代理状态

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "=== GitHub 网络诊断开始 ===\n"

# 1. 检查 DNS 解析
echo -n "正在解析 github.com IP... "
gh_ip=$(ping -c 1 github.com 2>/dev/null | head -n 1 | awk -F'(' '{print $2}' | awk -F')' '{print $1}')

if [ -z "$gh_ip" ]; then
    echo -e "${RED}失败 (无法解析)${NC}"
else
    echo -e "${GREEN}$gh_ip${NC}"
    # 简单判断 IP 地理位置（粗略）
    echo "尝试 Ping 延迟测试 (3次)..."
    ping -c 3 github.com
fi

echo -e "\n---------------------------------"

# 2. 检查 Git 专属代理配置
echo -e "检查 Git 全局代理配置:"
git_proxy=$(git config --global --get http.proxy)
if [ -z "$git_proxy" ]; then
    echo -e "git http.proxy: ${RED}未设置 (直连)${NC}"
else
    echo -e "git http.proxy: ${GREEN}$git_proxy${NC}"
fi

echo -e "\n---------------------------------"

# 3. 检查系统环境变量代理
echo -e "检查系统环境变量 (影响 curl/wget/conda):"
if [ -z "$http_proxy" ] && [ -z "$https_proxy" ] && [ -z "$ALL_PROXY" ]; then
    echo -e "系统代理变量: ${RED}未设置${NC}"
else
    echo -e "http_proxy:  $http_proxy"
    echo -e "https_proxy: $https_proxy"
fi

echo -e "\n---------------------------------"

# 4. 实际下载速度测试 (尝试连接 GitHub 的 API 端口)
echo "尝试连接 GitHub 端口 (443)..."
# 使用 curl 仅测试连接握手耗时
curl -o /dev/null -s -w "连接耗时: %{time_connect}s\n" https://github.com

echo -e "\n==============================="