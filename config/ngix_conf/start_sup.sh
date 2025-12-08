#!/bin/bash

# ==========================================
# Nginx 安全安装与启动脚本
# 适用系统: CentOS/RHEL 7+, Ubuntu 18.04+
# ==========================================

# 定义颜色，方便查看日志
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. 检查是否为 Root 用户
if [ "$EUID" -ne 0 ]; then 
  echo -e "${RED}请使用 sudo 或 root 权限运行此脚本。${NC}"
  exit 1
fi

echo -e "${GREEN}=== 开始 Nginx 安装与环境检查 ===${NC}"

# 2. 检查 Nginx 是否已存在
if command -v nginx &> /dev/null; then
    echo -e "${YELLOW}检测到 Nginx 已经安装。${NC}"
    nginx -v
    echo -e "跳过安装步骤，直接检查服务状态..."
else
    # 3. 自动识别系统并安装
    if [ -f /etc/redhat-release ]; then
        echo "检测到系统: CentOS/RHEL"
        echo "正在安装 EPEL 源和 Nginx..."
        yum install -y epel-release
        yum install -y nginx
    elif [ -f /etc/lsb-release ] || [ -f /etc/debian_version ]; then
        echo "检测到系统: Ubuntu/Debian"
        echo "正在更新源并安装 Nginx..."
        apt-get update
        apt-get install -y nginx
    else
        echo -e "${RED}未识别的操作系统，请手动安装 Nginx。${NC}"
        exit 1
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Nginx 安装成功！${NC}"
    else
        echo -e "${RED}Nginx 安装失败，请检查网络或源配置。${NC}"
        exit 1
    fi
fi

# 4. 备份默认配置 (仅在首次备份时执行)
CONF_PATH="/etc/nginx/nginx.conf"
if [ -f "$CONF_PATH" ] && [ ! -f "${CONF_PATH}.original.bak" ]; then
    echo "正在备份默认配置文件到 ${CONF_PATH}.original.bak ..."
    cp "$CONF_PATH" "${CONF_PATH}.original.bak"
fi

# 5. 启动服务并设置开机自启
echo -e "${GREEN}=== 正在启动服务 ===${NC}"

# 先进行语法检查，确保新装的环境没问题
nginx -t
if [ $? -ne 0 ]; then
    echo -e "${RED}配置文件语法错误，无法启动！${NC}"
    exit 1
fi

# 尝试启动或重载
if systemctl is-active --quiet nginx; then
    echo "Nginx 正在运行，执行平滑重载..."
    systemctl reload nginx
else
    echo "Nginx 未运行，正在启动..."
    systemctl start nginx
    systemctl enable nginx # 设置开机自启
fi

# 6. 最终状态检查
if systemctl is-active --quiet nginx; then
    echo -e "${GREEN}✅ Nginx 服务运行正常！${NC}"
    echo -e "监听端口情况："
    netstat -ntlp | grep nginx || ss -ntlp | grep nginx
    echo -e "${YELLOW}提示：如果无法访问，请检查服务器防火墙 (firewalld/ufw) 是否开放了 80/11126/11127 端口。${NC}"
else
    echo -e "${RED}❌ Nginx 启动失败，请使用 'systemctl status nginx' 查看详情。${NC}"
    exit 1
fi