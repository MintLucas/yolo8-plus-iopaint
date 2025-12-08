#!/bin/bash
#核心两句nginx -t; nginx -s reload; 先test 然后status reload

# 1. 定义 Nginx 命令路径
# 如果你的 nginx 是系统安装的 (yum/apt)，通常直接用 nginx 即可
# 如果是编译安装或在特定路径，请修改这里，例如: NGINX_CMD="/usr/local/nginx/sbin/nginx"
NGINX_CMD="nginx"

# 2. 核心步骤一：语法检查 (类似于编译器的 compile check)
# -t 参数表示 test configuration
echo "Checking Nginx configuration syntax..."
$NGINX_CMD -t

# $? 是上一个命令的返回值，0 表示成功
if [ $? -eq 0 ]; then
    # 3. 核心步骤二：平滑重载
    # -s reload 发送信号给主进程，让它重新加载配置，不中断当前正在处理的连接
    echo "Syntax OK. Reloading Nginx..."
    $NGINX_CMD -s reload
    echo "Nginx reloaded successfully!"
else
    # 4. 如果配置有错，报警并退出，千万别 reload
    echo "❌ Configuration test failed! Not reloading."
    exit 1
fi