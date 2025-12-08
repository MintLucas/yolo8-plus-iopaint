#!/bin/bash
# check_ports.sh
# 作用：列出所有监听端口、PID，以及该进程完整的启动命令

# 检查是否以 root 运行（否则看不到部分进程的 PID）
if [ "$EUID" -ne 0 ]; then 
  echo "请使用 sudo 运行此脚本以获取完整信息"
  exit 1
fi

echo -e "\n正在扫描端口占用情况..."
printf "%-10s %-10s %-10s %-s\n" "协议" "端口" "PID" "完整启动命令"
echo "--------------------------------------------------------------------------------"

# 使用 ss 命令获取端口信息，相比 netstat 更快且格式更规范
# 遍历 TCP 和 UDP 监听端口
ss -lntuHp | while read -r line; do
    # 提取端口号 (处理 IPv4 和 IPv6 格式)
    port=$(echo "$line" | awk '{print $4}' | awk -F':' '{print $NF}')
    proto=$(echo "$line" | awk '{print $1}')
    
    # 提取 PID (ss 输出格式如 users:(("nginx",pid=1234,fd=6)))
    # 使用 grep 和 awk 提取 pid= 后面的数字
    pid=$(echo "$line" | grep -o 'pid=[0-9]*' | awk -F'=' '{print $2}' | head -n1)

    if [ -n "$pid" ]; then
        # 通过 PID 查询完整的启动命令
        cmd=$(ps -p "$pid" -o args=)
        # 如果命令太长，截取前 100 个字符方便显示
        # cmd=$(echo "$cmd" | cut -c 1-100) 
        printf "%-10s %-10s %-10s %-s\n" "$proto" "$port" "$pid" "$cmd"
    else
        printf "%-10s %-10s %-10s %-s\n" "$proto" "$port" "-" "(权限不足或内核进程)"
    fi
done | sort -n -k2