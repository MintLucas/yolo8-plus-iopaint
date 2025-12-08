#!/bin/bash
# system_monitor_ai.sh
# 作用：AI工程师专用服务器体检 (包含IO延迟、VIRT/RES内存分析、中断监控)

# --- 颜色定义 ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
NC='\033[0m'


# ================= 1. CPU 深度分析 (关注 IO 等待) =================
# 获取 top 输出的 cpu 行
# 典型输出: %Cpu(s): 10.0 us,  2.0 sy,  0.0 ni, 85.0 id,  2.5 wa,  0.0 hi,  0.5 si
cpu_line=$(top -bn1 | grep "Cpu(s)")

# 提取关键指标
cpu_us=$(echo "$cpu_line" | awk -F'us,' '{print $1}' | awk '{print $NF}') # 用户空间 (模型计算)
cpu_sy=$(echo "$cpu_line" | awk -F'sy,' '{print $1}' | awk '{print $NF}') # 内核空间 (系统调用)
cpu_wa=$(echo "$cpu_line" | awk -F'wa,' '{print $1}' | awk '{print $NF}') # IO等待 (硬盘瓶颈)
cpu_id=$(echo "$cpu_line" | awk -F'id,' '{print $1}' | awk '{print $NF}') # 空闲

echo -e "${CYAN}[ CPU 瓶颈诊断 ]${NC}"
echo -n "用户计算(us): ${cpu_us}%  "
echo -n "系统内核(sy): ${cpu_sy}%  "
echo -e "IO等待(wa): ${cpu_wa}%"

# --- AI 工程师解读逻辑 ---
if (( $(echo "$cpu_wa > 10.0" | bc -l) )); then
    echo -e "⚠️  ${RED}诊断: IO等待过高 ($cpu_wa%)！${NC}"
    echo -e "   -> 原因: 硬盘读写太慢，CPU 在等数据，GPU 可能在空转。"
    echo -e "   -> 建议: 检查 DataLoader num_workers，或数据集是否在机械硬盘上，考虑上 SSD/内存盘。"
elif (( $(echo "$cpu_sy > 15.0" | bc -l) )); then
    echo -e "⚠️  ${YELLOW}诊断: 系统内核占用过高 ($cpu_sy%)！${NC}"
    echo -e "   -> 原因: 可能是大量小文件读写、网络中断过多(多机训练)或驱动异常。"
else
    echo -e "✅  ${GREEN}诊断: CPU 状态健康，主要用于计算。${NC}"
fi

echo -e "---------------------------------------------------------"

# ================= 2. 内存深度分析 (VIRT vs RES) =================
mem_total_mb=$(free -m | grep Mem | awk '{print $2}')
mem_used_mb=$(free -m | grep Mem | awk '{print $3}')
mem_percent=$(awk "BEGIN {printf \"%.2f\", ($mem_used_mb/$mem_total_mb)*100}")

echo -e "${CYAN}[ 内存使用概览 ]${NC}"
echo -e "物理内存: ${mem_used_mb}MB / ${mem_total_mb}MB (${mem_percent}%)"

if (( $(echo "$mem_percent > 90" | bc -l) )); then
    echo -e "⚠️  ${RED}警报: 内存极度危险，谨防 OOM Killer 杀掉训练进程！${NC}"
fi

echo -e "---------------------------------------------------------"

# ================= 3. 进程详细追踪 (VIRT/RES 分离) =================
echo -e "${CYAN}[ Top 5 内存大户 (区分 '申请' 与 '实占') ]${NC}"
# 格式说明: 
# VIRT (Virtual Image): 进程"申请"的显存/内存总量 (画的大饼)
# RES  (Resident): 进程"实际"吃掉的物理内存 (吃进肚的)
# 这里的单位我们统一转换为 GB 显示方便看大模型

printf "%-8s %-10s %-10s %-10s %-s\n" "PID" "用户" "申请(VIRT)" "实占(RES)" "命令(前60字)"

# ps 获取数据: pid, user, vsz(kb), rss(kb), command
ps -eo pid,user,vsz,rss,command --sort=-rss | head -n 6 | tail -n 5 | while read pid user vsz rss cmd; do
    # 转换 KB -> GB
    vsz_gb=$(awk "BEGIN {printf \"%.2fG\", $vsz/1048576}")
    rss_gb=$(awk "BEGIN {printf \"%.2fG\", $rss/1048576}")
    
    # 截断长命令
    short_cmd=$(echo $cmd | cut -c 1-150)
    
    printf "%-8s %-10s %-10s %-10s %-s\n" "$pid" "$user" "$vsz_gb" "$rss_gb" "$short_cmd"
done

echo -e "\n${BLUE}💡 内存解读提示:${NC}"
echo -e "1. 如果 ${GREEN}VIRT${NC} 很大但 ${GREEN}RES${NC} 很小: 正常。说明申请了很大 Tensor 但还没塞数据 (Lazy Allocation)。"
echo -e "2. 如果 ${GREEN}RES${NC} 持续上涨不掉: 警惕 Memory Leak (如 list 存 Tensor 未 detach)。"

echo -e "---------------------------------------------------------"

# ================= 4. Top 5 CPU 狂魔 =================
echo -e "${CYAN}[ Top 5 CPU 算力消耗进程 ]${NC}"
printf "%-8s %-10s %-8s %-s\n" "PID" "用户" "%CPU" "命令"

ps -eo pid,user,%cpu,command --sort=-%cpu | head -n 6 | tail -n 5 | while read pid user cpu cmd; do
    short_cmd=$(echo $cmd | cut -c 1-150)
    printf "%-8s %-10s %-8s %-s\n" "$pid" "$user" "$cpu" "$short_cmd"
done
echo -e "${BLUE}💡 CPU解读提示:${NC} 对于多线程 DataLoader，%CPU 超过 100% 是正常的 (如 1200% 代表占满12核)。"

echo -e "\n==================================="