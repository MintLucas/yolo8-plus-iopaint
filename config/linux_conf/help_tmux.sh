#!/bin/bash
# tmux_help.sh
# 作用：输出 Tmux 常用快捷键和命令速查表

# 定义颜色
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
WHITE='\033[1;37m'
NC='\033[0m'

echo -e "${GREEN}=== Tmux AI 工程师速查手册 ===${NC}"
echo -e "核心前缀键 (Prefix): ${YELLOW}Ctrl + b${NC} (按下这两个键后松开，再按后面的键)\n"

echo -e "${CYAN}[1. 会话管理 (最常用)]${NC}"
echo -e "  ${WHITE}tmux new -s <名字>${NC}   : 创建新会话 (例如: tmux new -s train_v1)"
echo -e "  ${WHITE}tmux detach${NC} (或是 ${YELLOW}Prefix + d${NC}) : ${GREEN}临时离开${NC} (程序继续在后台跑)"
echo -e "  ${WHITE}tmux ls${NC}               : 列出所有后台运行的会话"
echo -e "  ${WHITE}tmux attach -t <名字>${NC}: ${GREEN}回到现场${NC} (恢复会话)"
echo -e "  ${WHITE}tmux kill-session -t <名>${NC}: 彻底杀死某个会话"

echo -e "\n${CYAN}[2. 窗格操作 (分屏神器)]${NC}"
echo -e "  ${YELLOW}Prefix + %${NC}      : ${GREEN}左右分屏${NC} (左边跑代码，右边 watch -n 1 nvidia-smi)"
echo -e "  ${YELLOW}Prefix + \"${NC}      : ${GREEN}上下分屏${NC}"
echo -e "  ${YELLOW}Prefix + 方向键${NC} : 在不同窗格间切换光标"
echo -e "  ${YELLOW}Prefix + x${NC}      : 关闭当前窗格"
echo -e "  ${YELLOW}Prefix + z${NC}      : 最大化/还原当前窗格 (看日志时很有用)"

echo -e "\n${CYAN}[3. 窗口操作 (标签页)]${NC}"
echo -e "  ${YELLOW}Prefix + c${NC}      : 新建一个窗口"
echo -e "  ${YELLOW}Prefix + n / p${NC}  : 切换到 下一个/上一个 窗口"
echo -e "  ${YELLOW}Prefix + w${NC}      : 以菜单方式查看并切换窗口"

echo -e "\n${CYAN}[4. 翻页查看历史]${NC}"
echo -e "  ${YELLOW}Prefix + [${NC}      : 进入复制/翻页模式 (然后用方向键或PageUp翻页)"
echo -e "  ${WHITE}q${NC}               : 退出翻页模式"

echo -e "\n${GREEN}提示：如果不习惯用键盘，建议在 ~/.tmux.conf 开启鼠标支持。${NC}"
echo -e "=========================================="