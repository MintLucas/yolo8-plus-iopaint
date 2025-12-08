#!/bin/bash
# export_env_config.sh
# 用法: ./export_env_config.sh [环境名]

GREEN='\033[0;32m'
NC='\033[0m'

if [ -z "$1" ]; then
    read -p "请输入要导出的环境名称: " ENV_NAME
else
    ENV_NAME=$1
fi

# 确保 conda 可用
CONDA_BASE="/workspace/work/moniforge3"
[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ] && source "$CONDA_BASE/etc/profile.d/conda.sh"

OUTPUT_FILE="${ENV_NAME}.yaml"

echo "正在导出环境 ${ENV_NAME} 的配置..."

# 使用 --no-builds 可以让导出的文件兼容性更好（不锁定具体构建哈希）
conda env export -n "$ENV_NAME" --no-builds > "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}导出成功！${NC}"
    echo "配置文件已保存为: $(pwd)/$OUTPUT_FILE"
    echo "恢复命令: conda env create -f $OUTPUT_FILE"
else
    echo "导出失败，请检查环境名称是否正确。"
fi