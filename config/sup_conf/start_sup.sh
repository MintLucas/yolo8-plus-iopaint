#!/bin/bash
# start_sup.sh (优化版)

# ================= 配置区 =================
ENV_NAME="zhipeng16_cv"
CONF_FILENAME="supervisord.conf"

# ================= 1. 动态获取路径 =================

# 获取脚本所在的目录 (config/sup_conf)
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CONF_FILE="$SCRIPT_DIR/$CONF_FILENAME"

PROJ_ROOT=$(cd "$SCRIPT_DIR/../../" && pwd)

echo "📂 项目根目录 (PROJ_ROOT): $PROJ_ROOT"

# ================= 2. 获取 Conda 环境路径 =================

# ... (这里保留你之前的找 conda 的代码) ...
# 为了节省篇幅，假设已经找到了 conda 命令
# source .../conda.sh

echo "🔍 获取环境路径..."
CONDA_ROOT=$(conda run -n "$ENV_NAME" printenv CONDA_PREFIX)

if [ -z "$CONDA_ROOT" ]; then
    echo "❌ 无法获取 Conda 路径"
    exit 1
fi
echo "✅ Conda 环境路径 (CONDA_ROOT): $CONDA_ROOT"

# ================= 3. 【核心】导出变量并启动 =================

# 这一步最关键！Supervisor 读取的是当前 Shell 的环境变量
export PROJ_ROOT="$PROJ_ROOT"
export CONDA_ROOT="$CONDA_ROOT"

# 确保日志目录存在 (否则 supervisor 会启动失败)
if [ ! -d "$PROJ_ROOT/log" ]; then
    echo "创建日志目录: $PROJ_ROOT/log"
    mkdir -p "$PROJ_ROOT/log"
fi
if [ ! -d "$PROJ_ROOT/log/sup_log" ]; then
    echo "创建日志目录: $PROJ_ROOT/log/sup_log"
    mkdir -p "$PROJ_ROOT/log/sup_log"
fi

# 拼接 Python 和 Supervisord 的路径
PYTHON_BIN="$CONDA_ROOT/bin/python"
SUPERVISOR_BIN="$CONDA_ROOT/bin/supervisord"

echo "🚀 $PYTHON_BIN $SUPERVISOR_BIN -c $CONF_FILE"
"$PYTHON_BIN" "$SUPERVISOR_BIN" -c "$CONF_FILE"