#!/bin/bash
# -----------------------------------------------------------------------------
# 脚本位置: ~/yolo8-plus-iopaint/config/sup_conf/add_new_reread_update.sh
# 作用: 自动定位环境，并执行 supervisorctl 的 reread 和 update
# -----------------------------------------------------------------------------

# ================= 配置区 =================
ENV_NAME="zhipeng16_cv"
CONF_FILENAME="supervisord.conf"

# ================= 脚本逻辑 =================

# 1. 获取当前脚本所在的绝对目录 (即 config/sup_conf)
# 这样无论你在哪执行，它都知道配置文件在脚本旁边
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CONF_FILE="$SCRIPT_DIR/$CONF_FILENAME"

echo "📂 脚本所在目录: $SCRIPT_DIR"

# 2. 检查配置文件
if [ ! -f "$CONF_FILE" ]; then
    echo "❌ 错误：找不到配置文件 $CONF_FILE"
    exit 1
fi

# 3. 初始化 Conda (为了找路径)
for conda_sh in \
    "$HOME/miniforge3/etc/profile.d/conda.sh" \
    "/workspace/work/jiangtao16/miniforge3/etc/profile.d/conda.sh" \
    "/workspace/work/moniforge3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [ -f "$conda_sh" ]; then
        source "$conda_sh"
        break
    fi
done

if ! command -v conda &> /dev/null; then
    echo "❌ 找不到 conda 命令。"
    exit 1
fi

# 4. 获取环境绝对路径
# echo "🔍 定位环境: $ENV_NAME ..."
ENV_ABS_PATH=$(conda run -n "$ENV_NAME" printenv CONDA_PREFIX)

if [ -z "$ENV_ABS_PATH" ]; then
    echo "❌ 无法获取环境路径。"
    exit 1
fi

# 5. 定位 supervisorctl (直接找二进制文件，不加 python)
SUPERVISOR_CTL="$ENV_ABS_PATH/bin/supervisorctl"

if [ ! -f "$SUPERVISOR_CTL" ]; then
    echo "❌ 找不到: $SUPERVISOR_CTL"
    exit 1
fi

# ================= 执行命令 =================
echo "🚀 正在执行配置重载..."
echo "----------------------------------------"

# 1. Reread: 读取新增配置
echo ">> Executing reread..."
"$SUPERVISOR_CTL" -c "$CONF_FILE" reread

# 2. Update: 应用变更
echo ">> Executing update..."
"$SUPERVISOR_CTL" -c "$CONF_FILE" update

echo "----------------------------------------"
echo "✅ 完成！"
#/workspace/work/jiangtao16/miniforge3/envs/zhipeng16_cv/bin/python /workspace/work/jiangtao16/miniforge3/envs/zhipeng16_cv/bin/supervisorctl -c /workspace/work/zhipeng16/yolo8-plus-iopaint/config/sup_conf/supervisord.conf reread
#/workspace/work/jiangtao16/miniforge3/envs/zhipeng16_cv/bin/python /workspace/work/jiangtao16/miniforge3/envs/zhipeng16_cv/bin/supervisorctl -c /workspace/work/zhipeng16/yolo8-plus-iopaint/config/sup_conf/supervisord.conf update