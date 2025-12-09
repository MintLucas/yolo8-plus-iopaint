#!/bin/bash
# deploy_env.sh
# 作用：自动解压 Conda 环境包并执行路径修正 (conda-unpack)
# 用法：./deploy_env.sh <压缩包路径> <目标安装路径>

#先拷贝过去
#rsync -avz -e 'ssh -p 1100' --progress $1 10.136.234.255:$2

# --- 颜色定义 ---
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. 参数检查
if [ $# -ne 2 ]; then
    echo -e "${RED}错误：参数数量不对。${NC}"
    echo "用法: $0 <压缩包路径> <目标安装目录>"
    echo "示例: $0 ./zhipeng16_cv_packed.tar.gz /data/anaconda3/envs/zhipeng16_cv"
    exit 1
fi

TAR_FILE="$1"
DEST_DIR="$2"

# 获取绝对路径，防止出错
TAR_FILE=$(realpath "$TAR_FILE")
# 注意：DEST_DIR 可能还不存在，所以不能直接用 realpath，但这不影响 mkdir

echo -e "${GREEN}=== 开始部署 Conda 环境 ===${NC}"
echo -e "源文件: $TAR_FILE"
echo -e "目标地: $DEST_DIR"

# 2. 检查源文件是否存在
if [ ! -f "$TAR_FILE" ]; then
    echo -e "${RED}错误：找不到压缩包文件 $TAR_FILE${NC}"
    exit 1
fi

# 3. 创建目标目录
if [ -d "$DEST_DIR" ]; then
    echo -e "${YELLOW}警告：目标目录 $DEST_DIR 已存在。${NC}"
    read -p "是否继续？这可能会覆盖现有文件 (y/n): " confirm
    if [[ $confirm != "y" ]]; then
        echo "操作已取消。"
        exit 0
    fi
else
    echo "正在创建目录..."
    mkdir -p "$DEST_DIR"
fi

# 4. 解压 (带简单的等待提示)
echo -n "正在解压 (这可能需要几分钟)... "
# 使用 tar 解压，如果出错直接退出
tar -xzf "$TAR_FILE" -C "$DEST_DIR"
if [ $? -ne 0 ]; then
    echo -e "\n${RED}解压失败！请检查磁盘空间或文件权限。${NC}"
    exit 1
fi
echo -e "${GREEN}完成！${NC}"

# 5. 关键步骤：修正路径
echo "正在修正环境路径 (conda-unpack)..."

# 检查是否存在 bin/activate，确保解压正确
if [ ! -f "$DEST_DIR/bin/activate" ]; then
    echo -e "${RED}错误：在解压目录中未找到 bin/activate。可能是压缩包损坏或结构不对。${NC}"
    exit 1
fi

# 在子 shell 中执行，以免污染当前 shell 环境
(
    source "$DEST_DIR/bin/activate"
    
    # 检查是否有 conda-unpack 命令
    if ! command -v conda-unpack &> /dev/null; then
        echo -e "${RED}错误：环境中未找到 conda-unpack 命令！${NC}"
        echo "请确认打包时使用了 conda-pack。"
        exit 1
    fi

    conda-unpack
)

if [ $? -eq 0 ]; then
    echo -e "${GREEN}=== ✅ 环境部署成功！ ===${NC}"
    echo -e "激活命令: source $DEST_DIR/bin/activate"
    echo -e "或者 (如果已放入 envs 目录): conda activate $(basename "$DEST_DIR")"
else
    echo -e "${RED}conda-unpack 执行失败，请手动检查。${NC}"
    exit 1
fi