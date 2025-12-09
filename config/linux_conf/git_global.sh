# 1. 配置身份信息 (完全对应机器一)
git config --global user.email "mintlzp@mail.ustc.edu.cn"
git config --global user.name "MintLucas"

# 2. 配置核心编辑器 (习惯用 vim)
git config --global core.editor "vim"

# 3. 配置 SSL 后端 (保持一致)
# git config --global http.sslbackend "gnutls"

# 4. 配置 Git LFS (大文件存储支持)
git config --global filter.lfs.smudge "git-lfs smudge -- %f"
git config --global filter.lfs.process "git-lfs filter-process"
git config --global filter.lfs.required true
git config --global filter.lfs.clean "git-lfs clean -- %f"

# 5. 配置安全目录信任
# 注意：如果第二台机器没有这个路径，这行配置也不会报错，只是没生效而已
# git config --global --add safe.directory "/workspace/work/duanguokai/StyleShot"

# --- 验证环节 ---
echo "✅ 配置已同步完成，当前配置如下："
git config --global --list