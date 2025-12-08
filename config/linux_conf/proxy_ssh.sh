# 语法拆解：
# -p 1100             : 指定连接容器的端口（去1100号房间）
# -R 17890:127.0.0.1:1087 : 带着你的代理隧道一起进去
# root@10.78.9.45     : 目标主机地址
#17890随便的端口目标机器可用即可
#ssh -R 17890:127.0.0.1:1087 root@10.78.9.45 -p 1100

# export all_proxy="http://127.0.0.1:17890"
# export http_proxy=$all_proxy
# export https_proxy=$all_proxy
# 在服务器 .bashrc 里只放这些，不要放 export http_proxy=...
alias proxy_on='export all_proxy=http://127.0.0.1:17890; export http_proxy=$all_proxy; export https_proxy=$all_proxy; echo "🚀 代理起飞"'
alias proxy_off='unset all_proxy; unset http_proxy; unset https_proxy; echo "🛑 代理关闭"'
proxy_on
curl -I https://www.google.com

#vscode的配置
#RemoteForward 17890 127.0.0.1:1087