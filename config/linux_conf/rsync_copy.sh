#目标机器快速路径；vi /etc/rsyncd.conf
# [zhipeng16]
# path = /workspace/work/zhipeng16
# read only = no
# host allow = *
# uid = root
# gid = root


#双冒号demon守护进程模式需要对方起服务监听873，单冒号不需要走ssh
rsync -avz -e 'ssh -p 1100' --progress $1 10.136.234.255:$2

#可以修改~/.ssh/config，省去-e 'ssh -p 1100'
# Host l20
#     HostName 10.136.234.255
#     User root
#     Port 1100