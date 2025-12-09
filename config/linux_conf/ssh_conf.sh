#本机生成密码
#ssh-keygen -t ed25519 -C "zzzp50@ustc.edu"
ssh-keygen -t rsa -b 4096
#拷贝到远程，远程记录本机密码；对号访问
ssh-copy-id -p 1100 root@10.78.9.45

#mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys