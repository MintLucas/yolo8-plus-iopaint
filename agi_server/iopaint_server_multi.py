#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/09/18 17:05
# @Author  : zhipeng16
# @Email   : zzzp50@ustc.edu
# @File    : iopain_server_multi.py
# @Usage   : Describe the file's purpose
import subprocess
import time
import sys,os
sys.path.append(os.getcwd())
sys.path.append("/workspace/work/zhipeng16/yolo8-plus-iopaint")
from util.mylogging import get_logger
logging = get_logger("iopaint_server_multi")
import subprocess
import threading
import time
from typing import List

# --- 配置区 ---
# 只需提供GPU的索引号即可
#起8个服务占用8个G
DEVICES_INDICES = [0, 0, 2, 2, 2, 2, 2, 2] 
BASE_PORT = 8500
MODEL = "lama"
HOST = "0.0.0.0"


def log_pipe(pipe, port):
    """从子进程的管道中读取输出并记录日志"""
    try:
        for line in iter(pipe.readline, b''):
            log_message = line.decode('utf-8', errors='ignore').strip()
            if log_message:
                logging.info(f"[PORT:{port}] {log_message}")
    except Exception as e:
        logging.error(f"[PORT:{port}] Log pipe reader failed: {e}")
    finally:
        if pipe:
            pipe.close()


def start_iopaint_instance(port: int, device_index: int):
    """
    构建命令并启动一个 iopaint 服务实例
    """
    logging.info(f"准备启动服务 -> Port: {port}, GPU Index: {device_index}, Model: {MODEL}")
    
    # 构建命令
    # --device 和 --interactive-seg-device 现在固定为 'cuda'
    command = [
        "python", "-m", "iopaint", "start",
        f"--model={MODEL}",
        "--device=cuda",
        f"--host={HOST}",
        f"--port={port}",
        "--enable-interactive-seg",
        "--interactive-seg-device=cuda"
    ]
    
    logging.info(f"执行命令: {' '.join(command)}")
    
    try:
        # --- 关键改动：为子进程设置独立的环境变量 ---
        # 复制当前环境
        proc_env = os.environ.copy()
        # 设置CUDA_VISIBLE_DEVICES，让子进程只能看到指定的GPU
        proc_env['CUDA_VISIBLE_DEVICES'] = str(device_index)
        logging.info(f"为 Port:{port} 的服务设置环境变量: CUDA_VISIBLE_DEVICES={device_index}")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=proc_env  # <--- 将修改后的环境变量传入子进程
        )
        
        stdout_thread = threading.Thread(target=log_pipe, args=(process.stdout, port))
        stderr_thread = threading.Thread(target=log_pipe, args=(process.stderr, port))
        
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        
        stdout_thread.start()
        stderr_thread.start()
        
        return process
    except FileNotFoundError:
        logging.error("错误: 'python' 命令未找到。请确保Python已安装并在您的PATH中。")
        return None
    except Exception as e:
        logging.error(f"启动服务失败 (Port: {port}): {e}")
        return None


if __name__ == "__main__":
    processes: List[subprocess.Popen] = []
    
    for i, device_idx in enumerate(DEVICES_INDICES):
        port = BASE_PORT + i
        p = start_iopaint_instance(port, device_idx)
        if p:
            processes.append(p)

    if not processes:
        logging.fatal("没有任何服务成功启动，程序退出。")
        sys.exit(1)

    logging.info(f"成功启动 {len(processes)} 个服务。按 Ctrl+C 终止所有服务。")

    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        logging.warning("接收到中断信号 (Ctrl+C)，正在关闭所有服务...")
        for p in processes:
            logging.info(f"正在终止进程 PID: {p.pid}...")
            p.terminate()
        
        time.sleep(5)

        for p in processes:
            if p.poll() is None:
                logging.warning(f"进程 PID: {p.pid} 未能正常终止，强制关闭 (kill)。")
                p.kill()
        
        logging.info("所有服务已关闭。")
    except Exception as e:
        logging.error(f"主程序发生未知错误: {e}")
    finally:
        logging.info("主程序退出。")