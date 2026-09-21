import requests
import base64
import ipaddress
import socket
from urllib.parse import urlparse
 
def is_safe_url(url, allow_private=False):
    """
    校验URL是否安全，防止SSRF攻击。
    - 禁止非HTTP/HTTPS协议（如file://、gopher://、dict://等）
    - 默认禁止访问内网/私有IP地址（10.x、172.16-31.x、192.168.x、127.x、169.254.x等）
    - 对域名做DNS解析后再次校验IP，防止DNS rebinding
    
    :param url: 待校验的URL字符串
    :param allow_private: 是否允许访问内网地址（本地调试时可设为True）
    :return: (bool, str) 第一个元素为是否安全，第二个元素为原因说明
    """
    try:
        parsed = urlparse(url)
    except Exception as e:
        return False, f"URL解析失败: {e}"

    # 1. 协议白名单校验：只允许 http 和 https
    if parsed.scheme not in ("http", "https"):
        return False, f"不允许的协议: {parsed.scheme}（仅允许http/https）"

    # 2. 本地文件路径（如 /etc/hosts）直接拒绝
    if not parsed.netloc:
        return False, "无效的URL：缺少域名或主机部分"

    hostname = parsed.hostname
    if not hostname:
        return False, "无效的URL：无法解析主机名"

    # 3. 如果是IP地址格式，直接校验
    try:
        ip_obj = ipaddress.ip_address(hostname)
        if not allow_private and (ip_obj.is_private or ip_obj.is_loopback or
                                   ip_obj.is_link_local or ip_obj.is_reserved or
                                   ip_obj.is_multicast or ip_obj.is_unspecified):
            return False, f"禁止访问内网/保留地址: {hostname}"
    except ValueError:
        # 不是IP格式，是域名，继续做DNS解析校验
        pass

    # 4. 对域名做DNS解析，校验解析出的IP是否为内网地址（防止DNS rebinding）
    if not allow_private:
        try:
            addr_infos = socket.getaddrinfo(hostname, None)
            for addr_info in addr_infos:
                ip_str = addr_info[4][0]
                try:
                    ip_obj = ipaddress.ip_address(ip_str)
                    if ip_obj.is_private or ip_obj.is_loopback or \
                       ip_obj.is_link_local or ip_obj.is_reserved or \
                       ip_obj.is_multicast or ip_obj.is_unspecified:
                        return False, f"域名 {hostname} 解析到内网/保留地址: {ip_str}"
                except ValueError:
                    continue
        except socket.gaierror:
            return False, f"域名 {hostname} DNS解析失败"

    return True, "URL安全校验通过"


def url_to_base64(image_url):
    """
    将图片URL转换为Base64格式字符串
    :param image_url: 图片的在线URL（str）
    :return: 图片的Base64编码字符串（str），失败时返回None
    """
    try:
        # 1. 发送GET请求获取图片二进制数据
        # 设置超时避免无限等待，添加User-Agent模拟浏览器请求
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        }
        response = requests.get(image_url, headers=headers, timeout=10)
        # 检查请求是否成功（状态码200表示成功）
        response.raise_for_status()
        
        # 2. 将二进制数据编码为Base64字符串
        # b64encode返回bytes类型，需解码为str
        base64_str = base64.b64encode(response.content).decode("utf-8")
        
        # 3. 返回包含图片格式的完整Base64字符串（可直接用于HTML/img标签）
        # 从响应头获取图片MIME类型（如image/jpeg、image/png）
        content_type = response.headers.get("Content-Type", "image/unknown")
        return f"{base64_str}"
    
    except requests.exceptions.RequestException as e:
        # 捕获请求相关异常（超时、网络错误、404/500等状态码）
        print(f"请求图片失败：{str(e)}")
        return None
    except Exception as e:
        # 捕获其他未知异常
        print(f"编码Base64失败：{str(e)}")
        return None


# 示例：调用函数
if __name__ == "__main__":
    test_url = "http://bj.service.t.sinaimg.cn/orj480/683571b5ly1i7wo13b8vbj21rx2d8b2a.jpg"  # 替换为实际图片URL
    result = url_to_base64(test_url)
    if result:
        print("Base64编码结果（前50字符）：", result[:50])  # 打印前50字符避免输出过长
    else:
        print("转换失败")