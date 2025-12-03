import requests
import base64

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