import alibabacloud_oss_v2 as oss
import os,sys
import datetime
import requests,json
sys.path.append(os.getcwd())
from util.mylogging import get_logger
from urllib.parse import urlparse
from typing import Optional


with open("config/agi_config.json", "r") as f:
    json_config = json.load(f)
bucket = json_config["oss_config"]["bucket"]
key_id = json_config["oss_config"]["key_id"]
key_secret = json_config["oss_config"]["key_secret"]
os.environ['OSS_ACCESS_KEY_ID'] = key_id
os.environ['OSS_ACCESS_KEY_SECRET'] = key_secret
class oss_util:
    def __init__(self, log = get_logger("oss_util")):
        self.log = log
        pass
    def get_oss_client(self):
        """
        Python SDK V2 客户端初始化配置说明：
        1. 签名版本：Python SDK V2 默认使用 V4 签名，提供更高的安全性
        2. Region配置：初始化 Client 时，必须指定阿里云 Region ID 作为请求地域标识，例如华东1（杭州）Region ID：cn-hangzhou
        3. Endpoint配置：
        - 可通过Endpoint参数自定义服务请求的访问域名
        - 当不指定 Endpoint 时，将根据 Region 自动构造公网访问域名，例如Region为cn-hangzhou时，构造访问域名为：https://oss-cn-hangzhou.aliyuncs.com
        4. 协议配置：
        - SDK 默认使用 HTTPS 协议构造访问域名
        - 如需使用 HTTP 协议，在指定域名时明确指定：http://oss-cn-hangzhou.aliyuncs.com
        """
        
        # 从环境变量中加载凭证信息，用于身份验证
        credentials_provider = oss.credentials.EnvironmentVariableCredentialsProvider()
        
        # 加载SDK的默认配置，并设置凭证提供者
        cfg = oss.config.load_default()
        cfg.credentials_provider = credentials_provider
        
        # 方式一：只填写Region（推荐）
        # 必须指定Region ID，以华东1（杭州）为例，Region填写为cn-hangzhou，SDK会根据Region自动构造HTTPS访问域名
        cfg.region = 'cn-shanghai' 
        
        # 使用配置好的信息创建OSS客户端
        client = oss.Client(cfg)
        return client
    def upload_file(self, local_file_path = "/data2/zhipeng16/datasets/src_videos/source1_top_right.mp4", object_key = "wces/video.mp4"):

        client = self.get_oss_client()
        # 本地视频文件路径
        local_file_path = local_file_path
        # 存储桶中的对象名称（Object Key）
        object_key = object_key # 替换为你希望在OSS中存储的文件名
        
        # 执行上传对象的请求，指定存储空间名称、对象名称和本地文件路径
        result = client.put_object_from_file(filepath=local_file_path,request=oss.PutObjectRequest(
            bucket=bucket,
            key=object_key
        ))
        return object_key
        # 输出请求的结果状态码、请求ID、ETag，用于检查请求是否成功
        print(f'status code: {result.status_code}\n'
            f'request id: {result.request_id}\n'
            f'etag: {result.etag}')





    def get_url(self, file_name = "video.mp4"):
            # -*- coding: utf-8 -*-
        # 创建生成签名URL的请求
        client = self.get_oss_client()
        pre_result = client.presign(oss.GetObjectRequest(
            bucket=bucket,
            key=file_name,
        ),expires=datetime.timedelta(days=5))
        return pre_result.url
        # 生成签名URL

        with requests.get(pre_result.url) as resp:
            print(f'status code: {resp.status_code},'
                f' request id: {resp.headers.get("x-oss-request-id")},'
                f' hash crc64: {resp.headers.get("x-oss-hash-crc64ecma")},'
                f' content md5: {resp.headers.get("Content-MD5")},'
                f' content : {resp.content},'
                f' server time: {resp.headers.get("x-oss-server-time")},'
                )

        print(f'method: {pre_result.method},'
            f' expiration: {pre_result.expiration.strftime("%Y-%m-%dT%H:%M:%S.000Z")},'
            f' url: {pre_result.url}'
        )

        for key, value in pre_result.signed_headers.items():
            print(f'signed headers key: {key}, signed headers value: {value}')

    def trans_url_shanghai(input_url):

        from viapi.fileutils import FileUtils
        # 创建AccessKey ID和AccessKey Secret，请参考https://help.aliyun.com/document_detail/175144.html。
        # 如果您用的是RAM用户的AccessKey，还需要为RAM用户授予权限AliyunVIAPIFullAccess，请参考https://help.aliyun.com/document_detail/145025.html。
        # 从环境变量读取配置的AccessKey ID和AccessKey Secret。运行代码示例前必须先配置环境变量。
        file_utils = FileUtils(key_id, key_secret)
        # 场景一，使用本地文件，第一个参数为文件路径，第二个参数为生成的url后缀，但是并不能通过这种方式改变真实的文件类型，第三个参数True表示本地文件模式
        # oss_url = file_utils.get_oss_url("/tmp/bankCard.png", "png", True)
        # 场景二，使用任意可访问的url，第一个url，第二个参数为生成的url后缀，但是并不能通过这种方式改变真实的文件类型，第三个参数False表示非本地文件模式
        oss_url = file_utils.get_oss_url(input_url, "mp4", False)
        # 生成的url，可用于调用视觉智能开放平台的能力
        print(oss_url)
        
    def check_video_path(self, path: str):
        """
        检查路径，如果是URL则下载到本地并返回本地路径，否则返回原路径。
        通过文件后缀判断文件类型。

        :param path: 输入的URL或本地路径。
        :return: 本地文件路径，如果下载失败则返回None。
        """
        if not path:
            return None

        # 检查是否为URL
        is_url = urlparse(path).scheme in ['http', 'https']

        if not is_url:
            # 如果是本地路径，直接返回
            if os.path.exists(path):
                return path
            else:
                self.log.error(f"Local path '{path}' does not exist.")
                return None

        # URL处理
        try:
            # 从URL解析文件名和文件后缀
            file_name = os.path.basename(urlparse(path).path)
            if not file_name:
                file_name = "downloaded_file"
            
            file_ext = os.path.splitext(file_name)[1].lower()

            # 根据文件后缀确定下载目录
            tmp_dir = "tmp_data"
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm']
            image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
            
            if file_ext in image_extensions:
                download_dir = os.path.join(tmp_dir, "img")
            elif file_ext in video_extensions:
                download_dir = os.path.join(tmp_dir, "video")
            else:
                # 默认下载到tmp_data
                download_dir = tmp_dir
            
            os.makedirs(download_dir, exist_ok=True)
            local_path = os.path.join(download_dir, file_name)
            if os.path.exists(local_path):
                self.log.info(f"file exsits {local_path}")
                return (local_path, file_name)
            # 下载文件
            self.log.info(f"Downloading from {path} to {local_path}...")
            response = requests.get(path, stream=True)
            response.raise_for_status()

            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.log.info("Download successful!")
            return (local_path, file_name)

        except requests.exceptions.RequestException as e:
            self.log.error(f"Error downloading file from URL: {e}")
            return None
        except Exception as e:
            self.log.error(f"An unexpected error occurred: {e}")
            return None

    def delete_file(self, file_path: str) -> None:
        """
        根据传入路径删除文件。

        :param file_path: 需要删除的本地文件路径。
        """
        if not file_path:
            self.log.error("No file path provided for deletion.")
            return

        try:
            if os.path.exists(file_path) and os.path.isfile(file_path):
                os.remove(file_path)
                self.log.info(f"Successfully deleted file: {file_path}")
            else:
                self.log.info(f"File not found or not a file: {file_path}")
        except OSError as e:
            self.log.error(f"Error deleting file {file_path}: {e}")
            
            
# 当此脚本被直接运行时，调用main函数
if __name__ == "__main__":
    # trans_url_shanghai('https://aiclip.weibo.com/redirect?key=cph%2Fyt_dlp%2F9%2F87979%2F2025-09-11%2Fv_37e5855bc3edf666fa7161bd019ebf9a.mp4')
    local_file_path = "/data2/zhipeng16/datasets/src_videos/source1_top_right.mp4"
    o_u = oss_util()
    file_path = o_u.check_video_path("https://wb-channel-aiclip-media.oss-cn-beijing.aliyuncs.com/cph/yt_dlp/6/51746/2025-09-12/v_187a150723a101760c2d4584172fbe1d.mp4?x-oss-date=20250912T111818Z&x-oss-expires=604800&x-oss-signature-version=OSS4-HMAC-SHA256&x-oss-credential=LTAI5tHj9VxWxHdfk1rWYrdj%2F20250912%2Fcn-beijing%2Foss%2Faliyun_v4_request&x-oss-signature=d177c858922a59d4b05d809130ff5813adba31bcfb515024fa9642bcfb703c57")
    
    
    file_name = o_u.upload_file()
    res_url = o_u.get_url(file_name)  # 脚本入口，当文件被直接运行时调用main函数
    print(res_url)