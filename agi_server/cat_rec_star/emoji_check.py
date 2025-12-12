import json
import re
import requests

def parse_emoticon_data(raw_data):
    """
    【公共逻辑】解析微博接口返回的原始JSON数据，提取表情包。
    """
    emoticon_map = {}
    try:
        # 安全获取数据层级
        data = raw_data.get('data', {})
        emoticon_data = data.get('emoticon', {})
        
        # 遍历逻辑：语言 -> 分类 -> 表情列表
        for lang, categories in emoticon_data.items():
            if isinstance(categories, dict):
                for category_name, icon_list in categories.items():
                    if isinstance(icon_list, list):
                        for icon in icon_list:
                            phrase = icon.get('phrase')
                            if phrase:
                                emoticon_map[phrase] = phrase
    except Exception as e:
        print(f"解析数据时发生错误: {e}")
        return {}
        
    print(f"解析完成，共加载 {len(emoticon_map)} 个表情。")
    return emoticon_map

def load_emoticon_dict_from_file(file_path):
    """
    从本地文件读取
    """
    print(f"正在从本地文件 {file_path} 加载...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 兼容处理 source 标记
            if content.startswith('[source'):
                content = re.sub(r'^\\s*', '', content)
            raw_data = json.loads(content)
            return parse_emoticon_data(raw_data)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return {}

def load_emoticon_dict_from_url():
    """
    从微博接口 URL 读取 (模拟 curl 请求)
    """
    url = 'https://weibo.com/ajax/statuses/config'
    print(f"正在从接口 {url} 请求数据...")

    # 根据你提供的 curl 命令构造 headers
    # 注意：Cookie 和 X-XSRF-TOKEN 等是具有时效性的，
    # 如果一段时间后失效，需要重新抓包替换这里的 headers。
    headers = {
        'accept': 'application/json, text/plain, */*',
        'accept-language': 'en,zh-CN;q=0.9,zh;q=0.8,ja;q=0.7,zh-TW;q=0.6,it;q=0.5',
        'cache-control': 'no-cache',
        'client-version': 'v2.47.139',
        'pragma': 'no-cache',
        'priority': 'u=1, i',
        'referer': 'https://weibo.com/',
        'sec-ch-ua': '"Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'server-version': 'v2025.12.08.1',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36',
        'x-requested-with': 'XMLHttpRequest',
        'x-xsrf-token': 'DXUDaZt4E8AdHq9kIk22sA8L',
        # 将 curl 中的 -b 参数内容放入 Cookie 头
        'cookie': 'SCF=AvHSimXvAgIUXeLcRQ7FzHp3HZUg43Z3u7-0NhWmSO22e90zOWLjpm1HcjduP3txTNRVGL7g9kzXWmO540J4Zgc.; XSRF-TOKEN=DXUDaZt4E8AdHq9kIk22sA8L; _s_tentry=-; Apache=6547943836029.447.1757404012440; SINAGLOBAL=6547943836029.447.1757404012440; ULV=1757404012486:1:1:1:6547943836029.447.1757404012440:; ai-clipper_admin_v2=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiI2ODc1MzkyMjQwIiwiZW1haWwiOiJmdXFpYW5nNiIsImlhdCI6MTc2NTM2NTU5NiwiZXhwIjoxNzY1NDUxOTk2fQ.JiSna8rOFaKCKC1KIsBSjHZTqMZSJEWVA7_OQTbY44o; ALF=1767958212; SUB=_2A25EPSmUDeRhGeBG7FcS-SzOzzyIHXVnMyNcrDV8PUJbkNAbLUjjkW1NQeyGGxSRh9TPagMbQl0IePFlk5NyMbPa; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9W55FZr6cVveJBuAi4uCb.3-5JpX5KMhUgL.FoqRS0-01KzESh52dJLoI0qLxKnL1h5L1h2LxK-LBKBLBKMLxKML1hnLBoMLxKnLBo.LB-eLxKBLBonL1h.LxKnLB.BLB--t; WBPSESS=RPgNzT17ivmKS2bhfWS019EAiy7zk4I_3Qs6m9fjdlaj388gIX0v5vcpICNKC00Z38ArlLEviPoidHt1JnM52w2dnTzw7W143YvV5kOGFCD0VUdmmrojETODSeo-ixLgADwcUzp5SM-QdydKdA_T-Q==; PC_TOKEN=cd553c7b8d'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # 检查 4xx, 5xx 错误
        return parse_emoticon_data(response.json())
    except requests.exceptions.RequestException as e:
        print(f"网络请求失败: {e}")
        return {}

def filter_text_emoticons(text, emoticon_map):
    """
    优化后的过滤函数：
    能够区分 JSON 结构中的 '[' 和表情包的 '['，
    避免将 JSON 数组或对象误判为表情包。
    """
    if not text:
        return ""

    # 优化点解释：
    # 1. [^ ... ]+ : 匹配括号内除了指定字符以外的内容
    # 2. 排除 \[ \] : 不支持嵌套括号
    # 3. 排除 \{ \} : 排除 JSON 对象开始/结束符，防止匹配到 [{"content":...
    # 4. 排除 "     : 排除双引号，防止匹配到 JSON key/value，如 ["style_type"
    # 5. 排除 \n    : 表情包通常不会跨行
    pattern = r"\[[^\[\]\{\}\"\n]+\]"

    def replace_callback(match):
        captured_phrase = match.group(0)
        
        # 如果在白名单里，直接返回（保留）
        if captured_phrase in emoticon_map:
            return captured_phrase
        else:
            # 即使符合了更严格的正则，但不在白名单里（比如 [未知表情]），则按需求移除
            # 注意：如果文本中原本包含 [1,2,3] 这种非表情的数组，
            # 且没有命中上述排除字符，也会被视为"非法表情"而移除。
            # 这是为了满足"不在dict则去除"的需求。
            return ""

    return re.sub(pattern, replace_callback, text)

def main():
    # 1. 改为使用 URL 加载 (不再从文件加载)
    # valid_emoticons = load_emoticon_dict_from_file('tmp.txt') 
    valid_emoticons = load_emoticon_dict_from_url()
    
    if not valid_emoticons:
        print("未获取到表情数据，程序终止。")
        return

    # 2. 测试字符串
    # [点赞] 在白名单中，[未知表情] 不在
    test_str = "测试网络[爱心]加载功能：这个功能真不错[点赞]！但是[未知表情]会被删掉。[哈哈]"
    test_str = '{\n  "generated_texts": [心] 这样的[额外表情xxx ]好天气，适合出门走走呢～#微博跨域计划#"\n    },\n    {\n      "style_type": "温暖型",\n      "content": "被@小兽睡睡 的好天气治愈到啦～[太阳] 愿这份晴朗也能温暖你的一天～#微博跨域计划#"\n    },\n    {\n      "style_type": "幽默型",\n      "content": "天气好到被@小兽睡睡 承包啦～[哈哈][太开心] 这明媚的阳光，今天也是被治愈的一天～#微博跨域计划#"\n    },\n    {\n      "style_type": "文艺型",\n      "content": "晴光正好，被@小兽睡睡 温柔分享～[心][太阳] 愿这份晴朗，晕染你所有平凡日子～#微博跨域计划#"\n    },\n    {\n      "style_type": "正式型",\n      "content": "@小兽睡睡 今日分享天气晴好，这份温暖值得传递。[微笑] 愿大家享受美好天气。#微博跨域计划#"\n    }\n  ]\n}'
    print("-" * 30)
    print(f"原始字符串: {test_str}")
    
    # 3. 执行过滤
    cleaned_str = filter_text_emoticons(test_str, valid_emoticons)
    
    print("-" * 30)
    print(f"处理后字符串: {cleaned_str}")

if __name__ == '__main__':
    main()