"""
核心视频生成工具类
功能：接收API请求中的问题列表，生成图片文案与图片路径
"""
import sys
import os
import random
from typing import List, Dict, Any
sys.path.append(os.getcwd())

import base64
from io import BytesIO
from PIL import Image
from util.mylogging import get_logger

logger = get_logger("server_log/question_mul")

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

from util.token_util_new import token_fresh, models_dict  # 导入模型调用工具类
from util.oss_util import oss_util  # 导入模型调用工具类

ou = oss_util()  # 初始化OSS工具
def base64_to_image(base64_str, output_path):
    # 移除Base64字符串中的前缀（如果有）
    if 'base64,' in base64_str:
        base64_str = base64_str.split('base64,')[1]
    
    try:
        # 解码Base64数据为二进制
        img_data = base64.b64decode(base64_str)
        
        # 将二进制数据转为图像
        img = Image.open(BytesIO(img_data))
        
        # 保存图像
        img.save(output_path)
        logger.info(f"图片已保存至: {output_path}")
    except Exception as e:
        logger.info(f"转换失败: {str(e)}")

import random

def get_random_style():
    style_dict = {
        "ukiyoe": "日式浮世绘（木版画质感，色彩浓郁饱和，线条简洁有力）",
        "ink_style": "水墨国风（以黑白为主，少量色彩点缀，留白充足，意境悠远）",
        "pixel_art": "像素风（画面边缘有轻微像素锯齿，1990年代红白机游戏画面质感）",
        "us_cartoon": "美式卡通（色彩鲜艳饱和，线条粗黑圆润，阴影是纯色块）",
        "low_poly": "低多边形（三角形和四边形拼接，色彩明快干净，无渐变和阴影，简约几何感）",
        "morandi_color": "莫兰迪色系（色彩降低饱和度，带灰色调，画面柔和安静，无强烈光影）",
        "cyberpunk": "赛博朋克（冷色调为主，光影强烈对比）",
        "midjourney_style": "Midjourney风格（超现实构图，8K，HDR效果，细节丰富）",
        "forest_fairy": "森系童话风格（色彩明快柔和，边缘虚化，儿童绘本插画质感，4K）",
        "retro_hongkong": "复古港风（暖黄色胶片颗粒，轻微褪色效果，16:9电影画幅，无文字）",
        "impressionist_monet": "印象派莫奈风格（笔触松散细碎，色彩通透，油画质感，8K）"
    }
    
    # 随机选择键并返回对应值
    random_key = random.choice(list(style_dict.keys()))
    return random_key, f"{style_dict[random_key]}"
# ...（其他导入和前置代码保持不变）

def process_api_questions_generate_pictures(questions: List[Dict[str, Any]], tf, video_model_type, para_dict = {}) -> List[Dict[str, str]]:
    """
    核心处理函数：接收API请求的问题列表，逐一生成图片并返回结果
    新增特性：所有日志均包含receive_id，简化中间过程记录
    """
    receive_id = para_dict.get("receive_id", "unknown_receive_id")
    results = []
    total_questions = len(questions)
    
    if total_questions == 0:
        logger.info(f"receive_id={receive_id} 未从API请求中获取到任何问题数据")
        return results
    
    logger.info(f"receive_id={receive_id} 成功接收{total_questions}个问题，开始生成图片...")
    with open("config/prompt/pic_actor_prompt.md", encoding='utf-8') as f:
        user_base_prompt = f.read()

    for idx, question_item in enumerate(questions, 1):
        # 从API请求数据中提取关键信息（适配之前定义的QuestionItem模型）
        question_id = str(question_item.get("question_id", f"unknown_{idx}"))  # 确保ID不为空
        question_content = question_item.get("question", "")
        choose_list = question_item.get("choose", [])
        answer_identifier = question_item.get("answer", [""])[0]  # 适配answer为列表的格式，取第一个元素

        option_mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
        answer_content = ""
        option_index = option_mapping[answer_identifier]
        answer = choose_list[option_index]
        
        # 提取关键信息（保持原有逻辑）
        question_id = str(question_item.get("question_id", f"unknown_{idx}"))
        question_content = question_item.get("question", "")
        style_brief, style = get_random_style()  # 获取随机风格指令
        # --------------------------
        # 第一步：调用LLM生成视频文案
        # --------------------------
        system_prompt = " "
        # ...（其他原有逻辑保持不变）
        supply_dict = {
            "question_content": question_content,
            "answer": answer,
            "style": style
            
        }
        user_prompt = user_base_prompt.format(**supply_dict)
        # 简化日志：仅记录关键节点
        logger.info(f"receive_id={receive_id} | question_id={question_id} 开始生成图片文案...")
        
        try:
            # 调用LLM生成图片文案（简化日志）
            video_script = tf.call_model_zp(
                user_prompt=user_prompt,
                system_prompt=system_prompt
            )
        except Exception as e:
            logger.error(f"receive_id={receive_id} | question_id={question_id} 文案生成失败: {str(e)[:100]}")
            # 记录错误结果
            continue

        logger.info(f"receive_id={receive_id} | question_id={question_id} 开始调用图片模型...")
        try:
            # 调用图片模型生成图片
            test = tf.call_model_zp_img(
                video_script, 
                "", 
                "volcengine:Doubao-Seedream-4.0-250828", 
                source="116913455"
            )
            
            # 保存图片并获取URL
            picture_save_path = f"tmp_data/tmp_pic_res/{question_content}.png"
            ou.img2path(test, picture_save_path)
            url = ou.path2url(
                local_file_path=picture_save_path,
                object_key=f"ai_img/{style_brief}/"
            )
            
            logger.info(f"receive_id={receive_id} | question_id={question_id} 图片生成成功，保存路径: {picture_save_path}")
            
            # 记录成功结果
            results.append({
                "question_id": question_id,
                "question": question_content,
                "video_path": url,
                "status": "success",
                "error_msg": ""
            })

            
        except Exception as e:
            logger.error(f"receive_id={receive_id} | question_id={question_id} 图片生成失败: {str(e)[:100]}")
            # 记录错误结果
            results.append({
                "question_id": question_id,
                "question": question_content,
                "video_path": "",
                "status": "failed",
                "error_msg": f"图片生成失败: {str(e)[:100]}"
            })

    # 汇总日志
    success_count = len([r for r in results if r['status']=='success'])
    logger.info(f"receive_id={receive_id} 所有问题处理完毕！共{total_questions}个，成功{success_count}个，失败{total_questions-success_count}个")
    
    return results

# ...（其他函数保持不变）

def test_single_question_video(tf, video_model_type):
    """
    单问题测试函数（用于本地调试，非API流程）
    测试单个问题的视频生成逻辑，方便快速验证
    """
    test_question = {
        "question_id": 999,
        "question": "《静夜思》中‘举头望明月’的下一句是什么？",
        "choose": ["低头思故乡", "疑是地上霜", "床前明月光", "故人西辞黄鹤楼"],
        "answer": ["A"]
    }
    logger.info("=== 开始单问题视频生成测试 ===")
    result = process_api_questions_generate_pictures(
        questions=[test_question],
        tf=tf,
        video_model_type=video_model_type
    )
    logger.info(f"\n测试结果：{result}")
    return result


if __name__ == '__main__':
    # 本地调试：初始化工具类，测试单问题生成
    tf = token_fresh()
    video_model = models_dict["video-huoshan"]
    test_single_question_video(tf, video_model)