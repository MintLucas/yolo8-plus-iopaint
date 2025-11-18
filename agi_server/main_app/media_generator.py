#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/11/06 15:16
# @Author  : zhipeng16
# @Email   : zzzp50@ustc.edu
# @File    : media_generator.py
# @Usage   : Describe the file's purpose
"""
多媒体生成工具类（集成图片/视频生成）
功能：接收API请求中的问题列表，根据类型生成对应媒体文案与资源路径
"""
import sys
import os
import random
from typing import List, Dict, Any
sys.path.append(os.getcwd())

from util.mylogging import get_logger
from util.oss_util import oss_util  # OSS工具类
from util.token_util_new import token_fresh, models_dict  # 模型调用工具类
from agi_server.main_app.server_help import Server_Help
from typing import Dict, Tuple
# 初始化工具与日志
ou = oss_util()
server_help = Server_Help()
logger = server_help.log
# logger = get_logger("server_log/media_generator")

# 媒体类型常量
MEDIA_TYPE_IMAGE = "image"
MEDIA_TYPE_VIDEO = "video"
SOURCE = "116913455"
VIDEO_PARAMS = " --ratio 16:9 --duration 10 --wm true --rs 480p --fps 15"
def get_random_style(media_type: str) -> tuple:
    """
    获取随机媒体风格（区分图片/视频）
    :param media_type: 媒体类型（image/video）
    :return: (风格标识, 风格描述)
    """
    
    UNIFIED_AESTHETIC_STYLES = {
    
    # 1. [新增] 低保真/简笔画 (Low-Fidelity) - [VLM最稳定]
    "stick_figure_classic": "经典简笔画（纯白色背景，所有角色和物体均由简单的黑色火柴人线条构成，无色彩、无阴影、无体积感，画风潦草、简单）",
    "chalkboard_drawing": "粉笔画（模拟黑色或深绿色黑板的粗糙背景，所有线条均由白色或彩色的粉笔绘制，线条边缘有粉末感和不均匀的粗细）",
    "blueprint_schematic": "蓝图示意图（深蓝色背景，所有线条均由明亮的白色或青色细线构成，物体呈透明线框结构，带有网格背景）",
    "whiteboard_marker": "白板标记笔（纯白色、有轻微反光的白板背景，线条由粗黑或蓝色马克笔绘制，笔触有明显的圆头痕迹）",
    # 2. 绘画/艺术媒介 (Painting/Art Mediums)
    "ukiyoe": "日式浮世绘（木版画质感，色彩浓郁饱和，线条简洁有力，无阴影）",
    "ink_style": "水墨国风（以黑白为主，少量色彩点缀，留白充足，意境悠远，毛笔笔触）",
    "impressionist_monet": "印象派莫奈风格（笔触松散细碎，色彩通透，光影斑驳，油画质感）",
    "watercolor_ghibli": "水彩绘本（吉卜力风格，背景使用柔和的水彩笔触和晕染效果，色彩通透、治愈，角色线条简约，手绘感）", # [新增]
    "forest_fairy": "森系童话风格（色彩明快柔和，边缘虚化，儿童绘本插画质感）", 
    # 3. 动画风格 (Animation Styles)
    "cel_animation": "2D赛璐璐动画（清晰的黑色轮廓线，色彩平铺，阴影是硬色块，如80年代日本动画）",
    "us_cartoon": "美式卡通（色彩鲜艳饱和，线条粗黑圆润，无复杂光影）",
    "modern_anime": "现代日本动画（角色特征明显，线条干净利落，色彩鲜艳且对比度高，背景精致，插画感）", # [新增]
    "3d_pixar": "3D动画风格（CG渲染，模型表面光滑，有体积感，皮克斯风格，光影和景深模拟真实）",
    "claymation_3d": "3D粘土定格（CG模拟粘土材质，表面有指纹纹理，色彩鲜艳，有轻微的帧间跳动感以模拟定格动画）", # [新增]
    "papercraft_2.5d": "2.5D剪纸定格（所有物体由模拟的彩色卡纸剪裁而成，边缘清晰，画面有明显的图层感和轻微投影）", # [新增]
    "motion_graphics": "动态图形（扁平化设计，使用简洁的线条、抽象的几何形状和图标，色彩明快对比强烈，信息图表风格）", # [新增]
}
    style_maps = {
        MEDIA_TYPE_IMAGE: {
            "ukiyoe": "日式浮世绘（木版画质感，色彩浓郁饱和，线条简洁有力）",
            "ink_style": "水墨国风（以黑白为主，少量色彩点缀，留白充足，意境悠远）",
            "pixel_art": "像素风（画面边缘有轻微像素锯齿，1990年代红白机游戏画面质感）",
            "us_cartoon": "美式卡通（色彩鲜艳饱和，线条粗黑圆润，阴影是纯色块）",
            "low_poly": "低多边形（三角形和四边形拼接，色彩明快干净，无渐变和阴影，简约几何感）",
            # "morandi_color": "莫兰迪色系（色彩降低饱和度，带灰色调，画面柔和安静，无强烈光影）",
            # "cyberpunk": "赛博朋克（冷色调为主，光影强烈对比）",
            "midjourney_style": "Midjourney风格（超现实构图，1K，HDR效果，细节丰富）",
            "forest_fairy": "森系童话风格（色彩明快柔和，边缘虚化，儿童绘本插画质感，1K）",
            "retro_hongkong": "复古港风（暖黄色胶片颗粒，轻微褪色效果，16:9电影画幅，无文字）",
            "impressionist_monet": "印象派莫奈风格（笔触松散细碎，色彩通透，油画质感，1K）"
        },
        MEDIA_TYPE_VIDEO: {
# 1. 绘画/艺术媒介 (Painting/Art Mediums)
    "ukiyoe": "日式浮世绘（木版画质感，色彩浓郁饱和，线条简洁有力，无阴影）",
    "ink_style": "水墨国风（以黑白为主，少量色彩点缀，留白充足，意境悠远，毛笔笔触）",
    "impressionist_monet": "印象派莫奈风格（笔触松散细碎，色彩通透，光影斑驳，油画质感）",
    "forest_fairy": "森系童话风格（色彩明快柔和，边缘虚化，儿童绘本插画质感）", 
    # ^ 删除了 "4K"，因为“绘本插画”本身已经定义了风格。

    # 2. 电脑/数字风格 (Digital/Computer Styles)
    "pixel_art": "像素风（画面边缘有明显像素锯齿，色彩表有限，1990年代红白机游戏画面质感）",
    "low_poly": "低多边形（所有物体均由可见的三角形和四边形拼接，色彩明快干净，无渐变，简约几何感）",
    
    # 3. 动画风格 (Animation Styles)
    "cel_animation": "2D赛璐璐动画（清晰的黑色轮廓线，色彩平铺，阴影是硬色块，如80年代日本动画）",
    "animation": "c）",
    "us_cartoon": "美式卡通（色彩鲜艳饱和，线条粗黑圆润，无复杂光影）",
    "3d_pixar": "3D动画风格（CG渲染，模型表面光滑，有体积感，皮克斯风格，光影和景深模拟真实）",

    # 4. 模拟/胶片风格 (Analog/Film Styles)
    # [这些是“安全”风格，因为“颗粒”和“模糊”会主动“隐藏”VLM的瑕疵]
    "vintage_film": "复古胶片风格（画面有明显颗粒感，色彩饱和度偏低，轻微褪色，模拟8mm胶片效果）",
    "retro_hongkong": "复古港风（整体色调偏暖黄，有胶片颗粒感，轻微褪色效果，画面对比度略高）",

    # 5. [高风险] 真实/细节风格 (Realistic/Detailed Styles)
    # [我把 Midjourney 风格单独拿出来并做了“安全化”处理]
    "surreal_detailed": "超现实主义风格（戏剧性构图，光影对比强烈，细节丰富，纹理清晰）",
    # ^ 删除了 "Midjourney" (品牌词), "8K", "HDR" (技术词)。
        }
    }
    style_dict = style_maps[media_type]
    random_key = random.choice(list(style_dict.keys()))
    return random_key, style_dict[random_key]
def get_random_Q(directory_path : str = "agi_server/main_app/q_x" , filename_or_extension : str = ".png") -> Tuple[str, str]:
    """
    获取指定路径下的随机文件或指定文件。
    """
    try:
        # --- 步骤 1: 优先尝试作为【完整文件名】处理 ---
        full_path_for_file = os.path.join(directory_path, filename_or_extension)
        if os.path.isfile(full_path_for_file):
            return filename_or_extension, full_path_for_file
        extension = filename_or_extension
        if not extension.startswith('.'):
            # 如果传入的不是完整文件名，则视为后缀，并加上点号
            extension = '.' + extension
        file_map: Dict[str, str] = {}
        # 2.1 遍历目录，创建文件字典
        for filename in os.listdir(directory_path):
            current_full_path = os.path.join(directory_path, filename)

            # 检查是否为文件且后缀匹配（忽略大小写）
            if os.path.isfile(current_full_path) and filename.lower().endswith(extension.lower()):
                # key是文件名，value是完整路径
                file_map[filename] = current_full_path
        # 2.2 随机选取对应的 key 和 value
        if not file_map:
            return '', ''

        # 随机选择一个 key
        random_filename = random.choice(list(file_map.keys()))
        # 获取对应的 value（完整路径）
        random_full_path = file_map[random_filename]
        return random_filename, random_full_path

    except FileNotFoundError:
        print(f"❌ 错误：路径 '{directory_path}' 不存在。")
        return '', ''
    except Exception as e:
        print(f"❌ 处理文件时发生错误：{e}")
        return '', ''

def load_prompt_template(media_type: str) -> str:
    """加载对应媒体类型的提示词模板"""
    template_paths = {
        MEDIA_TYPE_IMAGE: "config/prompt/image_actor_prompt_v3.md",
        MEDIA_TYPE_VIDEO: "config/prompt/video_actor_prompt_v4.md"
    }
    try:
        with open(template_paths[media_type], encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"加载{media_type}提示词模板失败: {str(e)}")
        #  fallback默认模板
        return "根据问题{question_content}和答案{answer}，生成符合{style}的{media_type}描述"


def process_api_questions_generate_media(
    questions: List[Dict[str, Any]],
    tf,
    media_model_type,
    media_type: str = MEDIA_TYPE_IMAGE,
    para_dict: dict = {}
) -> List[Dict[str, str]]:
    """
    核心处理函数：生成图片或视频
    :param questions: 问题列表
    :param tf: token工具实例
    :param media_model_type: 媒体模型类型（如models_dict["image"]或["video-huoshan"]）
    :param media_type: 媒体类型（image/video）
    :param para_dict: 额外参数（含receive_id等）
    :return: 生成结果列表
    """
    receive_id = para_dict.get("task_id", "unknown_receive_id")
    results = []
    total = len(questions)

    if total == 0:
        logger.info(f"receive_id={receive_id} 未获取到问题数据")
        return results

    logger.info(f"receive_id={receive_id} 开始处理{total}个{media_type}生成任务")
    base_prompt = load_prompt_template(media_type)

    for idx, item in enumerate(questions, 1):
        question_id = str(item.get("question_id", f"unknown_{idx}"))
        question_content = item.get("question", "")
        choose_list = item.get("choose", [])
        answer_identifier = item.get("answer", [""])[0]

        # 解析答案
        try:
            option_index = {"A": 0, "B": 1, "C": 2, "D": 3}[answer_identifier]
            answer = choose_list[option_index]
            single_file_name = "_".join([question_content, answer])
        except (KeyError, IndexError):
            error = "答案格式错误或选项不存在"
            logger.error(f"receive_id={receive_id} | {question_id} {error}")
            results.append(build_result(question_id, question_content, "", "failed", error))
            continue
       
        # 跳过空问题
        if not question_content:
            error = "问题内容为空"
            logger.warning(f"receive_id={receive_id} | {question_id} {error}")
            results.append(build_result(question_id, question_content, "", "failed", error))
            continue

        # 生成媒体内容
        try:
            # 1. 生成媒体描述文案
            style_key, style_desc = get_random_style(media_type)
            if para_dict.get("style_type", ""):
                style_key, style_desc = para_dict["style_type"].split("#")
            if para_dict.get("with_pic", ""):
                style_key, style_desc = "q_version_chibi", "动画风格（画面流畅，色彩明快，角色动作夸张）"
            
                
            supply_dict = {
                "question_content": question_content,
                "answer": answer,
                "style": style_desc,
                "media_type": media_type
            }
            if para_dict.get("prompt1", ""):
                base_prompt = para_dict["prompt1"]
                
            user_prompt = base_prompt.format(** supply_dict)
            system_prompt = f"你是专业{media_type}场景描述师，擅长结合问题和答案生成符合{style_desc}的视觉文案"
            system_prompt = ""
            logger.info(f"receive_id={receive_id} | question_id={question_id} 生成{media_type}文案...")

                
            if para_dict.get("prompt2", ""):
                media_script = para_dict["prompt2"]
            else:
                media_script = tf.call_model_zp(user_prompt=user_prompt, system_prompt=system_prompt, source=SOURCE)

            # 2. 生成媒体文件
            file_pre_dir = f"tmp_data/ai_{media_type}/{style_key}/"
            os.makedirs(file_pre_dir, exist_ok=True)
            media_path = generate_media_file(
                tf=tf,
                media_script=media_script,
                model_type=media_model_type,
                media_type=media_type,
                single_file_name = file_pre_dir+single_file_name,
                para_dict = para_dict
            )
            if media_type == MEDIA_TYPE_IMAGE:
                picture_save_path = f"{file_pre_dir}{single_file_name}.png"
                media_path = ou.img2path(media_path, picture_save_path, size = ())
            # 3. 上传OSS并返回URL
            oss_path = file_pre_dir
            media_url = ou.path2url(local_file_path=media_path, object_key=oss_path, save_local=True)
            logger.info(f"receive_id={receive_id} | {question_id} {media_type}生成成功")
            results.append(build_result(question_id, question_content, media_url, "success", "", media_script))

        except Exception as e:
            error = f"{media_type}生成失败: {str(e)[:100]}"
            logger.error(f"receive_id={receive_id} | {question_id} {error}")
            results.append(build_result(question_id, question_content, "", "failed", error, media_script))
            continue

    # 汇总结果
    success = len([r for r in results if r["status"] == "success"])
    logger.info(f"receive_id={receive_id} 处理完成！总{total}个，成功{success}个，失败{total - success}个")
    return results


def generate_media_file(tf, media_script, model_type, media_type, single_file_name = "defaut", para_dict = {}) -> str:
    """生成媒体文件（图片/视频）并返回本地路径"""
    if media_type == MEDIA_TYPE_IMAGE:
        # 图片生成逻辑
        image_task =  tf.call_model_zp_img(
                media_script, 
                "", 
                model_type, 
                source=SOURCE
            )
        return image_task

    elif media_type == MEDIA_TYPE_VIDEO:
        # 视频生成逻辑（带参数）
        with_pic = para_dict.get("with_pic", [])
        if with_pic:
            img_name, img_path = get_random_Q()
            with_pic = tf.img_to_model_ext([img_path], type="base64")
        video_task = tf.call_model_zp_video(f"{media_script} {VIDEO_PARAMS}", model_type=model_type, source=SOURCE, imgs=with_pic)
        task_id = video_task.get("response_data", {}).get("id", "")
        if not task_id:
            raise ValueError("未获取到视频任务ID")
        video_path = tf.query_video_task_status(task_id, model_type = model_type, single_file_name=single_file_name)
        if not video_path or not os.path.exists(video_path):
            raise FileNotFoundError(f"视频路径不存在（任务ID：{task_id}）")
        return video_path

    else:
        raise ValueError(f"不支持的媒体类型：{media_type}")


def build_result(question_id: str, question: str, path: str, status: str, error: str, media_script: str) -> Dict[str, str]:
    """统一结果格式构造函数"""
    return {
        "question_id": question_id,
        "question": question,
        f"media_path": path,  # 动态键名（image_path/video_path）
        "status": status,
        "error_msg": error,
        "media_script": media_script
    }


def test_media_generator(tf, media_type: str = MEDIA_TYPE_IMAGE):
    """测试函数"""
    test_question = {
        "question_id": 999,
        "question": "《静夜思》中‘举头望明月’的下一句是什么？",
        "choose": ["低头思故乡", "疑是地上霜", "床前明月光", "故人西辞黄鹤楼"],
        "answer": ["A"]
    }
    model_map = {
        MEDIA_TYPE_IMAGE: models_dict["image"],
        MEDIA_TYPE_VIDEO: models_dict["video-huoshan"]
    }
    logger.info(f"=== 开始{media_type}生成测试 ===")
    result = process_api_questions_generate_media(
        questions=[test_question],
        tf=tf,
        media_model_type=model_map[media_type],
        media_type=media_type,
        para_dict={"receive_id": f"test_{media_type}_123"}
    )
    logger.info(f"测试结果：{result}")
    return result


if __name__ == '__main__':
    res = get_random_Q("agi_server/main_app/q_x", "q_weibo_zo.png")
    tf = token_fresh()
    # 测试图片生成
    test_media_generator(tf, MEDIA_TYPE_IMAGE)
    # 测试视频生成
    test_media_generator(tf, MEDIA_TYPE_VIDEO)