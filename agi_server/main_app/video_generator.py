"""
核心视频生成工具类
功能：接收API请求中的问题列表，生成视频文案与视频路径
"""
import sys
import os
from typing import List, Dict, Any
sys.path.append(os.getcwd())

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

from util.token_util_new import token_fresh, models_dict  # 导入模型调用工具类


def process_api_questions_generate_videos(questions: List[Dict[str, Any]], tf, video_model_type) -> List[Dict[str, str]]:
    """
    核心处理函数：接收API请求的问题列表，逐一生成视频并返回结果
    参数：
        questions：API请求中的问题列表（每个元素含question_id、question、answer等）
        tf：token工具类实例
        video_model_type：视频模型类型（如models_dict["video-huoshan"]）
    返回：
        含question_id、question、video_path、status的结果列表
    """
    results = []
    total_questions = len(questions)
    
    if total_questions == 0:
        print("未从API请求中获取到任何问题数据")
        return results
    
    print(f"成功接收{total_questions}个问题，开始生成视频...")

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


        # 跳过问题内容为空的条目
        if not question_content:
            print(f"第{idx}个问题（ID：{question_id}）内容为空，跳过处理")
            results.append({
                "question_id": question_id,
                "question": "",
                "video_path": "",
                "status": "failed",
                "error_msg": "问题内容为空"
            })
            continue

        print(f"\n==== 正在处理第{idx}/{total_questions}个问题 ====")
        print(f"问题ID：{question_id}")
        print(f"问题内容：{question_content}")
        print(f"答案：{answer}")

        # --------------------------
        # 第一步：调用LLM生成视频文案
        # --------------------------
        system_prompt = "你是专业短视频场景描述师，擅长结合问题和答案，生成符合‘中国古诗词意境’的纯视觉画面文案，能精准适配10秒视频生成需求。"
        user_prompt = (  
            f"现在是答题挑战赛，请根据题目‘{question_content}’及其答案‘{answer}’，生成一段10秒短视频的场景描述文案。要求如下：\n"
            "1. 严格围绕题目和答案核心，构建自然景观、人物动作、光影氛围等具象画面；\n"
            "2. 仅描述视觉可见内容，**不包含字幕、文字，不提及音乐/音效**；\n"
            "3. 语言简洁，适配10秒时长（约2-3个连贯镜头），可直接用于视频生成；\n"
            "4. 突出画面氛围感和动态感，体现与题目相关的核心意象。"
        )

        try:
            print("\n正在生成视频文案...")
            video_script = tf.call_model_zp(user_prompt=user_prompt, system_prompt=system_prompt)
            print(f"视频文案生成成功：\n{video_script}")
        except Exception as e:
            error_msg = f"文案生成失败：{str(e)}"
            print(error_msg)
            results.append({
                "question_id": question_id,
                "question": question_content,
                "video_path": "",
                "status": "failed",
                "error_msg": error_msg
            })
            continue

        # --------------------------
        # 第二步：调用视频模型生成视频（保持原有逻辑）
        # --------------------------
        video_params = "--ratio 16:9 --duration 10"  # 16:9比例、10秒时长（参数可按需调整）
        try:
            print("\n正在调用视频模型生成视频...")
            # 组合文案与参数，调用视频生成接口
            video_task = tf.call_model_zp_video(video_script + " " + video_params, model_type=video_model_type)

            # 获取视频任务ID，查询生成状态
            video_task_id = video_task.get("response_data", {}).get("id", "")
            if not video_task_id:
                error_msg = "未获取到视频生成任务ID"
                print(error_msg)
                results.append({
                    "question_id": question_id,
                    "question": question_content,
                    "video_path": "",
                    "status": "failed",
                    "error_msg": error_msg
                })
                continue

            # 查询视频生成结果，获取保存路径
            video_save_path = tf.query_video_task_status(video_task_id)
            if not video_save_path or not os.path.exists(video_save_path):
                error_msg = f"视频生成成功，但未找到保存路径（任务ID：{video_task_id}）"
                print(error_msg)
                results.append({
                    "question_id": question_id,
                    "question": question_content,
                    "video_path": "",
                    "status": "failed",
                    "error_msg": error_msg
                })
                continue

            # 视频生成成功，记录结果
            print(f"视频生成完成！保存路径：{video_save_path}")
            results.append({
                "question_id": question_id,
                "question": question_content,
                "video_path": video_save_path,
                "status": "success",
                "error_msg": ""
            })

        except Exception as e:
            error_msg = f"视频生成失败：{str(e)}"
            print(error_msg)
            results.append({
                "question_id": question_id,
                "question": question_content,
                "video_path": "",
                "status": "failed",
                "error_msg": error_msg
            })
            continue

    print(f"\n所有问题处理完毕！共{total_questions}个，成功{len([r for r in results if r['status']=='success'])}个，失败{len([r for r in results if r['status']=='failed'])}个")
    return results


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
    print("=== 开始单问题视频生成测试 ===")
    result = process_api_questions_generate_videos(
        questions=[test_question],
        tf=tf,
        video_model_type=video_model_type
    )
    print(f"\n测试结果：{result}")
    return result


if __name__ == '__main__':
    # 本地调试：初始化工具类，测试单问题生成
    tf = token_fresh()
    video_model = models_dict["video-huoshan"]
    test_single_question_video(tf, video_model)