#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/09/23 10:56
# @Author  : zhipeng16
# @Email   : zzzp50@ustc.edu
# @File    : poem_video.py
# @Usage   : Describe the file's purpose

import sys,os
sys.path.append(os.getcwd())
from util.token_util_new import token_fresh, models_dict

tf = token_fresh()
def test_video(tf = token_fresh(), prompt = "落霞与孤鹜齐飞，秋水共长天一色",  modelType = models_dict["video-huoshan"]):

    video_id = tf.call_model_zp_video(prompt,model_type=modelType)
    # video_id = {}
    save_path = tf.query_video_task_status(video_id.get("response_data", {}).get("id", "cgt-20250724193620-njqsn"))
    print(save_path)

if __name__ == '__main__':

    test_list = [
    "生成一段展现王之涣《登鹳雀楼》的视频：画面从黄河奔腾的壮阔远景开始，夕阳余晖洒在水面，渐推至鹳雀楼顶层，诗人凭栏远眺，镜头随其目光向上延伸，最后定格在‘欲穷千里目，更上一层楼’的诗句字幕，背景音乐用雄浑的古筝曲。",
    "生成一段演绎孟浩然《春晓》的视频：开篇是雨后清晨的庭院，露珠从嫩绿的枝叶滴落，几只小鸟在枝头跳跃鸣唱，镜头扫过零落的花瓣，切换到窗内诗人惺忪醒来的侧影，结尾浮现‘夜来风雨声，花落知多少’的诗句，搭配轻柔的鸟鸣与钢琴声。",
    "生成一段诠释李白《静夜思》的视频：以皎洁月光洒满房间的特写切入，镜头缓缓移到床前，诗人独坐榻边，抬头望向窗外高悬的明月，画面叠化出故乡庭院的虚影，最后以‘举头望明月，低头思故乡’的诗句收尾，背景是悠扬的二胡独奏。",
    "生成一段呈现柳宗元《江雪》的视频：全景展示白茫茫的江面，大雪纷飞中一叶孤舟漂荡，镜头拉近，可见披蓑戴笠的渔翁独自垂钓，四周山峦覆盖积雪，万籁俱寂，结尾字幕‘孤舟蓑笠翁，独钓寒江雪’，配以萧瑟的萧声。",
    "生成一段表现贺知章《咏柳》的视频：初春的微风中，万条垂下的柳丝轻摆，镜头特写嫩绿的柳叶与鹅黄色的柳芽，几只蝴蝶在柳枝间穿梭，远处有孩童追逐嬉戏，最后出现‘不知细叶谁裁出，二月春风似剪刀’的诗句，背景音乐用轻快的笛子曲。"
]
    test_list = [
    "《静夜思》：月光洒在床前，地上仿佛结了一层霜，一位旅人低头思念着家乡，然后抬头望向明亮的月亮。",
    "《春晓》：春天的早晨，阳光穿过窗户，屋里的人被鸟儿的啼叫声吵醒，窗外一片生机盎然的景象。",
    "《登鹳雀楼》：夕阳西下，一位诗人站在高楼上，眼前是黄河奔流不息的壮丽景色，远处的山脉被落日余晖染成金色，展现了诗人广阔的胸怀。",
    "《悯农》：烈日当空，一位农民在田间辛勤劳作，汗水滴落到泥土里，镜头切换到桌上的米饭，以此来体现粒粒皆辛苦的意境。",
    "《寻隐者不遇》：在苍翠的山林中，一位书童指着远处的白云，白云深处有隐士的居所，但隐士不在，只留下了一片空寂的山水景色。"
]
    for i in test_list:
        test_video(tf, i+"--ratio 16:9 --duration 10")

