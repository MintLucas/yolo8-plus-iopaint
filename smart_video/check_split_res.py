import os
import json
import subprocess

def ms_to_timestamp(seconds):
    """将秒转换为 HH:MM:SS.mmm 格式，方便日志查看"""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "{:02d}:{:02d}:{:06.3f}".format(int(h), int(m), s)

def cut_video_segments(video_path, segments_data, output_dir):
    """
    根据切分数据列表，将长视频切割为多个短视频
    """
    if not os.path.exists(video_path):
        print(f"❌ 错误：找不到视频文件: {video_path}")
        return

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📂 已创建输出目录: {output_dir}")

    # 获取原文件名（不带后缀）
    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    print(f"🚀 开始处理视频: {video_basename}")
    print(f"📊 总共 {len(segments_data)} 个片段")

    success_count = 0

    for i, seg in enumerate(segments_data):
        start_time = seg.get('beginTime')
        end_time = seg.get('endTime')
        
        # 计算持续时间，ffmpeg有些版本用 -t 更稳定
        duration = end_time - start_time
        
        # 构造输出文件名：原文件名_序号_开始时间_结束时间.mp4
        # 序号主要为了排序方便
        out_name = f"{i+1:03d}_{video_basename}_{start_time:.2f}_{end_time:.2f}.mp4"
        out_path = os.path.join(output_dir, out_name)

        # 检查片段是否已存在
        if os.path.exists(out_path):
            print(f"⏭️  跳过已存在: {out_name}")
            continue

        # 构造 FFmpeg 命令
        # -y: 覆盖输出
        # -ss: 开始时间 (放在 -i 前面是为了快速seek，但在重编码模式下，放在-i后面更精确，这里为了精度放在 -i 后或者配合重编码)
        # 注意：为了绝对的切分精度，建议不使用 -c copy，而是重新编码
        cmd = [
            "ffmpeg",
            "-y",                   # 自动覆盖
            "-hide_banner",         # 隐藏多余日志
            "-loglevel", "error",   # 减少刷屏
            "-i", video_path,       # 输入文件
            "-ss", str(start_time), # 开始时间
            "-t", str(duration),    # 持续时间 (比 -to 更不容易出bug)
            "-c:v", "libx264",      # 强制重编码，保证切分时间精确到帧
            "-c:a", "aac",          # 音频编码
            out_path
        ]

        try:
            # print(f"正在处理第 {i+1} 个片段...")
            subprocess.run(cmd, check=True)
            print(f"✅ [{i+1}/{len(segments_data)}] 完成: {out_name} (时长: {duration:.2f}s)")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"❌ 切分失败: {out_name}, Error: {e}")

    print(f"\n🎉 处理完毕！成功切分 {success_count} 个视频。")
    print(f"📁 结果保存在: {output_dir}")

if __name__ == "__main__":
    # ================= 配置区域 =================
    
    # 1. 视频路径 (修改为你实际的路径)
    video_file_path = '/workspace/work/zhipeng16/datasets/videos/3分钟游戏集锦.mp4'
    
    # 2. 输出目录
    output_directory = './smart_video/vit_cut_results2'

    # 3. 切分数据 (直接使用你提供的SOTA数据)
    # 如果你有json文件，可以使用: 
    # with open('path/to/json', 'r') as f: sota_data = json.load(f)
    sota_data = [
        {"beginTime": 0.1, "endTime": 5.18, "theme": "我突然发现比起狂徒，幻影才是更适合新手和低分段宝宝体质的枪械。", "type": "common", "by": "multimodal"},
        {"beginTime": 5.18, "endTime": 10.72, "theme": "包括在本次曼谷大师赛上，也是有越来越多的职业选手选择了幻影这把枪。", "type": "common", "by": "multimodal"},
        {"beginTime": 10.72, "endTime": 16.61, "theme": "但很多新手朋友用幻影就只会见人就蹲扫，大大降低了幻影这把枪的击杀效率。", "type": "common", "by": "multimodal"},
        {"beginTime": 16.61, "endTime": 18.76, "theme": "那怎么样用好幻影这把枪呢？", "type": "common", "by": "multimodal"},
        {"beginTime": 18.76, "endTime": 29.83, "theme": "我用这一个视频跟你们讲清楚一下，压四连发和狂徒的单点或者两连发不同，幻影单次开枪的节奏可以以四连发为一组，就是这样四发四发四发的打。", "type": "common", "by": "multimodal"},
        {"beginTime": 29.83, "endTime": 31.26, "theme": "为什么是四连发呢？", "type": "common", "by": "multimodal"},
        {"beginTime": 31.26, "endTime": 37.87, "theme": "因为这样是幻影容错率最高的射击方式，幻影加强后零到二十米打，身体的伤害是三十九。", "type": "common", "by": "multimodal"},
        {"beginTime": 37.87, "endTime": 56.8, "theme": "我们假设在射击一名敌人时，本来是想打头的，结果子弹全打到身上去了，那么三连发的伤害就是一百一十七点，四连发的伤害是一百五十六，也就是说用四连发，即便我们打头没打中，只要我们四连发全部打中敌人身体，那么也能击杀掉一名满血满甲的敌人。", "type": "common", "by": "multimodal"},
        {"beginTime": 56.8, "endTime": 60.02, "theme": "那如果说在远距离的情况下，四发子弹里有。", "type": "common", "by": "multimodal"},
        {"beginTime": 60.02, "endTime": 62.69, "theme": "一发打中了敌人头部那就更简单了。", "type": "common", "by": "multimodal"},
        {"beginTime": 62.69, "endTime": 69.99, "theme": "因为剩下三发子弹里，只要有一发打中敌人，无论是打头还是打身体，或者是打腿，敌人都必死无疑了。", "type": "common", "by": "multimodal"},
        {"beginTime": 69.99, "endTime": 71.06, "theme": "我有讲清楚吗？", "type": "common", "by": "multimodal"},
        {"beginTime": 71.06, "endTime": 74.45, "theme": "那我们的第一个练习就是掌握四连发的节奏。", "type": "common", "by": "multimodal"},
        {"beginTime": 74.45, "endTime": 78.72, "theme": "来到训练场对着墙练习，基本三分钟左右就可以快速掌握。", "type": "common", "by": "multimodal"},
        {"beginTime": 78.72, "endTime": 83.71, "theme": "要掌握好四连发之后，我们就要在开枪的同时添加一个下压的动作。", "type": "common", "by": "multimodal"},
        {"beginTime": 83.71, "endTime": 95.46, "theme": "下压就是为了刚才我们提到的，即便四连发没有打中头，我们也可以通过下压的方式将四发子弹全部打到敌人身上去，而不是让子弹往上飘，最后一发都打不中。", "type": "common", "by": "multimodal"},
        {"beginTime": 95.46, "endTime": 96.53, "theme": "那怎么下压呢？", "type": "common", "by": "multimodal"},
        {"beginTime": 96.53, "endTime": 99.2, "theme": "其实很简单，就是添加一个蹲的动作。", "type": "common", "by": "multimodal"},
        {"beginTime": 99.2, "endTime": 108.64, "theme": "但是这里一定一定一定要注意，蹲的动作是在你瞄准敌人头部开枪之后，再紧接着按吨，而不是在开枪之前，这里一定要分清顺序。", "type": "common", "by": "multimodal"},
        {"beginTime": 108.64, "endTime": 114.16, "theme": "如果说当敌人还在移动中，你先墩再开枪，敌人在动而你不动，你就是活靶子。", "type": "common", "by": "multimodal"},
        {"beginTime": 114.16, "endTime": 120.04, "theme": "然而当你瞄准了敌人头部，你急停开枪下蹲，这时候的下蹲一是可以帮你控制弹。", "type": "common", "by": "multimodal"},
        {"beginTime": 120.04, "endTime": 130.75, "theme": "二十米急停后，敌人也在趁机抓你不动的时候开枪，那么你蹲下来是可以躲避敌人的爆头线的，只是差了一点点顺序，效果天差地别，所以一定要注意。", "type": "common", "by": "multimodal"},
        {"beginTime": 130.75, "endTime": 132.68, "theme": "那下压四连发怎么练习呢？", "type": "common", "by": "multimodal"},
        {"beginTime": 132.68, "endTime": 150.4, "theme": "我们在训练场先打开机器人的护甲，这个很重要，接着开启一百个机器人，来到训练场的左边或者右边，移动中瞄准机器人头部，然后停下来，开枪下蹲，打掉之后继续移动，瞄准、急停、开枪、下蹲，往复循环，很快你就能找到这种下压四连发秒人的感觉了。", "type": "common", "by": "multimodal"}
    ]
    asr_data = [{'text': '我突然发现，比起狂徒幻影，才是更适合新手和低分段宝宝体质的枪械。', 'beginTime': 0.05, 'endTime': 4.89, 'start_str': '00:00:00.050', 'end_str': '00:00:04.890', 'duration': 4840, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '包括在本次曼谷大师赛上，也是有越来越多的职业选手选择了幻影这把枪，但很多新手朋友用幻影就只会见人就蹲扫，大大假设了幻影这把枪的击杀效率。', 'beginTime': 4.91, 'endTime': 14.65, 'start_str': '00:00:04.910', 'end_str': '00:00:14.650', 'duration': 9740, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '那怎么样用好幻影这把枪呢？', 'beginTime': 14.65, 'endTime': 16.81, 'start_str': '00:00:14.650', 'end_str': '00:00:16.810', 'duration': 2160, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '我用这一个视频跟你们讲清楚一下，压四连发和狂徒的单点或者两连发不同幻影，单次开枪的节奏可以以四连发为一组，就是这样，四发四发四发的打，为什么是四连发呢？', 'beginTime': 16.81, 'endTime': 30.29, 'start_str': '00:00:16.810', 'end_str': '00:00:30.290', 'duration': 13480, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '因为这样是幻影容错率最高的射击方式，幻影加强后零到二十米，打身体的伤害是三十九。', 'beginTime': 30.29, 'endTime': 37.31, 'start_str': '00:00:30.290', 'end_str': '00:00:37.310', 'duration': 7020, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '我们假设在设计一名敌人时，本来是想打头的，结果子弹全打到身上去了。', 'beginTime': 37.45, 'endTime': 42.83, 'start_str': '00:00:37.450', 'end_str': '00:00:42.830', 'duration': 5380, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '那么三连发的伤害就是一百一十七，而四连发的伤害是一百五十六，也就是说用四连发。', 'beginTime': 42.83, 'endTime': 49.83, 'start_str': '00:00:42.830', 'end_str': '00:00:49.830', 'duration': 7000, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '即便我们打头没打中，只要我们四连发全部打中敌人身体，那么也能击杀掉一名满血满甲的敌人。', 'beginTime': 49.91, 'endTime': 56.61, 'start_str': '00:00:49.910', 'end_str': '00:00:56.610', 'duration': 6700, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '那如果说在远距离的情况下，四发子弹里有一发打中了敌人头部，那就更简单了。', 'beginTime': 56.61, 'endTime': 62.5, 'start_str': '00:00:56.610', 'end_str': '00:01:02.500', 'duration': 5890, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '因为剩下三发子弹里只要有一发打中敌人，无论是打头还是打身体，或者是打腿，敌人都必死无疑了。', 'beginTime': 62.5, 'endTime': 69.24, 'start_str': '00:01:02.500', 'end_str': '00:01:09.240', 'duration': 6740, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '我有讲清楚吗？', 'beginTime': 69.24, 'endTime': 70.12, 'start_str': '00:01:09.240', 'end_str': '00:01:10.120', 'duration': 880, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '那我们的第一个练习就是掌握四连发的动作，来到训练场，对着墙练习，基本三分钟左右就可以快速掌握。', 'beginTime': 70.14, 'endTime': 76.94, 'start_str': '00:01:10.140', 'end_str': '00:01:16.940', 'duration': 6800, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '那掌握好四连发之后，我们就要在开枪的同时添加一个下压的动作。', 'beginTime': 77.0, 'endTime': 81.68, 'start_str': '00:01:17.000', 'end_str': '00:01:21.680', 'duration': 4680, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '下压就是为了刚才我们提到的，即便四连发没有打中头，我们也可以通过下压的方式将四发子弹全部打到敌人身上去。', 'beginTime': 81.82, 'endTime': 89.7, 'start_str': '00:01:21.820', 'end_str': '00:01:29.700', 'duration': 7880, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '而不是让子弹往上飘，最后一发都打不中。那怎么下压呢？', 'beginTime': 89.72, 'endTime': 94.02, 'start_str': '00:01:29.720', 'end_str': '00:01:34.020', 'duration': 4300, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '其实很简单，就是添加一个蹲的动作，但是这里一定一定一定要注意蹲的动作是在你瞄准敌人头部开枪之后，再紧接着按蹲，而不是在开枪之前，这里一定要分清顺序。', 'beginTime': 94.04, 'endTime': 107.14, 'start_str': '00:01:34.040', 'end_str': '00:01:47.140', 'duration': 13100, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '如果说当敌人还在移动中，你先蹲再开枪，敌人在动，而你不动也就是活靶子。', 'beginTime': 107.14, 'endTime': 113.38, 'start_str': '00:01:47.140', 'end_str': '00:01:53.380', 'duration': 6240, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '然而，当你瞄准了敌人头部，你急停开枪下蹲，这时候的下蹲一是可以帮你控制弹。', 'beginTime': 113.44, 'endTime': 119.92, 'start_str': '00:01:53.440', 'end_str': '00:01:59.915', 'duration': 6475, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '到二十急停后，敌人也在趁机抓你不动的时候开枪，那么你蹲下来是可以躲避敌人的，爆头线的，只是差了一点点顺序效果天差地别，所以一定要注意那下压四连发怎么练习呢？', 'beginTime': 120.23, 'endTime': 132.81, 'start_str': '00:02:00.230', 'end_str': '00:02:12.810', 'duration': 12580, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '我们在训练场先打开机器人的护甲，这个很重要。', 'beginTime': 132.81, 'endTime': 135.99, 'start_str': '00:02:12.810', 'end_str': '00:02:15.990', 'duration': 3180, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '接着开启一百个机器人来到训练场的左边或者右边移动中瞄准机器人的头部，然后停下来开枪下蹲打掉之后继续移动，瞄准急停开枪下蹲往复循环，很快你就能找到这种下压四连发秒人的感觉了。', 'beginTime': 136.05, 'endTime': 151.47, 'start_str': '00:02:16.050', 'end_str': '00:02:31.470', 'duration': 15420, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '我现在每天的练枪就是左边打五组，正面打五组，再来到右边打五组，然后就去打乱斗。', 'beginTime': 151.47, 'endTime': 157.39, 'start_str': '00:02:31.470', 'end_str': '00:02:37.390', 'duration': 5920, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '结合上育苗幻影的下压四连发真的很厉害，在排位实战里的击杀效率也丝毫不比狂突差。', 'beginTime': 157.41, 'endTime': 163.45, 'start_str': '00:02:37.410', 'end_str': '00:02:43.450', 'duration': 6040, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '二、控枪控枪就是控制幻影在扫射中的弹道。虽然无畏区别是随机弹道，但是你还是可以通过练习，将幻影的弹道控制在一块很小的区域内。', 'beginTime': 163.57, 'endTime': 173.51, 'start_str': '00:02:43.570', 'end_str': '00:02:53.510', 'duration': 9940, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '这样你混烟也好，火力压制也好，都会很有成效。', 'beginTime': 173.51, 'endTime': 176.57, 'start_str': '00:02:53.510', 'end_str': '00:02:56.570', 'duration': 3060, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '在训练场里关掉无限子弹，然后来到这个测试机这里先打最近的一口气，扫完三十发子弹，尽最大努力把所有子弹控制在靶子的最中心扫完一整，把枪的子弹再买一把，接着将靶子调到十米的距离，扫完一把枪，然后是二十米，距离二十米，再往后就不用练了。', 'beginTime': 176.57, 'endTime': 195.68, 'start_str': '00:02:56.570', 'end_str': '00:03:15.680', 'duration': 19110, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '太远距离的扫射没有意义。练习几次，你就会发现你对幻影的弹道，在小幅度的调整上有了自己的心得和技巧，在混烟和压制甚至击杀敌人的时候，都会越来越得心应手。', 'beginTime': 195.68, 'endTime': 207.36, 'start_str': '00:03:15.680', 'end_str': '00:03:27.360', 'duration': 11680, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '比如这里三扫射转移，在面对多命敌人时，幻影的扫射转移是非常好用且帅气的技巧。', 'beginTime': 207.38, 'endTime': 218.27, 'start_str': '00:03:27.380', 'end_str': '00:03:38.270', 'duration': 10890, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '那怎么练习扫射转移呢？', 'beginTime': 218.27, 'endTime': 219.89, 'start_str': '00:03:38.270', 'end_str': '00:03:39.890', 'duration': 1620, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '在训练场选出钢索，打开练习，用大招对准天花板的横梁发射两次，捆到机器人之后取消练习。', 'beginTime': 219.89, 'endTime': 227.31, 'start_str': '00:03:39.890', 'end_str': '00:03:47.310', 'duration': 7420, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '等大招结束，你就会发现训练场里只有两个机器人了。', 'beginTime': 227.37, 'endTime': 230.49, 'start_str': '00:03:47.370', 'end_str': '00:03:50.490', 'duration': 3120, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '我们打第一个机器人用下压四连发，打完后立马扫射转移到第二个机器人身上。', 'beginTime': 230.49, 'endTime': 235.87, 'start_str': '00:03:50.490', 'end_str': '00:03:55.870', 'duration': 5380, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '就这样练，等你觉得熟练一点了，就可以用钢索捆三个机器人打。', 'beginTime': 235.87, 'endTime': 239.81, 'start_str': '00:03:55.870', 'end_str': '00:03:59.810', 'duration': 3940, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '第一个用扫射转移打。第二个和第三个，如果一直打机器人没意思，你也可以来到外面扫射一些动态的小目标。', 'beginTime': 239.81, 'endTime': 246.5, 'start_str': '00:03:59.810', 'end_str': '00:04:06.495', 'duration': 6685, 'theme': '', 'type': 'Semantic', 'by': 'model1'}]
    vit_data = [{"text": "", "beginTime": 0.0, "endTime": 4.87, "start_str": "00:00:00,000", "end_str": "00:00:04,866", "duration": 4866, "theme": "", "type": "Semantic", "by": "CLIP_Scene_Detect"}, {"text": "", "beginTime": 4.87, "endTime": 5.53, "start_str": "00:00:04,866", "end_str": "00:00:05,533", "duration": 666, "theme": "", "type": "Semantic", "by": "CLIP_Scene_Detect"}, {"text": "", "beginTime": 5.53, "endTime": 20.07, "start_str": "00:00:05,533", "end_str": "00:00:20,066", "duration": 14533, "theme": "", "type": "Semantic", "by": "CLIP_Scene_Detect"}, {"text": "", "beginTime": 20.07, "endTime": 53.87, "start_str": "00:00:20,066", "end_str": "00:00:53,866", "duration": 33799, "theme": "", "type": "Semantic", "by": "CLIP_Scene_Detect"}, {"text": "", "beginTime": 53.87, "endTime": 54.8, "start_str": "00:00:53,866", "end_str": "00:00:54,799", "duration": 933, "theme": "", "type": "Semantic", "by": "CLIP_Scene_Detect"}, {"text": "", "beginTime": 54.8, "endTime": 55.4, "start_str": "00:00:54,799", "end_str": "00:00:55,399", "duration": 599, "theme": "", "type": "Semantic", "by": "CLIP_Scene_Detect"}, {"text": "", "beginTime": 55.4, "endTime": 56.6, "start_str": "00:00:55,399", "end_str": "00:00:56,599", "duration": 1199, "theme": "", "type": "Semantic", "by": "CLIP_Scene_Detect"}, {"text": "", "beginTime": 56.6, "endTime": 62.43, "start_str": "00:00:56,599", "end_str": "00:01:02,433", "duration": 5833, "theme": "", "type": "Semantic", "by": "CLIP_Scene_Detect"}, {"text": "", "beginTime": 62.43, "endTime": 70.1, "start_str": "00:01:02,433", "end_str": "00:01:10,099", "duration": 7666, "theme": "", "type": "Semantic", "by": "CLIP_Scene_Detect"}, {"text": "", "beginTime": 70.1, "endTime": 128.47, "start_str": "00:01:10,099", "end_str": "00:02:08,466", "duration": 58366, "theme": "", "type": "Semantic", "by": "CLIP_Scene_Detect"}, {"text": "", "beginTime": 128.47, "endTime": 129.7, "start_str": "00:02:08,466", "end_str": "00:02:09,699", "duration": 1233, "theme": "", "type": "Semantic", "by": "CLIP_Scene_Detect"}, {"text": "", "beginTime": 129.7, "endTime": 132.77, "start_str": "00:02:09,699", "end_str": "00:02:12,766", "duration": 3066, "theme": "", "type": "Semantic", "by": "CLIP_Scene_Detect"}, {"text": "", "beginTime": 132.77, "endTime": 176.6, "start_str": "00:02:12,766", "end_str": "00:02:56,599", "duration": 43833, "theme": "", "type": "Semantic", "by": "CLIP_Scene_Detect"}, {"text": "", "beginTime": 176.6, "endTime": 178.8, "start_str": "00:02:56,599", "end_str": "00:02:58,799", "duration": 2199, "theme": "", "type": "Semantic", "by": "CLIP_Scene_Detect"}, {"text": "", "beginTime": 178.8, "endTime": 187.7, "start_str": "00:02:58,799", "end_str": "00:03:07,699", "duration": 8899, "theme": "", "type": "Semantic", "by": "CLIP_Scene_Detect"}, {"text": "", "beginTime": 187.7, "endTime": 188.1, "start_str": "00:03:07,699", "end_str": "00:03:08,099", "duration": 399, "theme": "", "type": "Semantic", "by": "CLIP_Scene_Detect"}, {"text": "", "beginTime": 188.1, "endTime": 203.43, "start_str": "00:03:08,099", "end_str": "00:03:23,433", "duration": 15333, "theme": "", "type": "Semantic", "by": "CLIP_Scene_Detect"}, {"text": "", "beginTime": 203.43, "endTime": 207.27, "start_str": "00:03:23,433", "end_str": "00:03:27,266", "duration": 3833, "theme": "", "type": "Semantic", "by": "CLIP_Scene_Detect"}, {"text": "", "beginTime": 207.27, "endTime": 211.73, "start_str": "00:03:27,266", "end_str": "00:03:31,733", "duration": 4466, "theme": "", "type": "Semantic", "by": "CLIP_Scene_Detect"}, {"text": "", "beginTime": 211.73, "endTime": 213.3, "start_str": "00:03:31,733", "end_str": "00:03:33,299", "duration": 1566, "theme": "", "type": "Semantic", "by": "CLIP_Scene_Detect"}, {"text": "", "beginTime": 213.3, "endTime": 214.13, "start_str": "00:03:33,299", "end_str": "00:03:34,133", "duration": 833, "theme": "", "type": "Semantic", "by": "CLIP_Scene_Detect"}, {"text": "", "beginTime": 214.13, "endTime": 218.23, "start_str": "00:03:34,133", "end_str": "00:03:38,233", "duration": 4099, "theme": "", "type": "Semantic", "by": "CLIP_Scene_Detect"}, {"text": "", "beginTime": 218.23, "endTime": 220.53, "start_str": "00:03:38,233", "end_str": "00:03:40,533", "duration": 2299, "theme": "", "type": "Semantic", "by": "CLIP_Scene_Detect"}, {"text": "", "beginTime": 220.53, "endTime": 246.73, "start_str": "00:03:40,533", "end_str": "00:04:06,733", "duration": 26199, "theme": "", "type": "Semantic", "by": "CLIP_Scene_Detect"}]
    #case2音乐
    sota_data = [
    {"beginTime":0.1,"endTime":0.2,"theme":"","type":"common","by":"multimodal"},
    {"beginTime":0.2,"endTime":12.65,"theme":"适者才配生存。","type":"common","by":"multimodal"},
    {"beginTime":12.65,"endTime":21.58,"theme":"我的地盘。","type":"common","by":"multimodal"},
    {"beginTime":21.58,"endTime":25.8,"theme":"生长在。","type":"common","by":"multimodal"},
    {"beginTime":25.8,"endTime":27.84,"theme":"for me. ","type":"common","by":"multimodal"},
    {"beginTime":27.84,"endTime":29.32,"theme":"听我。","type":"common","by":"multimodal"},
    {"beginTime":29.32,"endTime":41.55,"theme":"我操。","type":"common","by":"multimodal"},
    {"beginTime":41.55,"endTime":57.26,"theme":"我还剩一米。","type":"common","by":"multimodal"},
    {"beginTime":57.26,"endTime":62.58,"theme":"我都这么好帅。","type":"common","by":"multimodal"},
    {"beginTime":62.58,"endTime":68.08,"theme":"baby嘴巴还在路上。","type":"common","by":"multimodal"},
    {"beginTime":68.08,"endTime":86.6,"theme":"没错。","type":"common","by":"multimodal"},
    {"beginTime":86.6,"endTime":87.04,"theme":"耶。","type":"common","by":"multimodal"},
    {"beginTime":87.04,"endTime":90.02,"theme":"快手出击。","type":"common","by":"multimodal"},
    {"beginTime":90.02,"endTime":97.94,"theme":"oh, thank you. ","type":"common","by":"multimodal"},
    {"beginTime":97.94,"endTime":99.02,"theme":"而已。","type":"common","by":"multimodal"},
    {"beginTime":99.02,"endTime":104.62,"theme":"飞机。","type":"common","by":"multimodal"},
    {"beginTime":104.62,"endTime":119.44,"theme":"stand. ","type":"common","by":"multimodal"},
    {"beginTime":119.44,"endTime":123.52,"theme":"baby嘴巴很疼。","type":"common","by":"multimodal"},
    {"beginTime":123.52,"endTime":132.5,"theme":"早已不是可以。","type":"common","by":"multimodal"},
    {"beginTime":132.5,"endTime":139.05,"theme":"的情况下。","type":"common","by":"multimodal"},
    {"beginTime":139.05,"endTime":151.12,"theme":"哎呀。","type":"common","by":"multimodal"},
    {"beginTime":151.12,"endTime":155.78,"theme":"我要鲨鱼。","type":"common","by":"multimodal"},
    {"beginTime":155.78,"endTime":175.58,"theme":"打开。","type":"common","by":"multimodal"},
    {"beginTime":175.58,"endTime":177.91,"theme":"你要买什么枪，我请客。","type":"common","by":"multimodal"},
    {"beginTime":177.91,"endTime":182.5,"theme":"赶快赶快，我要去前线，我要出手了。","type":"common","by":"multimodal"},
    {"beginTime":182.5,"endTime":184.16,"theme":"悠着点儿。","type":"common","by":"multimodal"},
    {"beginTime":184.16,"endTime":188.75,"theme":"给我上。","type":"common","by":"multimodal"}
]
    asr_data = [{'text': '适者退我的地盘，好险好，险风正在我trilyformel你看见了清楚m难道比因为一条苹果难谢谢你玫瑰树苹我的妈呀你我自己为什么样子的一天就还有一小点的还剩一厘米夜小并眼泪落哭涩都已无所谓泪', 'beginTime': 0.77, 'endTime': 58.9, 'start_str': '00:00:00.770', 'end_str': '00:00:58.900', 'duration': 58130, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '你们都进laoughplayerplanding别夜所们我都以为所可以h', 'beginTime': 61.39, 'endTime': 71.06, 'start_str': '00:01:01.390', 'end_str': '00:01:11.065', 'duration': 9675, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '你送给你之后r我们还还我碎便那个一点点讨厌快快手出击暖风跟你说你thankyou，我天飞你的歌事那', 'beginTime': 72.33, 'endTime': 101.56, 'start_str': '00:01:12.330', 'end_str': '00:01:41.560', 'duration': 29230, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': "it'syouwithyouwhereyourstand快", 'beginTime': 101.64, 'endTime': 106.03, 'start_str': '00:01:41.640', 'end_str': '00:01:46.030', 'duration': 4390, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '你对过寒暗跟你说有多乖的是你们是可以走一生飞', 'beginTime': 119.73, 'endTime': 127.74, 'start_str': '00:01:59.730', 'end_str': '00:02:07.740', 'duration': 8010, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '然后我们新朋友的所有的未名人 你早已晚上看耶稣 其实我我有欢迎盒子里为何可以我还大呢对', 'beginTime': 132.43, 'endTime': 155.65, 'start_str': '00:02:12.430', 'end_str': '00:02:35.645', 'duration': 23215, 'theme': '', 'type': 'Semantic', 'by': 'model1'}, {'text': '富的这也是吧有儿童可能哪怕你要买什么枪。我请客快快快快的我要去前线人我要出手了上悠着点儿都给我找有', 'beginTime': 159.0, 'endTime': 185.06, 'start_str': '00:02:39.000', 'end_str': '00:03:05.065', 'duration': 26065, 'theme': '', 'type': 'Semantic', 'by': 'model1'}]
    vit_data = [
    {
        "text": "",
        "beginTime": 0.0,
        "endTime": 12.03,
        "start_str": "00:00:00,000",
        "end_str": "00:00:12,033",
        "duration": 12033,
        "theme": "",
        "type": "Semantic",
        "by": "CLIP_Scene_Detect"
    },
    {
        "text": "",
        "beginTime": 12.03,
        "endTime": 12.27,
        "start_str": "00:00:12,033",
        "end_str": "00:00:12,266",
        "duration": 233,
        "theme": "",
        "type": "Semantic",
        "by": "CLIP_Scene_Detect"
    },
    {
        "text": "",
        "beginTime": 12.27,
        "endTime": 12.5,
        "start_str": "00:00:12,266",
        "end_str": "00:00:12,499",
        "duration": 233,
        "theme": "",
        "type": "Semantic",
        "by": "CLIP_Scene_Detect"
    },
    {
        "text": "",
        "beginTime": 12.5,
        "endTime": 12.7,
        "start_str": "00:00:12,499",
        "end_str": "00:00:12,699",
        "duration": 199,
        "theme": "",
        "type": "Semantic",
        "by": "CLIP_Scene_Detect"
    },
    {
        "text": "",
        "beginTime": 12.7,
        "endTime": 12.93,
        "start_str": "00:00:12,699",
        "end_str": "00:00:12,933",
        "duration": 233,
        "theme": "",
        "type": "Semantic",
        "by": "CLIP_Scene_Detect"
    },
    {
        "text": "",
        "beginTime": 12.93,
        "endTime": 43.73,
        "start_str": "00:00:12,933",
        "end_str": "00:00:43,733",
        "duration": 30799,
        "theme": "",
        "type": "Semantic",
        "by": "CLIP_Scene_Detect"
    },
    {
        "text": "",
        "beginTime": 43.73,
        "endTime": 58.2,
        "start_str": "00:00:43,733",
        "end_str": "00:00:58,199",
        "duration": 14466,
        "theme": "",
        "type": "Semantic",
        "by": "CLIP_Scene_Detect"
    },
    {
        "text": "",
        "beginTime": 58.2,
        "endTime": 91.87,
        "start_str": "00:00:58,199",
        "end_str": "00:01:31,866",
        "duration": 33666,
        "theme": "",
        "type": "Semantic",
        "by": "CLIP_Scene_Detect"
    },
    {
        "text": "",
        "beginTime": 91.87,
        "endTime": 94.2,
        "start_str": "00:01:31,866",
        "end_str": "00:01:34,199",
        "duration": 2333,
        "theme": "",
        "type": "Semantic",
        "by": "CLIP_Scene_Detect"
    },
    {
        "text": "",
        "beginTime": 94.2,
        "endTime": 119.33,
        "start_str": "00:01:34,199",
        "end_str": "00:01:59,333",
        "duration": 25133,
        "theme": "",
        "type": "Semantic",
        "by": "CLIP_Scene_Detect"
    },
    {
        "text": "",
        "beginTime": 119.33,
        "endTime": 119.53,
        "start_str": "00:01:59,333",
        "end_str": "00:01:59,533",
        "duration": 199,
        "theme": "",
        "type": "Semantic",
        "by": "CLIP_Scene_Detect"
    },
    {
        "text": "",
        "beginTime": 119.53,
        "endTime": 122.8,
        "start_str": "00:01:59,533",
        "end_str": "00:02:02,799",
        "duration": 3266,
        "theme": "",
        "type": "Semantic",
        "by": "CLIP_Scene_Detect"
    },
    {
        "text": "",
        "beginTime": 122.8,
        "endTime": 123.13,
        "start_str": "00:02:02,799",
        "end_str": "00:02:03,133",
        "duration": 333,
        "theme": "",
        "type": "Semantic",
        "by": "CLIP_Scene_Detect"
    },
    {
        "text": "",
        "beginTime": 123.13,
        "endTime": 124.43,
        "start_str": "00:02:03,133",
        "end_str": "00:02:04,433",
        "duration": 1299,
        "theme": "",
        "type": "Semantic",
        "by": "CLIP_Scene_Detect"
    },
    {
        "text": "",
        "beginTime": 124.43,
        "endTime": 126.83,
        "start_str": "00:02:04,433",
        "end_str": "00:02:06,833",
        "duration": 2399,
        "theme": "",
        "type": "Semantic",
        "by": "CLIP_Scene_Detect"
    },
    {
        "text": "",
        "beginTime": 126.83,
        "endTime": 131.57,
        "start_str": "00:02:06,833",
        "end_str": "00:02:11,566",
        "duration": 4733,
        "theme": "",
        "type": "Semantic",
        "by": "CLIP_Scene_Detect"
    },
    {
        "text": "",
        "beginTime": 131.57,
        "endTime": 155.93,
        "start_str": "00:02:11,566",
        "end_str": "00:02:35,933",
        "duration": 24366,
        "theme": "",
        "type": "Semantic",
        "by": "CLIP_Scene_Detect"
    },
    {
        "text": "",
        "beginTime": 155.93,
        "endTime": 192.83,
        "start_str": "00:02:35,933",
        "end_str": "00:03:12,833",
        "duration": 36899,
        "theme": "",
        "type": "Semantic",
        "by": "CLIP_Scene_Detect"
    }
]


    # ================= 执行区域 =================
    cut_video_segments(video_file_path, vit_data, output_directory)