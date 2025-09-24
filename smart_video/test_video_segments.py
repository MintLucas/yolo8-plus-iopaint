#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/09/24 16:36
# @Author  : zhipeng16
# @Email   : zzzp50@ustc.edu
# @File    : test_video_segments.py
# @Usage   : Describe the file's purpose
from typing import List, Dict, Union

video_segments = [
    {"beginTime": 0.1, "endTime": 5.34, "theme": "", "type": "common", "by": "multimodal"},
    {"beginTime": 5.34, "endTime": 9.5, "theme": "top一。", "type": "common", "by": "multimodal"},
    {"beginTime": 9.5, "endTime": 22.94, "theme": "my friends. ", "type": "common", "by": "multimodal"},
    {"beginTime": 22.94, "endTime": 36.9, "theme": "with people who. ", "type": "common", "by": "multimodal"},
    {"beginTime": 36.9, "endTime": 53.23, "theme": "哦耶。", "type": "common", "by": "multimodal"},
    {"beginTime": 53.23, "endTime": 54.78, "theme": "哦耶。", "type": "common", "by": "multimodal"},
    {"beginTime": 54.78, "endTime": 59.86, "theme": "我不是。", "type": "common", "by": "multimodal"},
    {"beginTime": 59.86, "endTime": 96.84, "theme": "in that can way so we can break a party. ", "type": "common", "by": "multimodal"},
    {"beginTime": 96.84, "endTime": 115.7, "theme": "some with people s. ", "type": "common", "by": "multimodal"},
    {"beginTime": 115.7, "endTime": 119.78, "theme": "by beaching to be ten. ", "type": "common", "by": "multimodal"},
    {"beginTime": 119.78, "endTime": 143.16, "theme": "哦。", "type": "common", "by": "multimodal"},
    {"beginTime": 143.16, "endTime": 146.24, "theme": "靠谱啊。", "type": "common", "by": "multimodal"},
    {"beginTime": 146.24, "endTime": 169.6, "theme": "是累。", "type": "common", "by": "multimodal"},
    {"beginTime": 169.6, "endTime": 175.82, "theme": "but i wanted to be wanted to be. ", "type": "common", "by": "multimodal"},
    {"beginTime": 175.82, "endTime": 181.6, "theme": "的心。", "type": "common", "by": "multimodal"},
    {"beginTime": 181.6, "endTime": 182.9, "theme": "哦。", "type": "common", "by": "multimodal"},
    {"beginTime": 182.9, "endTime": 193.6, "theme": "挖头球。", "type": "common", "by": "multimodal"},
    {"beginTime": 193.6, "endTime": 225.78, "theme": "critic. ", "type": "common", "by": "multimodal"},
    {"beginTime": 225.78, "endTime": 229.46, "theme": "你。", "type": "common", "by": "multimodal"},
    {"beginTime": 229.46, "endTime": 233.14, "theme": "you can see what i'm trying to be. ", "type": "common", "by": "multimodal"},
    {"beginTime": 233.14, "endTime": 242.94, "theme": "i see. ", "type": "common", "by": "multimodal"},
    {"beginTime": 242.94, "endTime": 244.94, "theme": "of you. ", "type": "common", "by": "multimodal"},
    {"beginTime": 244.94, "endTime": 249.68, "theme": "很难。", "type": "common", "by": "multimodal"},
    {"beginTime": 249.68, "endTime": 289.14, "theme": "你看你看你。", "type": "common", "by": "multimodal"},
    {"beginTime": 289.14, "endTime": 291.34, "theme": "top三。", "type": "common", "by": "multimodal"},
    {"beginTime": 291.34, "endTime": 311.58, "theme": "oh, with first one thing to be, though, this are made to be come, and friends can never be. ", "type": "common", "by": "multimodal"},
    {"beginTime": 311.58, "endTime": 315.8, "theme": "goes and matter how long. ", "type": "common", "by": "multimodal"},
    {"beginTime": 315.8, "endTime": 322.14, "theme": "we don't know how to quit. ", "type": "common", "by": "multimodal"},
    {"beginTime": 322.14, "endTime": 346.36, "theme": "onion ight in world the. ", "type": "common", "by": "multimodal"},
    {"beginTime": 346.36, "endTime": 368.56, "theme": "get up to you. ", "type": "common", "by": "multimodal"},
    {"beginTime": 368.56, "endTime": 371.78, "theme": "welcome to the other member. ", "type": "common", "by": "multimodal"},
    {"beginTime": 371.78, "endTime": 389.14, "theme": "自己次。", "type": "common", "by": "multimodal"},
    {"beginTime": 389.14, "endTime": 394.98, "theme": "man, we alone to. ", "type": "common", "by": "multimodal"},
    {"beginTime": 394.98, "endTime": 402.72, "theme": "this water tonight line tonight. ", "type": "common", "by": "multimodal"},
    {"beginTime": 402.72, "endTime": 408.68, "theme": "he who stops won't get them once play. ", "type": "common", "by": "multimodal"},
    {"beginTime": 408.68, "endTime": 411.04, "theme": "i want the thing to know. ", "type": "common", "by": "multimodal"},
    {"beginTime": 411.04, "endTime": 453.3, "theme": "啊。", "type": "common", "by": "multimodal"},
    {"beginTime": 453.3, "endTime": 456.34, "theme": "嗯。", "type": "common", "by": "multimodal"},
    {"beginTime": 456.34, "endTime": 462.16, "theme": "enough things gonna be the same. ", "type": "common", "by": "multimodal"},
    {"beginTime": 462.16, "endTime": 474.24, "theme": "top四。", "type": "common", "by": "multimodal"},
    {"beginTime": 474.24, "endTime": 484.26, "theme": "第一天。", "type": "common", "by": "multimodal"},
    {"beginTime": 484.26, "endTime": 489.3, "theme": "we can set the woman next to me. ", "type": "common", "by": "multimodal"},
    {"beginTime": 489.3, "endTime": 493.72, "theme": "give me some earning. ", "type": "common", "by": "multimodal"},
    {"beginTime": 493.72, "endTime": 497.78, "theme": "sing of fear to play my all my. ", "type": "common", "by": "multimodal"},
    {"beginTime": 497.78, "endTime": 501.88, "theme": "don't blame it on me. ", "type": "common", "by": "multimodal"},
    {"beginTime": 501.88, "endTime": 506.3, "theme": "let me on the night. ", "type": "common", "by": "multimodal"},
    {"beginTime": 506.3, "endTime": 517.42, "theme": "总归。", "type": "common", "by": "multimodal"},
    {"beginTime": 517.42, "endTime": 523.1, "theme": "左边。", "type": "common", "by": "multimodal"},
    {"beginTime": 523.1, "endTime": 561.1, "theme": "每秒。", "type": "common", "by": "multimodal"},
    {"beginTime": 561.1, "endTime": 569.1, "theme": "play it on the night. ", "type": "common", "by": "multimodal"},
    {"beginTime": 569.1, "endTime": 591.76, "theme": "我没。", "type": "common", "by": "multimodal"},
    {"beginTime": 591.76, "endTime": 612.08, "theme": "top五。", "type": "common", "by": "multimodal"},
    {"beginTime": 612.08, "endTime": 626.6, "theme": "i was stronger. ", "type": "common", "by": "multimodal"},
    {"beginTime": 626.6, "endTime": 629.08, "theme": "ok. ", "type": "common", "by": "multimodal"},
    {"beginTime": 629.08, "endTime": 631.72, "theme": "我，so are you? ", "type": "common", "by": "multimodal"},
    {"beginTime": 631.72, "endTime": 636.02, "theme": "the same, the same role, the same world. ", "type": "common", "by": "multimodal"},
    {"beginTime": 636.02, "endTime": 645.84, "theme": "but this is i want we just go back back to how we came from. ", "type": "common", "by": "multimodal"},
    {"beginTime": 645.84, "endTime": 647.88, "theme": "summer air. ", "type": "common", "by": "multimodal"},
    {"beginTime": 647.88, "endTime": 655.12, "theme": "summer air, why do we just go back back to how we came from. ", "type": "common", "by": "multimodal"},
    {"beginTime": 655.12, "endTime": 656.64, "theme": "什么爱？", "type": "common", "by": "multimodal"},
    {"beginTime": 656.64, "endTime": 683.02, "theme": "ok, i get some too high. ", "type": "common", "by": "multimodal"},
    {"beginTime": 683.02, "endTime": 685.56, "theme": "ok. ", "type": "common", "by": "multimodal"},
    {"beginTime": 685.56, "endTime": 687.66, "theme": "还有new。", "type": "common", "by": "multimodal"},
    {"beginTime": 687.66, "endTime": 692.5, "theme": "i want you want to all in one more time. ", "type": "common", "by": "multimodal"},
    {"beginTime": 692.5, "endTime": 694.66, "theme": "see the night. ", "type": "common", "by": "multimodal"},
    {"beginTime": 694.66, "endTime": 696.97, "theme": "maybe two. ", "type": "common", "by": "multimodal"},
    {"beginTime": 696.97, "endTime": 702.32, "theme": "what we just go back back to we came from. ", "type": "common", "by": "multimodal"},
    {"beginTime": 702.32, "endTime": 704.0, "theme": "summer air. ", "type": "common", "by": "multimodal"},
    {"beginTime": 704.0, "endTime": 706.24, "theme": "什么l。", "type": "common", "by": "multimodal"},
    {"beginTime": 706.24, "endTime": 711.56, "theme": "why don't we just go back back to how we came from. ", "type": "common", "by": "multimodal"},
    {"beginTime": 711.56, "endTime": 713.6, "theme": "summer air. ", "type": "common", "by": "multimodal"},
    {"beginTime": 713.6, "endTime": 717.4, "theme": "summer work. ", "type": "common", "by": "multimodal"},
    {"beginTime": 717.4, "endTime": 730.14, "theme": "有。", "type": "common", "by": "multimodal"},
    {"beginTime": 730.14, "endTime": 739.74, "theme": "you could be the paradise. ", "type": "common", "by": "multimodal"},
    {"beginTime": 739.74, "endTime": 744.3, "theme": "chance to forget this night. ", "type": "common", "by": "multimodal"},
    {"beginTime": 744.3, "endTime": 748.74, "theme": "some back home, i feel like. ", "type": "common", "by": "multimodal"},
    {"beginTime": 748.74, "endTime": 751.1, "theme": "对你足够。", "type": "common", "by": "multimodal"},
    {"beginTime": 751.1, "endTime": 755.46, "theme": "let's make it right. ", "type": "common", "by": "multimodal"},
    {"beginTime": 755.46, "endTime": 767.9, "theme": "we came from. ", "type": "common", "by": "multimodal"}
]

def filter_closest_duration_clips(
    video_clips: List[Dict[str, Union[float, str]]],
    part_num: int,
    part_duration: float
) -> List[Dict[str, Union[float, str]]]:
    """
    Filters a list of video clips to find a specific number of clips
    with durations closest to a target duration.

    Args:
        video_clips (List[Dict]): A list of dictionaries, where each dictionary
                                  represents a video clip and must contain
                                  'beginTime' and 'endTime' keys.
        part_num (int): The number of clips to return.
        part_duration (float): The target duration in seconds.

    Returns:
        List[Dict]: A list containing the `part_num` clips that are
                    closest in duration to `part_duration`.
    """
    if not video_clips or part_num <= 0:
        return []

    # Sort the clips based on the absolute difference between their actual
    # duration and the target part_duration.
    # The 'key' argument uses a lambda function for a concise, inline calculation.
    sorted_clips = sorted(
        video_clips,
        key=lambda clip: abs((clip['endTime'] - clip['beginTime']) - part_duration)
    )

    # Return the top `part_num` clips from the sorted list.
    return sorted_clips[:part_num]


def filter_and_combine_video_segments(segments, part_num, part_duration):
    """
    Filters and combines a list of video segments to find 'part_num' results
    that are closest to 'part_duration'. The results are returned in chronological order.
    If individual segments are too short, they are combined with adjacent segments.

    Args:
        segments (list): A list of dictionaries, where each dictionary represents
                         a video segment and contains 'beginTime' and 'endTime'.
        part_num (int): The number of final segments to return.
        part_duration (float): The target duration for each segment.

    Returns:
        list: A new list containing the 'part_num' segments that are closest
              to the target duration, sorted by their 'beginTime'.
    """

    # 1. Calculate duration for each original segment
    processed_segments = [
        {
            'segment': seg,
            'duration': seg['endTime'] - seg['beginTime'],
            'beginTime': seg['beginTime']
        }
        for seg in segments
    ]

    # 2. Sort the segments by their beginTime to ensure chronological order for combination
    processed_segments.sort(key=lambda x: x['beginTime'])

    # 3. Combine small segments to meet the target duration
    combined_segments = []
    i = 0
    while i < len(processed_segments):
        current_segment = processed_segments[i]
        
        # If the current segment is much shorter than the target duration, try to combine it
        # with subsequent segments. The threshold (e.g., 0.5 * part_duration) is a heuristic.
        # You can adjust this value based on your specific needs.
        if current_segment['duration'] < 0.5 * part_duration and i + 1 < len(processed_segments):
            
            # Start a combination group with the current segment
            combined_group = [current_segment['segment']]
            current_duration = current_segment['duration']
            j = i + 1
            
            # Add adjacent segments until the duration is close to the target
            while j < len(processed_segments) and current_duration < 1.5 * part_duration:
                current_duration += processed_segments[j]['duration']
                combined_group.append(processed_segments[j]['segment'])
                j += 1
            
            # Create a new combined segment
            new_begin = combined_group[0]['beginTime']
            new_end = combined_group[-1]['endTime']
            
            combined_segments.append({
                'segment': {
                    'beginTime': new_begin,
                    'endTime': new_end,
                    'theme': "Combined Segment",
                    'type': "combined",
                    'by': "code"
                },
                'duration': new_end - new_begin,
                'beginTime': new_begin
            })
            # Move the pointer past the segments that were just combined
            i = j
        else:
            # If the segment is long enough, just add it to the list
            combined_segments.append(current_segment)
            i += 1

    # 4. Filter the combined segments to find the 'part_num' closest to part_duration
    # Calculate the absolute difference from the target duration for each combined segment.
    final_segments = []
    for seg in combined_segments:
        final_segments.append({
            'segment': seg['segment'],
            'diff_from_target': abs(seg['duration'] - part_duration)
        })

    # Sort these final segments based on the absolute difference
    final_segments.sort(key=lambda x: x['diff_from_target'])

    # Get the top 'part_num' segments
    top_segments = final_segments[:part_num]
    
    # 5. Sort the final result by beginTime
    final_result_segments = [item['segment'] for item in top_segments]
    final_result_segments.sort(key=lambda x: x['beginTime'])

    return final_result_segments



# print(filter_closest_duration_clips(video_segments, 5, 45))
# a = filter_and_combine_video_segments(video_segments, 5, 45)