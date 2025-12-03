#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/12/02 16:27
# @Author  : zhipeng16
# @Email   : zzzp50@ustc.edu
# @File    : excel_util.py
# @Usage   : Describe the file's purpose
import pandas as pd

def excel_to_dict(path):
    df = pd.read_excel(path)
    df.iloc[:, 0] = df.iloc[:, 0].ffill()
    return df.groupby(df.columns[0])[df.columns[1]].apply(list).to_dict()

def excel_merged_to_complex_dict(file_path, sheet_name=None):
    """
    读取Excel，处理第一列合并单元格。
    将数据转换为：{ 第一列值: [ {列名2: 值, 列名3: 值...}, ... ] } 的格式。
    """
    # 1. 读取Excel文件 (需要安装 openpyxl 库);sheet_name传None会变成一个sheet_name:pandas的dict
    if file_path.endswith('.csv'):
        # CSV读取
        df = pd.read_csv(file_path, sheet_name=0 if not sheet_name else sheet_name, dtype='str')
    else:
        df = pd.read_excel(file_path,sheet_name=0 if not sheet_name else sheet_name, dtype='str')
    
    # 2. 处理合并单元格：
    # 选中第一列 (iloc[:, 0])，使用 ffill() 向下填充空值，还原合并前的结构
    df.iloc[:, 0] = df.iloc[:, 0].ffill()
    
    # 3. 提取第一列的列名，作为后续分组的依据
    key_col = df.columns[0]
    
    # 4. 分组并清洗数据：
    # (1) groupby: 按第一列分组
    # (2) apply: 对每组数据进行转换
    # (3) drop: 组内数据去掉第一列（因为已经是Key了，不需要在Value里重复）
    # (4) to_dict('records'): 将剩余列转为 [{'col2': val, 'col3': val}, ...] 的标准格式
    result_dict = df.groupby(key_col).apply(
        lambda x: x.drop(columns=[key_col]).to_dict(orient='records')
    ).to_dict()
    
    return result_dict

def complex_dict_to_excel(save_path, data_dict, sheet_name='Sheet1', key_col_name='Category'):
    """
    将字典还原为Excel，强制写入为文本格式
    """
    rows = []
    for key, item_list in data_dict.items():
        for item in item_list:
            row = item.copy()
            row[key_col_name] = key
            rows.append(row)
            
    df = pd.DataFrame(rows)
    
    # 1. 调整列顺序
    cols = [key_col_name] + [c for c in df.columns if c != key_col_name]
    df = df[cols]
    
    # 2. 关键步骤：处理空值并强制转为字符串
    # 先将 NaN 填充为空字符串，否则转字符串后会变成 'nan' 字样
    df = df.fillna('')
    # 强制所有数据转为字符串类型
    df = df.astype(str)
    
    # 3. 设置索引并写入
    df.set_index(key_col_name, inplace=True)
    
    try:
        df.to_excel(save_path, sheet_name=sheet_name, merge_cells=True)
        print(f"成功保存（全文本格式）至: {save_path}")
    except Exception as e:
        print(f"保存失败: {e}")

# --- 使用示例 ---
# dummy_data = {
#     '自然型': [{'ID': 101, '链接': 'url_a'}, {'ID': 102, '链接': 'url_b'}],
#     '文艺型': [{'ID': 201, '链接': 'url_c'}]
# }
# complex_dict_to_excel('output.xlsx', dummy_data, sheet_name='结果页', key_col_name='风格类型')

if __name__ == "__main__":
    path = "tmp_data/excel_data/跨域领域星推官各风格参考优质博文.xlsx"  # 替换为你实际的文件路径
    result = excel_merged_to_complex_dict(path)
    complex_dict_to_excel('tmp_data/excel_data/tmp.xlsx', result)