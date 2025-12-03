#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/12/02 16:27
# @Author  : zhipeng16
# @Email   : zzzp50@ustc.edu
# @File    : model_compare.py
# @Usage   : Describe the file's purpose
import sys,os,json
sys.path.append(os.getcwd())
import ast
from util.excel_util import excel_merged_to_complex_dict,complex_dict_to_excel
from util.token_util_new import token_fresh, models_dict
from util.api_client_new import Api_client


def style_exctrat_from_excel():
    t_f = token_fresh()
    a_p = Api_client()
    path = "tmp_data/excel_data/跨域领域星推官各风格参考优质博文_gen.xlsx"  # 替换为你实际的文件路径
    result = excel_merged_to_complex_dict(path, 0)
    a_p
    new_dict = {}
    with open('config/prompt/cat_rec_star/create_blog_v2_noref.md', encoding='utf-8') as f:
        base_prompt = f.read()
    for col_one,row_list in result.items():
        new_row_list = []
        for one_row in row_list:
            new_one_row = {}
            for col_name, col_value in one_row.items():
                new_one_row[col_name] = col_value
                if col_name == '博文mid':
                    text = a_p.get_mid_text(col_value)
                    new_one_row['历史优质文案'] = text
                    llm_res_str = t_f.call_model_google(base_prompt+"/n用户输入文案："+text, "")
                    llm_res_json = json.loads(llm_res_str.strip("```json").strip("```"))
                    all_summary_res = llm_res_json['generated_texts']
                    for i in all_summary_res:
                        new_one_row[i['style']] = i['content']
            new_row_list.append(new_one_row)
        new_dict[col_one] = new_row_list
        
        
def text_model_compare():
    t_f = token_fresh()
    a_p = Api_client()
    path = "tmp_data/excel_data/跨域领域星推官各风格参考优质博文_gen.xlsx"  # 替换为你实际的文件路径
    result = excel_merged_to_complex_dict(path, 1)
    a_p
    new_dict = {}
    with open('config/prompt/cat_rec_star/create_blog_v2_noref.md', encoding='utf-8') as f:
        base_prompt = f.read()
    for col_one,row_list in result.items():
        new_row_list = []
        for one_row in row_list:
            new_one_row = {}
            for col_name, col_value in one_row.items():
                new_one_row[col_name] = col_value
                # if col_name == '博文mid':
                    # text = a_p.get_mid_text(col_value)
                if col_name == '原博文链接':
                    mid_normal = a_p.get_normal_mid(col_value)
                    text = a_p.get_mid_text(mid_normal)
                    new_one_row['原始文案'] = text
                    modes_list = ['text-qwen', 'text-huoshan','text-ds','text-google']
                    for model_name in modes_list:
                        try:
                            if 'google'in model_name :
                                llm_res_str = t_f.call_model_google(base_prompt+"/n要转发的博文内容："+text, "")
                            else:
                                llm_res_str = t_f.call_model_zp(base_prompt+"/n要转发的博文内容："+text, "",model_type=models_dict[model_name])
                            llm_res_json = ast.literal_eval(llm_res_str.strip("```json").strip("```"))
                            # llm_res_json = json.loads(llm_res_str)
                            all_summary_res = llm_res_json['generated_texts']
                            for i in all_summary_res:
                                new_one_row[i['style_type']+model_name.split("-")[-1]] = i['content']
                        except Exception as e:
                            print("zperror"+str(e))
                            import traceback
                            print(traceback.format_exc())
                            
            new_row_list.append(new_one_row)
        new_dict[col_one] = new_row_list
    complex_dict_to_excel('summary_model_compare.xlsx', new_dict)
            
    
    

if __name__ == "__main__":
    text_model_compare()
    