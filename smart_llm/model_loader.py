# model_loader.py
import argparse
import os
import torch
from typing import Dict, List, Tuple

class LLMModel:
    """
    一个封装了LLM加载和推理逻辑的类。
    """
    def __init__(self, model_path: str, model_type: str, device: str = '0', dtype: str = 'bf16', dim: str = '5120'):
        """
        初始化模型加载器。
        
        Args:
            model_path (str): 模型文件的路径。
            model_type (str): 模型类型 ('baichuan2', 'chatglm3')。
            device (str): 使用的GPU设备ID。
            dtype (str): 模型的数据类型 ('bf16', 'fp16')。
            dim (str): 模型维度（为兼容旧加载逻辑保留）。
        """
        print("Initializing LLMModel...")
        self.model_path = model_path
        self.model_type = model_type
        self.device_id = device.split(',')[0] # 取第一个GPU作为主设备
        self.dtype = dtype
        self.dim = dim
        
        self.device = torch.device(f"cuda:{self.device_id}" if torch.cuda.is_available() and self.device_id else "cpu")
        
        # 加载模型和分词器
        self.model, self.tokenizer = self._load_model()
        print(f"✅ Model '{self.model_path}' loaded successfully on device '{self.device}'.")

    def _load_model(self):
        """
        私有方法，根据模型类型加载预训练模型和分词器。
        """
        compute_dtype = torch.bfloat16 if self.dtype == 'bf16' else torch.float16
        config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": None,
        "revision": 'main',
        "use_auth_token": None,
    }
        loaded_model, loaded_tokenizer = None, None

        if self.model_type == "baichuan2":
            from baichuan.tokenization_baichuan import BaichuanTokenizer
            from baichuan.configuration_baichuan import BaichuanConfig
            if int(self.dim) == 64:
                from baichuan.modeling_baichuan_amlp import BaichuanForCausalLM
            else:
                from baichuan.modeling_baichuan_hidden import BaichuanForCausalLM
            
            loaded_tokenizer = BaichuanTokenizer.from_pretrained(self.model_path, use_fast=False, **config_kwargs)
                        # --- 开始修改 ---
            # # 构造 tokenizer.model 文件的完整路径
            # vocab_file_path = os.path.join(self.model_path, "tokenizer.model")
            # # 直接使用类的构造函数 __init__ 进行实例化，绕开 from_pretrained
            # loaded_tokenizer = BaichuanTokenizer(vocab_file=vocab_file_path)
            # # --- 修改结束 ---
            
            config = BaichuanConfig.from_pretrained(self.model_path, **config_kwargs)
            loaded_model = BaichuanForCausalLM.from_pretrained(
                self.model_path, config=config, torch_dtype=compute_dtype, low_cpu_mem_usage=True
            )

        elif self.model_type == "chatglm3":
            from chatglm3.tokenization_chatglm import ChatGLMTokenizer
            from chatglm3.configuration_chatglm import ChatGLMConfig
            if int(self.dim) == 64:
                from chatglm3.modeling_chatglm_bak import ChatGLMForConditionalGeneration
            else:
                from chatglm3.modeling_chatglm import ChatGLMForConditionalGeneration

            loaded_tokenizer = ChatGLMTokenizer.from_pretrained(self.model_path, **config_kwargs)
            config = ChatGLMConfig.from_pretrained(self.model_path, **config_kwargs)
            loaded_model = ChatGLMForConditionalGeneration.from_pretrained(
                self.model_path, config=config, torch_dtype=compute_dtype, low_cpu_mem_usage=True
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        loaded_model = loaded_model.to(self.device).eval()
        return loaded_model, loaded_tokenizer

    def chat_embdding(self, prompt: str, history: List[Dict[str, str]], max_length: int, temperature: float, top_p: float) -> Tuple[str, List[Dict[str, str]]]:
        """
        执行对话生成。
        
        Returns:
            Tuple[str, List[Dict[str, str]]]: (生成的回复, 更新后的对话历史)
        """
        if self.model_type == "baichuan2":
            messages = history + [{'role': 'user', 'content': prompt}]
            # Baichuan的chat方法通常是流式的，这里我们直接获取最终结果
            response_text = self.model.chat(self.tokenizer, messages)
            updated_history = messages + [{'role': 'assistant', 'content': response_text}]
            return response_text, updated_history

        elif self.model_type == "chatglm3":
            response_text, updated_history = self.model.chat(
                self.tokenizer,
                prompt,
                history=history,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
            return response_text, updated_history
        else:
            raise ValueError(f"Chat function not implemented for model type: {self.model_type}")


    # model_loader.py

    def chat(self, prompt: str, history: List[Dict[str, str]], max_length: int, temperature: float, top_p: float) -> Tuple[str, List[Dict[str, str]]]:
        """
        执行对话生成。
        """
        if self.model_type == "baichuan2":
            # 1. 准备 messages 列表用于生成提示
            messages = history + [{'role': 'user', 'content': prompt}]
            
            # 2. 使用 tokenizer 将 messages 列表转换为模型输入ID
            #    注意：新版 transformers 使用 apply_chat_template，这里我们手动构建输入
            #    手动构建输入字符串（适用于Baichuan-13B-Chat）
            total_prompt = ""
            for message in messages:
                if message['role'] == 'user':
                    total_prompt += f"<reserved_102>{message['content']}"
                elif message['role'] == 'assistant':
                    total_prompt += f"<reserved_103>{message['content']}"
            
            inputs = self.tokenizer(total_prompt, return_tensors="pt", add_special_tokens=False).to(self.device)

            # 3. 使用标准的 .generate() 方法生成文本的 token ID
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.1, # 添加重复惩罚，避免模型说车轱辘话
                eos_token_id=self.tokenizer.eos_token_id
            )

            # 4. 将生成的 token ID 解码成文字
            #    outputs[0] 包含了输入的ids和新生成的ids，我们需要切片掉输入部分
            input_len = inputs.input_ids.shape[1]
            response_text = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
            
            # 5. 更新历史记录
            updated_history = messages + [{'role': 'assistant', 'content': response_text}]
            return response_text, updated_history

        elif self.model_type == "chatglm3":
            response_text, updated_history = self.model.chat(
                self.tokenizer,
                prompt,
                history=history,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
            return response_text, updated_history
        else:
            raise ValueError(f"Chat function not implemented for model type: {self.model_type}")


if __name__ == '__main__':
    """
    当直接运行此文件时，执行此调试代码块。
    用法: 
    python model_loader.py --model_type baichuan2 --model_path /path/to/your/model --device 0
    """
    parser = argparse.ArgumentParser(description="LLM Model Loader and Tester")
    parser.add_argument('--model_type', default='baichuan2', required=False, type=str, help='模型类型: baichuan2, chatglm3')
    parser.add_argument('--model_path', default="/data3/push_recall/models/Baichuan2-13B-Chat", required=False, type=str, help='模型文件路径')
    parser.add_argument('--device', default='0', type=str, help='GPU设备ID')
    parser.add_argument('--dtype', default='bf16', type=str, help='模型数据类型')
    parser.add_argument('--dim', default='5120', type=str, help='模型维度')
    args = parser.parse_args()

    # 设置CUDA设备
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    print("--- [调试模式] 正在加载模型... ---")
    try:
        llm_instance = LLMModel(
            model_path=args.model_path,
            model_type=args.model_type,
            device=args.device,
            dtype=args.dtype,
            dim=args.dim
        )
        print("\n--- 模型加载成功，进入交互式对话测试 ---")
        print("--- 输入 'quit' 或 'exit' 退出 ---")
        
        history = []
        while True:
            prompt = input("用户: ")
            if prompt.lower() in ["quit", "exit"]:
                break
            
            response, history = llm_instance.chat(prompt, history, 8192, 0.7, 0.8)
            print(f"模型: {response}")
            # print(f"  [调试信息] 当前历史长度: {len(history)}")

    except Exception as e:
        print(f"\n--- [错误] 模型加载或测试失败: {e} ---")