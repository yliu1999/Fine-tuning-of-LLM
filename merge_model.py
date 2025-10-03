# scripts/merge_model.py
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def merge_and_save_model():
    # 基础模型路径
    base_model_name = "Qwen/Qwen2-1.5B-Instruct"
    # LoRA适配器路径
    adapter_path = "./outputs/medical_lora_adapter"
    # 合并后模型保存路径
    output_path = "./merged_medical_model"

    print("正在加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    print("正在加载LoRA适配器并合并...")
    # 加载适配器并合并
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = model.merge_and_unload()  # 关键步骤：合并

    # 保存完整模型
    print(f"保存合并后的模型到 {output_path}...")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print("✅ 模型合并完成！")


if __name__ == "__main__":
    merge_and_save_model()