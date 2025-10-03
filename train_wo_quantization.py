# train_fixed.py
import os
import json
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import torch

# 设置环境变量（在脚本开始时设置）
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"


def load_training_data(use_subset=True, subset_size=500):
    """加载训练数据"""
    data_file = 'data/train.jsonl'

    with open(data_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    if use_subset:
        data = data[:subset_size]
        print(f"⚠️ 使用数据子集进行训练: {len(data)} 条")
    else:
        print(f"✅ 使用全部数据训练: {len(data)} 条")

    training_data = []
    for item in data:
        if item['input'].strip():
            text = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"
        else:
            text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
        training_data.append({"text": text})

    return Dataset.from_list(training_data)


def main():
    print("=== 医疗大模型微调训练开始 ===")

    # 加载训练数据
    dataset = load_training_data(use_subset=True, subset_size=500)

    # --- 1. 加载模型与分词器 ---
    model_name = "Qwen/Qwen2-1.5B-Instruct"

    print("正在加载模型和分词器...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("✅ 模型加载成功")

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # --- 2. 配置LoRA ---
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    model = get_peft_model(model, peft_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"📊 可训练参数: {trainable_params:,} | 总参数: {total_params:,} | 占比: {100 * trainable_params / total_params:.4f}%")

    # --- 3. 定义训练参数 ---
    training_args = TrainingArguments(
        output_dir="./outputs",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
        warmup_steps=100,
        report_to=None,
        dataloader_pin_memory=False,
        # 禁用分布式训练（单GPU）
        no_cuda=False,  # 确保使用CUDA
        local_rank=int(os.getenv("LOCAL_RANK", -1)),  # 用于分布式训练
    )

    # --- 4. 初始化Trainer ---
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        packing=False,
    )

    # --- 5. 开始训练 ---
    print("🚀 开始训练...")

    try:
        trainer.train()
        print("✅ 训练完成！")
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        return

    # --- 6. 保存适配器权重 ---
    output_dir = "./outputs/medical_lora_adapter"
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"💾 适配器权重已保存到: {output_dir}")


if __name__ == "__main__":
    main()