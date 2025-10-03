# scripts/train.py
from datasets import Dataset
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import torch


def load_training_data():
    """加载训练数据"""
    with open('data/train.jsonl', 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # 创建训练文本
    training_data = []
    for item in data:
        if item['input'].strip():
            text = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"
        else:
            text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
        training_data.append({"text": text})

    return Dataset.from_list(training_data)


def main():
    # --- 1. 加载模型与分词器 (4-bit量化) ---
    model_name = "Qwen/Qwen2-7B-Instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print("正在加载模型和分词器...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 设置pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 2. 准备模型 for PEFT Training ---
    model = prepare_model_for_kbit_training(model)

    # --- 3. 配置LoRA ---
    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)

    # 打印可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"可训练参数: {trainable_params} | 总参数: {total_params} | 占比: {100 * trainable_params / total_params:.2f}%")

    # --- 4. 加载数据集 ---
    print("正在加载训练数据...")
    dataset = load_training_data()
    print(f"训练数据量: {len(dataset)} 条样本")

    # --- 5. 定义训练参数 ---
    training_args = TrainingArguments(
        output_dir="./outputs",
        per_device_train_batch_size=2,  # 根据GPU调整
        gradient_accumulation_steps=4,  # 有效batch size = 2 * 4 = 8
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        optim="paged_adamw_32bit",
        report_to="wandb",
    )

    # --- 6. 初始化Trainer ---
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
    )

    # --- 7. 开始训练 ---
    print("开始训练...")
    trainer.train()

    # --- 8. 保存适配器权重 ---
    output_dir = "./outputs/medical_lora_adapter"
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ 训练完成！适配器权重已保存到: {output_dir}")


if __name__ == "__main__":
    main()