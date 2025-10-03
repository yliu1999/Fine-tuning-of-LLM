# data/prepare_for_training.py
import json
from datasets import Dataset


def create_training_text(example):
    """将instruction, input, output组合成训练文本"""
    if example['input'].strip():
        text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    else:
        text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    return {"text": text}


def prepare_training_data():
    # 读取之前保存的JSONL文件
    with open('data/train.jsonl', 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    print(f"读取到 {len(data)} 条数据")

    # 创建Hugging Face Dataset
    dataset = Dataset.from_list(data)

    # 添加text字段
    dataset = dataset.map(create_training_text)

    # 查看处理后的样本
    print("\n=== 处理后的训练文本样本 ===")
    for i in range(2):
        print(f"\n--- 样本 {i + 1} ---")
        print(dataset[i]['text'])
        print("-" * 50)

    return dataset


if __name__ == "__main__":
    dataset = prepare_training_data()
    print(f"✅ 训练数据准备完成，共有 {len(dataset)} 条样本")