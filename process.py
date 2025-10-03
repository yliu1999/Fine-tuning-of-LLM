# data/process.py
from datasets import load_dataset
import json


def process_medical_dataset():
    # 加载数据集
    print("正在加载数据集...")
    dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards")

    print("=== 数据集信息 ===")
    print(f"数据量: {len(dataset['train'])} 条")

    # 直接使用现有的指令格式
    formatted_data = []

    for i, item in enumerate(dataset['train']):
        # 数据集已经是标准格式，直接使用
        new_item = {
            "instruction": item['instruction'],
            "input": item['input'],
            "output": item['output']
        }
        formatted_data.append(new_item)

    # 保存为JSONL文件
    output_file = 'data/train.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✅ 成功处理 {len(formatted_data)} 条数据")
    print(f"💾 已保存到 {output_file}")

    # 显示样本作为验证
    print("\n=== 样本数据验证 ===")
    print("第一条数据:")
    print(f"Instruction: {formatted_data[0]['instruction']}")
    print(f"Input: {formatted_data[0]['input']}")
    print(f"Output: {formatted_data[0]['output']}")

    print(f"\n第二条数据:")
    print(f"Instruction: {formatted_data[1]['instruction']}")
    print(f"Input: {formatted_data[1]['input']}")
    print(f"Output: {formatted_data[1]['output']}")


if __name__ == "__main__":
    process_medical_dataset()