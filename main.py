from datasets import load_dataset
import json

# 加载数据集
dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards")

# 转换为指令格式
formatted_data = []
for item in dataset['train']:
    # 根据数据集的实际结构调整，目标是指令-输入-输出的格式
    new_item = {
        "instruction": "Based on medical knowledge, answer the following question.",
        "input": item['question'],
        "output": item['answer']
    }
    formatted_data.append(new_item)

# 保存为JSONL文件
with open('data/train.jsonl', 'w') as f:
    for item in formatted_data:
        f.write(json.dumps(item) + '\n')
print(f"Processed {len(formatted_data)} examples.")