# scripts/evaluate_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


def compare_models():
    # 测试问题
    test_questions = [
        "What are the symptoms of diabetes?",
        "How to treat a common cold?",
        "What is hypertension?",
        "Explain the causes of heart attack."
    ]

    # 原始模型
    print("=== 原始模型回答 ===")
    original_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-1.5B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    original_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct", trust_remote_code=True)

    # 微调后模型
    print("\n=== 微调后模型回答 ===")
    fine_tuned_model = AutoModelForCausalLM.from_pretrained(
        "./merged_medical_model",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    fine_tuned_tokenizer = AutoTokenizer.from_pretrained("./merged_medical_model", trust_remote_code=True)

    for i, question in enumerate(test_questions, 1):
        print(f"\n--- 问题 {i}: {question} ---")

        # 原始模型回答
        inputs = original_tokenizer(question, return_tensors="pt").to(original_model.device)
        with torch.no_grad():
            outputs = original_model.generate(**inputs, max_new_tokens=150, temperature=0.7)
        original_answer = original_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"原始模型: {original_answer[len(question):][:200]}...")

        # 微调后模型回答
        inputs = fine_tuned_tokenizer(question, return_tensors="pt").to(fine_tuned_model.device)
        with torch.no_grad():
            outputs = fine_tuned_model.generate(**inputs, max_new_tokens=150, temperature=0.7)
        fine_tuned_answer = fine_tuned_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"微调后: {fine_tuned_answer[len(question):][:200]}...")


if __name__ == "__main__":
    compare_models()