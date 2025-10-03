# test_quantization_dual_gpu.py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os


def setup_dual_gpu():
    """设置双GPU环境"""
    print("=== 双GPU环境检查 ===")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"GPU数量: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {prop.name}")
            print(f"  - 显存: {prop.total_memory / 1024 ** 3:.1f} GB")
            print(f"  - 计算能力: {prop.major}.{prop.minor}")

    return torch.cuda.device_count()


def test_gpu_quantization(gpu_id, model_name="Qwen/Qwen2-1.5B-Instruct"):
    """测试单个GPU的量化支持"""
    print(f"\n{'=' * 60}")
    print(f"测试 GPU {gpu_id}")
    print(f"{'=' * 60}")

    # 设置当前GPU
    torch.cuda.set_device(gpu_id)
    current_device = torch.cuda.current_device()
    print(f"当前使用 GPU: {current_device} - {torch.cuda.get_device_name(current_device)}")

    results = {}

    # 测试1: 不使用量化
    print(f"\n1. GPU{gpu_id} - 测试不使用量化...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=f"cuda:{gpu_id}",  # 指定GPU
            trust_remote_code=True
        )

        # 简单推理测试
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        test_prompt = "What is machine learning?"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results['no_quant'] = "成功"
        print(f"✅ GPU{gpu_id} - 不使用量化: 成功")
        print(f"   推理测试: {response[:100]}...")

        del model
        torch.cuda.empty_cache()

    except Exception as e:
        results['no_quant'] = f"失败: {str(e)[:100]}"
        print(f"❌ GPU{gpu_id} - 不使用量化失败: {e}")

    # 测试2: 8-bit量化
    print(f"\n2. GPU{gpu_id} - 测试8-bit量化...")
    try:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=f"cuda:{gpu_id}",
            trust_remote_code=True
        )

        results['8bit'] = "成功"
        print(f"✅ GPU{gpu_id} - 8-bit量化: 成功")

        # 显示模型设备信息
        if hasattr(model, 'hf_device_map'):
            print(f"   设备分布: {model.hf_device_map}")

        del model
        torch.cuda.empty_cache()

    except Exception as e:
        results['8bit'] = f"失败: {str(e)[:100]}"
        print(f"❌ GPU{gpu_id} - 8-bit量化失败: {e}")

    # 测试3: 4-bit量化
    print(f"\n3. GPU{gpu_id} - 测试4-bit量化...")
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=f"cuda:{gpu_id}",
            trust_remote_code=True
        )

        results['4bit'] = "成功"
        print(f"✅ GPU{gpu_id} - 4-bit量化: 成功")

        del model
        torch.cuda.empty_cache()

    except Exception as e:
        results['4bit'] = f"失败: {str(e)[:100]}"
        print(f"❌ GPU{gpu_id} - 4-bit量化失败: {e}")

    return results


def test_multi_gpu_quantization():
    """测试多GPU量化（模型并行）"""
    print(f"\n{'=' * 60}")
    print("测试多GPU模型并行")
    print(f"{'=' * 60}")

    model_name = "Qwen/Qwen2-1.5B-Instruct"

    try:
        # 使用auto设备映射，让Transformers自动分配模型到多个GPU
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",  # 自动分配到所有可用GPU
            trust_remote_code=True
        )

        print("✅ 多GPU自动分配: 成功")
        print(f"设备映射: {model.hf_device_map}")

        # 测试推理
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        test_prompt = "Explain the benefits of multi-GPU training."
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"多GPU推理测试: {response[:150]}...")

        del model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"❌ 多GPU自动分配失败: {e}")


def test_memory_usage():
    """测试GPU显存使用情况"""
    print(f"\n{'=' * 60}")
    print("GPU显存使用测试")
    print(f"{'=' * 60}")

    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
        total = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3

        print(f"GPU {i}:")
        print(f"  已分配: {allocated:.2f} GB")
        print(f"  已保留: {reserved:.2f} GB")
        print(f"  总显存: {total:.2f} GB")
        print(f"  使用率: {(allocated / total) * 100:.1f}%")


def main():
    # 设置环境
    gpu_count = setup_dual_gpu()

    if gpu_count < 2:
        print("❌ 检测到的GPU数量少于2个，无法进行双GPU测试")
        return

    # 清空GPU缓存
    torch.cuda.empty_cache()

    # 分别测试每个GPU
    all_results = {}

    for gpu_id in range(2):  # 测试GPU 0 和 GPU 1
        all_results[gpu_id] = test_gpu_quantization(gpu_id)

    # 测试多GPU并行
    test_multi_gpu_quantization()

    # 测试显存使用
    test_memory_usage()

    # 打印汇总结果
    print(f"\n{'=' * 60}")
    print("测试结果汇总")
    print(f"{'=' * 60}")

    for gpu_id, results in all_results.items():
        print(f"\nGPU {gpu_id} 结果:")
        for quant_type, result in results.items():
            status = "✅" if "成功" in result else "❌"
            print(f"  {status} {quant_type}: {result}")

    # 给出建议
    print(f"\n{'=' * 60}")
    print("使用建议")
    print(f"{'=' * 60}")

    working_gpus = []
    for gpu_id, results in all_results.items():
        if "成功" in results.get('no_quant', ''):
            working_gpus.append(gpu_id)

    if working_gpus:
        print(f"✅ 可用的GPU: {working_gpus}")
        if len(working_gpus) == 2:
            print("建议使用多GPU训练以获得更好性能")
        else:
            print(f"建议使用 GPU {working_gpus[0]} 进行训练")
    else:
        print("❌ 没有可用的GPU，请检查CUDA安装")


if __name__ == "__main__":
    main()