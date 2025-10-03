# Fine-tuning-of-LLM
本项目基于Qwen2-1.5B模型，使用医疗问答数据进行了指令微调，并实现了本地API部署和Web界面。
# 医疗领域大模型微调与部署项目

## 项目概述
本项目基于Qwen2-1.5B模型，使用医疗问答数据进行了指令微调，并实现了本地API部署和Web界面。

## 技术栈
- **基础模型**: Qwen2-1.5B-Instruct
- **微调技术**: LoRA (PEFT)
- **训练框架**: Hugging Face Transformers + TRL
- **部署方案**: FastAPI + Gradio
- **评估方式**: 人工评估 + 对比测试

## 项目结构
project/
├── data/ # 训练数据
├── scripts/ # 训练和评估脚本
├── outputs/ # 训练输出
├── merged_medical_model/ # 合并后的模型
├── api/ # API服务代码
├── app/ # Web界面代码
└── README.md # 项目说明


## 快速开始
1. 启动API服务: `python api/transformers_server.py`
2. 启动Web界面: `python app/gradio_app.py`
3. 访问: http://localhost:7860
