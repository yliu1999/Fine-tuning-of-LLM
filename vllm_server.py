# api/vllm_server.py
from vllm import SamplingParams
from vllm import LLM
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="医疗问答大模型API")


class QueryRequest(BaseModel):
    question: str
    max_tokens: int = 200
    temperature: float = 0.7


# 加载模型
print("正在加载模型...")
llm = LLM(
    model="./merged_medical_model",
    trust_remote_code=True,
    max_model_len=1024
)


@app.post("/ask")
async def ask_question(request: QueryRequest):
    # 构造提示词
    prompt = f"基于医学知识，回答以下问题：\n\n问题：{request.question}\n回答："

    # 设置生成参数
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )

    # 生成回答
    outputs = llm.generate([prompt], sampling_params)
    response = outputs[0].outputs[0].text

    return {
        "question": request.question,
        "answer": response.strip(),
        "model": "Qwen2-1.5B-Medical"
    }


@app.get("/")
async def root():
    return {"message": "医疗问答大模型API服务运行中"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)