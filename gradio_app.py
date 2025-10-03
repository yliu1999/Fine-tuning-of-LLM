import gradio as gr
import requests
import time

# API配置
API_URL = "http://localhost:8000/ask"


def mock_response(message):
    """模拟后端响应，用于测试"""
    mock_responses = {
        "糖尿病的典型症状有哪些？": "糖尿病的典型症状包括多饮、多食、多尿和体重减轻，即'三多一少'症状。此外还可能出现疲劳、视力模糊等症状。",
        "如何有效预防高血压？": "预防高血压应保持健康饮食，减少盐的摄入，坚持适量运动，保持健康体重，戒烟限酒，保持良好的心态和充足的睡眠。",
        "普通感冒和流感的区别是什么？": "普通感冒症状较轻，主要表现为鼻塞、流涕、咽痛等上呼吸道症状；流感症状较重，常伴有高热、全身酸痛、乏力等全身症状，传染性也更强。"
    }
    # 如果没有匹配的预设问题，返回通用回答
    return mock_responses.get(message,
                              f"这是模拟的回答。你问的是：{message}\n\n注意：当前处于离线模式，实际使用时请启动后端服务。")


# def chat_with_model(message, history):
#     """与医疗模型对话：返回符合Gradio Chatbot要求的格式"""
#     # 初始化历史对话（若为空则创建空列表）
#     history = history or []
#     try:
#         # 1. 调用后端API
#         response = requests.post(API_URL, json={
#             "question": message,  # 已修正为正确的参数名
#             "max_tokens": 300,
#             "temperature": 0.7
#         })
#
#         # 2. 处理API响应
#         if response.status_code == 200:
#             result = response.json()
#             model_answer = result["answer"]  # 获取模型回答
#             # 3. 更新对话历史：在原有历史中添加“(用户问题, 模型回答)”
#             history.append((message, model_answer))
#             # 4. 返回更新后的历史（符合Chatbot格式要求）
#             return history
#         else:
#             # 若API请求失败，也添加错误信息到历史
#             error_msg = f"API请求失败: {response.status_code}"
#             history.append((message, error_msg))
#             return history
#
#     except Exception as e:
#         # 若发生异常，添加异常信息到历史
#         error_msg = f"发生错误: {str(e)}"
#         history.append((message, error_msg))
#         return history


def chat_with_model(message, history, temperature, max_tokens, offline_mode):
    """与医疗模型对话，支持离线模式"""
    if not message.strip():
        return history, "❌ 问题不能为空", ""

    # 添加用户消息到历史
    history.append([message, ""])

    try:
        if offline_mode:
            # 使用模拟响应
            answer = mock_response(message)
            # 模拟网络延迟
            time.sleep(1)
            history[-1][1] = answer
            return history, "✅ 离线模式 - 回答完成", ""
        else:
            # 调用API
            response = requests.post(API_URL, json={
                "question": message,
                "max_tokens": max_tokens,
                "temperature": temperature
            }, timeout=60)

            if response.status_code == 200:
                result = response.json()
                answer = result["answer"]
                history[-1][1] = answer
                return history, "✅ 回答完成", ""
            else:
                history[-1][1] = f"❌ API请求失败: {response.status_code}"
                return history, "❌ 请求失败", ""

    except Exception as e:
        error_msg = f"❌ 发生错误: {str(e)}"
        history[-1][1] = error_msg
        # 特别提示后端未启动的情况
        if "Connection refused" in str(e) or "Failed to connect" in str(e):
            return history, "❌ 后端服务未启动，请确保后端服务运行在8000端口", ""
        return history, "❌ 错误", ""


def clear_chat():
    """清空聊天记录"""
    return [], "聊天记录已清空", ""


def create_fixed_interface():
    """创建修复后的界面"""
    with gr.Blocks(title="智能医疗助手", theme=gr.themes.Soft()) as demo:

        # 头部区域
        gr.Markdown("""
        # 🏥 智能医疗问答助手
        **基于大语言模型的专业医疗咨询平台**
        """)

        with gr.Row():
            # 左侧聊天区域
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="💬 医疗对话",
                    height=400,
                    show_copy_button=True
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="📝 输入您的问题",
                        placeholder="请输入医疗相关问题... 按Ctrl+Enter发送，或点击发送按钮",
                        lines=2,
                        max_lines=4,
                        scale=4
                    )
                    submit_btn = gr.Button("🚀 发送", size="lg", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("🗑️ 清空对话")

            # 右侧控制面板
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ 参数设置")

                # 添加离线模式开关
                offline_mode = gr.Checkbox(label="离线模式", value=False,
                                           info="启用后不连接后端，使用模拟数据")

                temperature = gr.Slider(0.1, 1.0, value=0.7, label="创造性")
                max_tokens = gr.Slider(50, 500, value=300, step=50, label="回答长度")

                gr.Markdown("### 💡 示例问题")
                example_questions = [
                    "糖尿病的典型症状有哪些？",
                    "如何有效预防高血压？",
                    "普通感冒和流感的区别是什么？"
                ]

                for question in example_questions:
                    btn = gr.Button(question, size="sm")
                    btn.click(lambda q=question: q, outputs=[msg])

                status = gr.Textbox(label="状态", value="🟢 系统就绪", interactive=False)

        # 正确的交互逻辑
        def submit_message(message, history, temp, tokens, offline):
            if not message.strip():
                return history, "❌ 问题不能为空", ""
            return chat_with_model(message, history, temp, tokens, offline)

        # 按钮点击事件
        submit_btn.click(
            fn=submit_message,
            inputs=[msg, chatbot, temperature, max_tokens, offline_mode],
            outputs=[chatbot, status, msg]
        )

        # 文本框提交事件（使用Ctrl+Enter）
        msg.submit(
            fn=submit_message,
            inputs=[msg, chatbot, temperature, max_tokens, offline_mode],
            outputs=[chatbot, status, msg]
        )

        # 清空按钮
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, status, msg]
        )

        # 离线模式切换提示
        def update_status(offline):
            if offline:
                return "🟡 离线模式 - 不连接后端服务"
            else:
                return "🟢 在线模式 - 准备连接后端服务"

        offline_mode.change(
            fn=update_status,
            inputs=[offline_mode],
            outputs=[status]
        )

    return demo


if __name__ == "__main__":
    demo = create_fixed_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)
