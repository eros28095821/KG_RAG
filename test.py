import chainlit as cl
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
# 初始化模型
llm = OllamaLLM(model="kenneth85/llama-3-taiwan:8b-instruct")

# 儲存對話記憶
conversation_history = []
# 啟動時的歡迎訊息
@cl.on_chat_start
async def on_chat_start():
    global conversation_history
    conversation_history.clear()
    await cl.Message(content="您好！我是您的律師助手，有任何交通事故相關的法律問題請隨時告訴我！").send()

# 處理用戶訊息
@cl.on_message
async def on_message(message):
    global conversation_history

    user_input = message.content.strip()
    # 設置角色描述
    prompt = """
        您是一位專業律師，根據以下案件資料生成一份完整的民事交通事故起訴狀草稿：

        案件資料：
        {user_input}

        起訴狀應包括以下結構：
        一、事實緣由：
        詳述事故的發生經過，清楚指出被告的過失和責任。

        二、原告受傷情形：
        詳細描述原告的傷害和相關醫療情況。

        三、損害賠償：
        列出醫療費用、喪失工作所得等，計算總金額。

        四、引用法律條款：
        具體引用相關法條，並解釋其適用情況。

        五、請求項目：
        列出原告的具體請求，包括賠償金額及利息。
        
"""
    conversation_history.append({"role": "system", "content":prompt})
    # 檢查是否需要清除記憶
    if user_input.lower() in ["清除記憶", "reset"]:
        conversation_history.clear()
        conversation_history.append({"role": "system", "content": prompt})
        await cl.Message(content="記憶已清除！我仍然是一位律師助手，可以隨時為您提供幫助。").send()
        return

    try:
        # 保存用戶輸入到對話記憶中
        conversation_history.append({"role": "user", "content": user_input})

        # 組合對話記憶作為模型提示
        messages = [{"role": conv["role"], "content": conv["content"]} for conv in conversation_history]

        # 調用 LLM 生成回答
        result = llm.invoke(messages)

        # 處理返回結果
        if isinstance(result, str):
            assistant_reply = result
        else:
            assistant_reply = result.get("content", "抱歉，我無法生成回應。")

        # 保存模型回應到對話記憶中
        conversation_history.append({"role": "assistant", "content": assistant_reply})

        # 將回答發送給用戶
        await cl.Message(content=assistant_reply).send()

    except Exception as e:
        # 捕捉異常並提示用戶
        error_message = f"發生錯誤：{str(e)}"
        await cl.Message(content=error_message).send()
        print("錯誤詳情：", e)
