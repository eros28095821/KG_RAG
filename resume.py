from operator import itemgetter
import os
import ollama
from dotenv import load_dotenv

from chainlit.types import ThreadDict
import chainlit as cl

load_dotenv()

@cl.password_auth_callback
def auth():
    return cl.User(identifier="test")

# 新對話開始時，檢查是否有舊對話，如果有則載入
@cl.on_chat_start
async def on_chat_start():
    # 檢查是否有保存的對話歷史
    chat_history = cl.user_session.get("chat_history")
    if chat_history is None:
        # 如果沒有舊對話，初始化一個空列表
        cl.user_session.set("chat_history", [])
    else:
        # 如果有舊對話，將其載入
        cl.user_session.set("chat_history", chat_history)

# 恢復聊天時將歷史消息添加到聊天記錄中
@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    chat_history = cl.user_session.get("chat_history")
    
    # 清空舊對話
    cl.user_session.set("chat_history", [])
    
    # 根據傳入的歷史消息更新對話
    for message in thread["steps"]:
        if message["type"] == "user_message":
            chat_history.append({"role": "user", "content": message["output"]})
        elif message["type"] == "assistant_message":
            chat_history.append({"role": "assistant", "content": message["output"]})

    # 更新 session
    cl.user_session.set("chat_history", chat_history)

# 處理來自用戶的消息，並生成對應的回應
@cl.on_message
async def on_message(message: cl.Message):
    chat_history = cl.user_session.get("chat_history")

    # 使用 Ollama 模型進行聊天
    model = "kenneth85/llama-3-taiwan:8b-instruct"  # 可以根據你的需求更換 Ollama 模型名稱
    client = ollama

    chat_history.append({"role": "user", "content": message.content})

    try:
        # 發送聊天消息給 Ollama API
        chat_response = client.chat(
            model=model,
            messages=chat_history  # 傳遞聊天歷史
        )

        # 打印 chat_response 來檢查其結構
        print("Chat response:", chat_response)

        # 從 chat_response 提取 assistant 回應
        if 'message' in chat_response:
            response_content = chat_response['message']['content']  # 獲取 assistant 的回應
        else:
            response_content = "I couldn't understand the response structure."

        # 將回應加入聊天歷史
        chat_history.append({"role": "assistant", "content": response_content})

        # 更新 session 中的聊天歷史
        cl.user_session.set("chat_history", chat_history)

        # 發送回應給用戶
        await cl.Message(content=response_content).send()

    except Exception as e:
        # 錯誤處理
        await cl.Message(content=f"Error: {str(e)}").send()

