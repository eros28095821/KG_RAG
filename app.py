import chainlit as cl
from langchain_ollama import OllamaLLM

# 初始化模型
llm = OllamaLLM(model="kenneth85/llama-3-taiwan:8b-instruct")

# 儲存對話記憶和資料追蹤
conversation_history = []
required_fields = {
    "事故經過": None,
    "受傷情形": None,
    "損害賠償": None,
}
collecting_data = True  # 狀態：是否處於資料蒐集階段

# 啟動時的歡迎訊息
@cl.on_chat_start
async def on_chat_start():
    global conversation_history, required_fields, collecting_data
    conversation_history.clear()
    collecting_data = True
    # 初始化必填項目
    for key in required_fields.keys():
        required_fields[key] = None
    await cl.Message(content="您好！我是您的律師助手，讓我們一起蒐集案件資料以生成交通事故起訴狀草稿。請回答我的問題以提供關鍵資料。").send()

# 處理用戶訊息
@cl.on_message
async def on_message(message):
    global conversation_history, required_fields, collecting_data

    user_input = message.content.strip()

    # 若用戶請求生成起訴狀
    if user_input.lower() in ["生成起訴狀", "generate"]:
        missing_fields = [key for key, value in required_fields.items() if not value]
        if missing_fields:
            # 提示缺失的資料項目，但不阻止生成
            missing_message = f"注意：以下資料尚未提供，建議補充以生成更完整的起訴狀：\n" + "\n".join(missing_fields)
            await cl.Message(content=missing_message).send()

        # 組合案件資料生成起訴狀
        collected_data = "\n".join(
            f"{key}：{value}" for key, value in required_fields.items() if value
        )
        prompt = f"""
            您是一位專業律師，根據以下案件資料生成一份完整的民事交通事故起訴狀草稿：

            案件資料：
            {collected_data}

            起訴狀應包括以下結構：
            一、事實緣由：
            詳述事故的發生經過，清楚指出被告的過失和責任。

            二、原告受傷情形：
            詳細描述原告的傷害和相關醫療情況。

            三、損害賠償：
            列出醫療費用、喪失工作所得等，計算總金額。

            四、引用法律條款：
            具體引用相關法條，並詳細解釋其與案件事實的聯繫，以及適用的理由。

            五、請求項目：
            列出原告的具體請求，包括賠償金額及利息。
        """
        try:
            # 調用 LLM 生成起訴狀
            result = llm.invoke([{"role": "system", "content": prompt}])
            if isinstance(result, str):
                assistant_reply = result
            else:
                assistant_reply = result.get("content", "抱歉，我無法生成起訴狀。")

            await cl.Message(content=assistant_reply).send()

        except Exception as e:
            error_message = f"發生錯誤：{str(e)}"
            await cl.Message(content=error_message).send()
            print("錯誤詳情：", e)

        return

    # 資料蒐集階段的提問和記錄
    prompt = """
        您是一位專業律師，你善於用問題引導民眾提供所需的資料。
    """
    try:
        # 根據關鍵詞將用戶輸入存儲到特定欄位
        if "事故" in user_input and not required_fields["事故經過"]:
            required_fields["事故經過"] = user_input
        elif "受傷" in user_input and not required_fields["受傷情形"]:
            required_fields["受傷情形"] = user_input
        elif "賠償" in user_input and not required_fields["損害賠償"]:
            required_fields["損害賠償"] = user_input

        # 更新對話記憶（僅保留問題回答的邏輯，不存整段輸入到 case_data）
        conversation_history.append({"role": "user", "content": user_input})

        # 生成律師助手的回應
        result = llm.invoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}
        ])

        if isinstance(result, str):
            assistant_reply = result
        else:
            assistant_reply = result.get("content", "抱歉，我無法生成提問。")

        # 儲存助手回應到記憶中
        conversation_history.append({"role": "assistant", "content": assistant_reply})

        # 發送助手回應
        await cl.Message(content=assistant_reply).send()

        # 顯示目前蒐集到的案件資料和缺失項目
        collected_data_preview = "\n".join(
            f"{key}：{value}" for key, value in required_fields.items() if value
        )
        await cl.Message(content="目前蒐集到的案件資料如下：\n" + (collected_data_preview or "尚無資料")).send()

        missing_fields = [key for key, value in required_fields.items() if not value]
        if missing_fields:
            await cl.Message(content="以下資料仍未提供：\n" + "\n".join(missing_fields)).send()

    except Exception as e:
        # 捕捉異常並提示用戶
        error_message = f"發生錯誤：{str(e)}"
        await cl.Message(content=error_message).send()
        print("錯誤詳情：", e)
