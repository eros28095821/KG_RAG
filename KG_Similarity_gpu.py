from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
import torch
import numpy as np
import os
import sys
# 加載 .env 文件中的環境變數
load_dotenv()

# Neo4j 配置
uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(uri, auth=(username, password))

# 初始化 SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化 LLM
llm = OllamaLLM(model="kenneth85/llama-3-taiwan:8b-instruct",
                temperature=0.1,
                num_predict=5000 )

# Neo4j 查詢函數
def get_similar_facts_with_statutes(input_text, top_k=3):
    input_embedding = model.encode(input_text)
    input_embedding = torch.tensor(input_embedding, dtype=torch.float32).to(device)

    with driver.session() as session:
        results = session.run("MATCH (f:Fact) RETURN f.id AS id, f.text AS text, f.embedding AS embedding")
        fact_ids = []
        fact_texts = []
        embeddings = []

        for record in results:
            fact_ids.append(record["id"])
            fact_texts.append(record["text"])
            embeddings.append(torch.tensor(np.array(record["embedding"], dtype=np.float32)).to(device))

        embeddings = torch.stack(embeddings)

        input_norm = input_embedding / input_embedding.norm()
        embeddings_norm = embeddings / embeddings.norm(dim=1, keepdim=True)
        similarities = torch.matmul(embeddings_norm, input_norm.T).cpu().numpy()

        top_indices = similarities.argsort()[-top_k:][::-1]

        similar_cases = []
        all_statutes = set()  # 用於收集所有案件的法條

        for i in top_indices:
            fact_id = fact_ids[i]
            fact_text = fact_texts[i]
            similarity = similarities[i]
            statutes_for_fact = get_statutes_for_case(fact_id)

            # 收集當前案件的法條
            current_statutes = [
                statute_id for statute in statutes_for_fact for statute_id in statute['statutes']
            ]
            all_statutes.update(current_statutes)  # 合併到總集合中

            similar_cases.append({
                "id": fact_id,
                "content": fact_text,
                "similarity": similarity,
                "statutes": ", ".join(current_statutes)  # 單筆案件法條
            })

        return similar_cases, all_statutes  # 返回所有案件的集合


def get_statutes_for_case(fact_id):
    with driver.session() as session:
        results = session.run(
            """
            MATCH (c:Case)-[:案件事實]->(f:Fact {id: $fact_id})
            MATCH (c)-[:案件相關法條]->(l:LegalReference)
            MATCH (l)-[:引用法條]->(s:Statute)
            RETURN c.id AS case_id, collect(s.id) AS statutes
            """,
            fact_id=fact_id
        )
        return [{"case_id": record["case_id"], "statutes": record["statutes"]} for record in results]


def main():
    print("您好！我將引導您生成起訴書，請描述案件細節。")

    while True:
        print("請輸入案件描述，結束輸入請按 Ctrl+D（或輸入空行並按 Enter）：")
        input_fact = sys.stdin.read()  # 讀取多行直到結束

        try:
            # 使用 Neo4j 查詢相似案例和法條集合
            similar_cases, all_statutes = get_similar_facts_with_statutes(input_fact)

            # 將所有法條集合轉換為字符串
            statutes_text = ", ".join(all_statutes)
            
            # 組織最接近的三筆案例
            cases_output = "\n\n".join([ 
                f"第{i+1}相似\n"
                f"事實 ID: {case['id']}\n"
                f"事實內容: {case['content']}\n"
                f"相似度: {case['similarity']:.6f}\n"
                f"引用的法條: {case['statutes']}"
                for i, case in enumerate(similar_cases)
            ])

            # 構建提示詞
            prompt = f"""
            以下是參考的法律條款：
            {statutes_text}

            您是一位專業律師，根據以下案件資料生成一份完整的民事交通事故起訴狀草稿：

            案件資料：
            {input_fact}

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

            # 調用 LLM 生成起訴狀
            result = llm.invoke([{"role": "user", "content": prompt}])

            # 如果結果是字符串，直接使用
            if isinstance(result, str):
                generated_text = result
            else:
                # 如果是對象，處理 `content` 屬性
                generated_text = result.get("content", "無法生成起訴書內容。")

            # 返回生成的起訴書和相似案例
            print(statutes_text)
            print(f"\n以下是生成的起訴書：\n\n{generated_text}")
            print(f"\n以下是最接近的三筆案例：\n\n{cases_output}")

            # 自動清除案件描述
            input_fact = ""  # 清除 input_fact
        except Exception as e:
            print(f"發生錯誤：{e}")
if __name__ == "__main__":  
    main()
