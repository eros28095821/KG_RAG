from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
from fpdf import FPDF
import torch
import numpy as np
import os
import gradio as gr
import tempfile

# 加載 .env 文件中的環境變數
load_dotenv()

# Neo4j 配置
uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(uri, auth=(username, password))

# 初始化模型
model = SentenceTransformer('all-MiniLM-L6-v2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

llm = OllamaLLM(model="kenneth85/llama-3-taiwan:8b-instruct", temperature=0.1, num_predict=5000)

# 保存歷史記錄
history = []


def get_similar_facts_with_statutes(input_text, top_k=3):
    input_embedding = model.encode(input_text)
    input_embedding = torch.tensor(input_embedding, dtype=torch.float32).to(device)

    with driver.session() as session:
        results = session.run("MATCH (f:Fact) RETURN f.id AS id, f.text AS text, f.embedding AS embedding")
        fact_ids, fact_texts, embeddings = [], [], []

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
        all_statutes = set()

        for i in top_indices:
            fact_id = fact_ids[i]
            fact_text = fact_texts[i]
            similarity = similarities[i]
            statutes_for_fact = get_statutes_for_case(fact_id)

            current_statutes = [statute_id for statute in statutes_for_fact for statute_id in statute['statutes']]
            all_statutes.update(current_statutes)

            similar_cases.append({
                "id": fact_id,
                "content": fact_text,
                "similarity": similarity,
                "statutes": ", ".join(current_statutes)
            })

        return similar_cases, all_statutes


def get_statutes_for_case(fact_id):
    with driver.session() as session:
        results = session.run("""
            MATCH (c:Case)-[:案件事實]->(f:Fact {id: $fact_id})
            MATCH (c)-[:案件相關法條]->(l:LegalReference)
            MATCH (l)-[:引用法條]->(s:Statute)
            RETURN c.id AS case_id, collect(s.id) AS statutes
            """, fact_id=fact_id)
        return [{"case_id": record["case_id"], "statutes": record["statutes"]} for record in results]


def generate_step_one(input_fact, top_k):
    try:
        similar_cases, all_statutes = get_similar_facts_with_statutes(input_fact, top_k=top_k)
        statutes_text = ", ".join(all_statutes)

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
        result = llm.invoke([{"role": "user", "content": prompt}])
        indictment_text = result if isinstance(result, str) else result.get("content", "無法生成起訴書內容。")

        history.append({
            "indictment_text": indictment_text,
            "cases": similar_cases
        })

        cases_summary = "\n".join([f"案例 {i+1}: {case['content'][:30]}..." for i, case in enumerate(similar_cases)])
        return statutes_text, indictment_text, cases_summary
    except Exception as e:
        return f"發生錯誤：{e}", "", ""


def view_case_detail_step_two(case_number):
    try:
        if case_number is None:
            return "請輸入有效的案例編號"

        index = int(case_number) - 1
        if index < 0 or index >= len(history[-1]["cases"]):
            return "無效的案例編號"

        case = history[-1]["cases"][index]
        return f"""
        事實 ID: {case['id']}
        事實內容: {case['content']}
        相似度: {case['similarity']:.6f}
        引用的法條: {case['statutes']}
        """
    except (ValueError, IndexError):
        return "無效的案例選擇"

def generate_pdf(indictment_text):
    """
    生成 PDF 文件。
    """
    try:
        # 指定字體的本地路徑
        font_path = "/home/chen/KG_RAG/NotoSansCJK-Regular.ttc"

        pdf = FPDF()
        pdf.add_page()

        # 加載字體
        pdf.add_font("NotoSans", fname=font_path, uni=True)
        pdf.set_font("NotoSans", size=12)

        # 添加內容
        for line in indictment_text.split("\n"):
            pdf.multi_cell(w=0, h=10, txt=line, ln=True, align="L")

        # 保存為臨時文件
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf.output(temp_file.name)
        return temp_file.name
    except Exception as e:
        return f"無法生成 PDF: {e}"



def update_history_dropdown():
    return gr.update(choices=[f"記錄 {i+1}" for i in range(len(history))])


def view_history(selected_index):
    try:
        if not selected_index:
            return "請選擇一條歷史記錄", ""

        record_index = int(selected_index.split(" ")[1]) - 1
        record = history[record_index]
        return record["indictment_text"], "\n".join([f"案例 {i+1}: {case['content'][:30]}..." for i, case in enumerate(record["cases"])])
    except (ValueError, IndexError):
        return "無效的歷史記錄選擇", ""


with gr.Blocks() as gradio_interface:
    gr.Markdown("## 起訴書生成器")

    input_fact = gr.Textbox(label="案件描述")
    top_k = gr.Slider(minimum=1, maximum=10, step=1, value=3, label="相似案例數量")
    generate_button = gr.Button("生成起訴書與案例")
    statutes_text = gr.Textbox(label="引用的法條集合", interactive=False)
    indictment_text = gr.Textbox(label="生成的起訴書", interactive=False)
    cases_summary = gr.Textbox(label="相似案例摘要", interactive=False)

    case_number = gr.Number(label="輸入案例編號", value=1)
    view_button = gr.Button("查看案例詳細信息")
    case_detail = gr.Textbox(label="案例詳細信息", interactive=False)

    history_dropdown = gr.Dropdown(label="查看歷史記錄", choices=[], interactive=True)
    history_button = gr.Button("查看歷史記錄")
    history_indictment = gr.Textbox(label="歷史起訴書", interactive=False)
    history_cases = gr.Textbox(label="歷史相似案例摘要", interactive=False)

    pdf_button = gr.Button("下載起訴書 PDF")
    pdf_output = gr.File(label="下載 PDF")

    generate_button.click(
        generate_step_one,
        inputs=[input_fact, top_k],
        outputs=[statutes_text, indictment_text, cases_summary]
    )
    generate_button.click(
        update_history_dropdown,
        outputs=[history_dropdown]
    )
    view_button.click(
        view_case_detail_step_two,
        inputs=[case_number],
        outputs=[case_detail]
    )
    pdf_button.click(
        generate_pdf,
        inputs=[indictment_text],
        outputs=[pdf_output]
    )
    history_button.click(
        view_history,
        inputs=[history_dropdown],
        outputs=[history_indictment, history_cases]
    )

if __name__ == "__main__":
    gradio_interface.launch()
