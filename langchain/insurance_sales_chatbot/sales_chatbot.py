import gradio as gr

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS


def initialize_sales_bot(vector_store_dir: str="real_scct_agent_test"):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(api_key="sk-InMcXCrwx83hEtui3d4242A6C7574aC397AdA0EcC07f56E4", base_url="https://api.xiaoai.plus/v1"), allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-4", temperature=0, api_key="sk-InMcXCrwx83hEtui3d4242A6C7574aC397AdA0EcC07f56E4", base_url="https://api.xiaoai.plus/v1")
    
    global SALES_BOT    
    SALES_BOT = RetrievalQA.from_chain_type(llm,
                                           retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                     search_kwargs={"score_threshold": 0.8}))
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT

def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True
    llm = ChatOpenAI(model_name="gpt-4", temperature=1, base_url="https://api.xiaoai.plus/v1", api_key="sk-InMcXCrwx83hEtui3d4242A6C7574aC397AdA0EcC07f56E4")
    messages = [
        ("system", "你是SCCT Agent，帮助用户回答有关SCCT的相关知识问题。如果没有准确答案，则回复'暂无相关参考'，不要自己编造。如果问的问题不与航运物流业相关，则引导用户问航运物流相关问题。"),
        ("human", message),
    ]
    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if len(ans["source_documents"]) != 0:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    elif enable_chat:
        response = llm.invoke(messages)
        print(f"[source_documents]{ans['source_documents']}")
        print(f"[llm_result]{response.content}")
        return response.content
    # 否则输出套路话术
    else:
        return "这个问题我要问问领导"

def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="SCCT Agent",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    # 初始化保险销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
