import gradio as gr

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent

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

    # 定义Multiply函数
    @tool
    def multiply(first_int: int, second_int: int) -> int:
        """Multiply two integers together."""
        print('-------------------------------------------------------------------------------------')
        return first_int * second_int
    
    @tool
    def write_qa_pair(question: str, answer: str) -> str:
        """
            When the user wants to save their response as a QA pair, 
            the Agent needs to identify the question and answer in the user's response, 
            then call this function and write it to the file.
            User Input Sample:
                [Customer question]Internal Dashboard有几个视图。
                [System answer]3个。
            Then call function write_qa_pair(question="Internal Dashboard有几个视图。", answer="3个。")
            
        """
        print('-------------------------------------------------------------------------------------')
        with open('qa_pairs.txt', 'a', encoding='utf-8') as file:
            file.write("0.\n")
            file.write(f"[客户问题] {question}\n")
            file.write(f"[系统回答] {answer}\n")
            file.write("\n")

        return "知识问答对补充成功。"
    
    tools = [multiply, write_qa_pair]
    prompt = hub.pull("hwchase17/openai-tools-agent")
    
    prompt.messages[0].prompt.template = '''
                                            你是SCCT(Supply chain control tower) Agent，帮助用户回答有关SCCT的相关知识问题。
                                            如果用户愿意补充你的qa_pairs.txt，那就调用函数写入我们的qa_pairs.txt中。
                                            如果没有准确答案，则回复'暂无相关参考'或者
                                            让用户帮助补充相关知识(例如帮我补充这个问答对到知识库中[Customer question]Internal Dashboard有几个视图。[System answer]3个。Agent解析出来的参数的名字只能是question与answer,且必须都有。)，不要自己编造。
                                            如果问的问题不与航运物流业相关，则引导用户问航运物流相关问题。
                                            )
                                            '''
    llm = ChatOpenAI(model_name="gpt-4", temperature=0, base_url="https://api.xiaoai.plus/v1", api_key="sk-InMcXCrwx83hEtui3d4242A6C7574aC397AdA0EcC07f56E4")
    agent = create_tool_calling_agent(llm, tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    userQuery = {
        "input":message
    }
    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if len(ans["source_documents"]) != 0:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    else:
        response = agent_executor.invoke(userQuery)
        print(f"[source_documents]{ans['source_documents']}")
        print(f"[llm_result]{response['output']}")
        return response['output']

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
    initialize_sales_bot()
    launch_gradio()
