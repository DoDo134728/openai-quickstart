import gradio as gr

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from io import BytesIO
from openai import OpenAI
import base64
import requests
from PIL import Image

api_key = "sk-InMcXCrwx83hEtui3d4242A6C7574aC397AdA0EcC07f56E4"
base_url = "https://api.xiaoai.plus/v1"

def initialize_sales_bot(vector_store_dir: str="real_scct_agent_test"):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(api_key=api_key, base_url=base_url), allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-4", temperature=0, api_key=api_key, base_url=base_url)
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
    
    tools = [write_qa_pair]
    prompt = hub.pull("hwchase17/openai-tools-agent")
    
    prompt.messages[0].prompt.template = '''
                                            你是SCCT(Supply chain control tower) Agent，帮助用户回答有关SCCT的相关知识问题。
                                            如果用户愿意补充你的qa_pairs.txt，那就调用函数写入我们的qa_pairs.txt中。
                                            如果没有准确答案，则回复'暂无相关参考'或者
                                            让用户帮助补充相关知识(例如帮我补充这个问答对到知识库中[Customer question]Internal Dashboard有几个视图。[System answer]3个。Agent解析出来的参数的名字只能是question与answer,且必须都有。)，不要自己编造。
                                            如果问的问题不与航运物流业相关，则引导用户问航运物流相关问题。
                                            )
                                            '''
    llm = ChatOpenAI(model_name="gpt-4", temperature=0, api_key=api_key, base_url=base_url)
    agent = create_tool_calling_agent(llm, tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    userQuery = {
        "input":message
    }
    ans = SALES_BOT({"query": message})
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

def view_image(img):
    prompForDetactFunction = """
            任务描述：
                你将收到一张系统功能的截图。
                你的任务是识别图像中的功能点。
            输出格式：
                '输入用户名','登录','提交表单'
            注意事项：
                确保所有功能都已被识别到。
                如果图像中有多个功能点, 请参考输出格式列出。
            """
    
    client = OpenAI(
        base_url="https://api.xiaoai.plus/v1",
        api_key="sk-InMcXCrwx83hEtui3d4242A6C7574aC397AdA0EcC07f56E4"
    ) 
    
    pil_image = Image.fromarray(img)

    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # 构造请求的 HTTP Header
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {client.api_key}"
    }

    # 构造请求的负载
    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompForDetactFunction},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ],
        "max_tokens": 3000,
        "temperature": 0
    }

    # 发送 HTTP 请求
    response = requests.post("https://api.xiaoai.plus/v1/chat/completions", headers=headers, json=payload)
    
    # 检查响应并提取所需的 content 字段
    if response.status_code == 200:
        response_data = response.json()
        content = response_data['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code}, {response.text}"
    
    db = FAISS.load_local("real_scct_agent_test", OpenAIEmbeddings(api_key="sk-InMcXCrwx83hEtui3d4242A6C7574aC397AdA0EcC07f56E4", base_url="https://api.xiaoai.plus/v1"), allow_dangerous_deserialization=True)
    topK_retriever = db.as_retriever(search_kwargs={"k": 3, "score_threshold": 0.6})
    source_documents = ''
    for i in content.split(','):
        docs = topK_retriever.get_relevant_documents(i)
        for doc in docs:
            source_documents = source_documents + doc.page_content + "\n"
    prompt = """
            任务描述：
                你将收到一张系统功能的截图和相关的后端系统逻辑。
                后端逻辑会以[source_documents]拼接在本Prompt最后。
                你的任务是识别图像中的功能点，并为每个功能点编写测试点和验收标准（Acceptance Criteria），而且前段UI和后端系统逻辑都要参考，不能只看前端UI。
            输出格式：
                每个功能点的测试点和验收标准应按照以下格式编写：
                Given [前提条件]
                When [执行的操作]
                Then [预期结果]

            示例：
                功能点：登录按钮
                Given 用户在登录页面
                When 用户点击登录按钮
                Then 系统应验证用户凭证并重定向到主页

            注意事项：
                确保所有测试点和验收标准都清晰、具体且可测试。
                如果图像中有多个功能点，请分别列出每个功能点的测试点和验收标准。
            """

    promptForAC = prompt + '\n' + source_documents
    print(promptForAC)
    payloadForAC = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": promptForAC},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ],
        "max_tokens": 3000,
        "temperature": 0
    }
    
    response = requests.post("https://api.xiaoai.plus/v1/chat/completions", headers=headers, json=payloadForAC)
    
    # 检查响应并提取所需的 content 字段
    if response.status_code == 200:
        response_data = response.json()
        content = response_data['choices'][0]['message']['content']
        return content
    else:
        return f"Error: {response.status_code}, {response.text}"
    

def launch_gradio():

    app_one = gr.ChatInterface(
        fn=sales_chat,
        title="SCCT Agent",
        # retry_btn=None,
        # # # undo_btn=None,
        chatbot=gr.Chatbot(height=600))
    
    app_two = gr.Interface(fn = view_image, inputs="image", outputs="text")

    demo = gr.TabbedInterface(
        [
            app_one, app_two
        ],
        tab_names=["SCCT Agent", "SCCT Image Agent"]
    )
    

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    initialize_sales_bot()
    launch_gradio()
