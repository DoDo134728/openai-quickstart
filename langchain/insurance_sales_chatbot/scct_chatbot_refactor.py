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
from langchain_community.chat_message_histories import ChatMessageHistory

class vectorDB():
    def __init__(self, vector_store_dir) -> None:
        self.db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(api_key=api_key, base_url=base_url), allow_dangerous_deserialization=True)

class SCCTAgent():
    def __init__(self, api_key, base_url, vector_db) -> None:
        self.db = vector_db
        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0, api_key=api_key, base_url=base_url)
        self.retrievalQA = RetrievalQA.from_chain_type(self.llm, 
                                                       retriever=self.db.as_retriever(search_type="similarity_score_threshold",
                                                                                      search_kwargs={"score_threshold": 0.8}))
        self.retrievalQA.return_source_documents = True
        self.memory = ChatMessageHistory()
        self.construct_agent_prompt()
        self.initialize_scct_agent()
        
    def scct_bot_tool(self, input_text: str):
        """
            Use this tool to query the knowledge database and answer user questions related to shipping or SCCT.
    
            Parameters: 
                self (str):
                input_text (str): The user's question.

            Returns:
                str: A comprehensive answer based on the query results. If no relevant results are found, returns 'No relevant reference available'.
    
            Note:
                If no relevant results are found, honestly inform the user and do not fabricate answers.
        """
        return self.retrievalQA({"query": input_text})['source_documents']
    
    def write_qa_pair(self, question: str, answer: str) -> str:
        """
            This function saves a question-answer (QA) pair to a file.
            When the user wants to save their response as a QA pair, 
            the Agent needs to identify the question and answer in the user's response, 
            then call this function and write it to the file.
            
            User Input Sample:
                [Customer question] Internal Dashboard有几个视图。
                [System answer] 3个。
                
            Then call function write_qa_pair(question="Internal Dashboard有几个视图。", answer="3个。")
            
            Args:
                question (str): The customer's question.
                answer (str): The system's answer.
            
            Returns:
                str: Confirmation message that the QA pair has been successfully added. 
        """
        with open('qa_pairs.txt', 'a', encoding='utf-8') as file:
            file.write("0.\n")
            file.write(f"[客户问题] {question}\n")
            file.write(f"[系统回答] {answer}\n")
            file.write("\n")

        return "知识问答对补充成功。"
    
    def construct_agent_prompt(self):
        prompt = hub.pull("hwchase17/openai-tools-agent")
        prompt.messages[0].prompt.template = '''
                                            You are an SCCT (Supply Chain Control Tower) Agent. You can help users more accurately by retrieving information from the project's knowledge database.
                                            If the user is willing to supplement your knowledge base buffer, call the relevant methods to write to our knowledge base buffer.
                                            If there is no accurate answer, reply with 'No relevant reference available' or
                                            ask the user to help supplement the relevant knowledge (e.g., help me add this Q&A pair to the knowledge base: [Customer question] How many views are there in the Internal Dashboard? [System answer] User's Answer. 
                                            !The parameters parsed by the agent can only be named question and answer, both are user's input and must be present.
                                            Do not fabricate answers.
                                            If the question is not related to the shipping and logistics industry, guide the user to ask questions related to the shipping and logistics industry.
                                            '''
        self.qaAgentPrompt = prompt

    def initialize_scct_agent(self):

        self.tools = [tool(self.write_qa_pair), tool(self.scct_bot_tool)]
        self.agent = create_tool_calling_agent(self.llm, tools=self.tools, prompt=self.qaAgentPrompt)
        self.agentExecutor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    def chat_with_user(self, message, history):
        userQuery = {
            "input":message,
            "chat_history":self.memory.messages
            }
        response = self.agentExecutor.invoke(userQuery)
        self.memory.add_user_message(message)
        self.memory.add_ai_message(response['output'])
        return response['output']

class SCCTImageAgent():
    def __init__(self, api_key, base_url, vector_db) -> None:
        self.db = vector_db
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.prompForDetactFunction = """
            任务描述：
                你将收到一张系统功能的截图。
                你的任务是识别图像中的功能点。
            输出格式：
                '输入用户名','登录','提交表单'
            注意事项：
                确保所有功能都已被识别到。
                如果图像中有多个功能点, 请参考输出格式列出。
            """
        self.promptForGenerateAC = """
            任务描述：
                你将收到一张系统功能的截图和相关的后端系统逻辑。
                后端逻辑会以[source_documents]拼接在本Prompt最后。
                你的任务是识别图像中的功能点, 并为每个功能点编写测试点和验收标准(Acceptance Criteria), 而且前段UI和后端系统逻辑都要参考, 不能只看前端UI。
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

    def get_img_to_text_result(self, prompt, img):
        pil_image = Image.fromarray(img)
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.client.api_key}"
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                }
            ],
            "max_tokens": 3000,
            "temperature": 0
        }

        response = requests.post(base_url +"/chat/completions", headers=headers, json=payload)
        
        if response.status_code == 200:
            response_data = response.json()
            content = response_data['choices'][0]['message']['content']
            return content
        else:
            return f"Error: {response.status_code}, {response.text}"
    
    def view_img(self, img):
        function_list = self.get_img_to_text_result(self.prompForDetactFunction, img)
        topK_retriever = self.db.as_retriever(search_kwargs={"k": 3})
        source_documents = ''
        for i in function_list.split(','):
            docs = topK_retriever.get_relevant_documents(i)
            for doc in docs:
                source_documents = source_documents + doc.page_content + "\n"
        promptForAC = self.promptForGenerateAC + '\n' + source_documents
        return self.get_img_to_text_result(promptForAC, img)

def launch_gradio(fnForSCCTAgent, fnForSCCTImageAgent):

    scct_agent = gr.ChatInterface(
        fn=fnForSCCTAgent,
        title="SCCT Agent",
        chatbot=gr.Chatbot(height=600))
    
    scct_image_agent = gr.Interface(
        fn = fnForSCCTImageAgent, 
        inputs="image", 
        outputs="text")

    demo = gr.TabbedInterface(
        [
            scct_agent, scct_image_agent
        ],
        tab_names=["SCCT Agent", "SCCT Image Agent"]
    )
    
    demo.launch(share=False, server_name="0.0.0.0")

if __name__ == "__main__":
    api_key = "sk-wrPvJlufPtE3zmYAB62cDe655eD54521B0B4996767F0Ea29"
    base_url = "https://api.xiaoai.plus/v1"
    vector_db = vectorDB(vector_store_dir="real_scct_agent_test").db
    scctAgent = SCCTAgent(api_key=api_key, base_url=base_url, vector_db=vector_db)
    scctImageAgent = SCCTImageAgent(api_key=api_key, base_url=base_url, vector_db=vector_db)
    launch_gradio(scctAgent.chat_with_user, scctImageAgent.view_img)
