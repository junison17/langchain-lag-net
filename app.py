import os
import time
import json
from dotenv import load_dotenv
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings  # 수정된 부분
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain import hub
from langchain_openai import ChatOpenAI

# 환경 변수 로드
load_dotenv()

# API 키 확인 및 설정
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")

if not TAVILY_API_KEY or not OPENAI_API_KEY:
    st.error("Tavily API 키 또는 OpenAI API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    st.stop()

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

if SERP_API_KEY:
    os.environ["SERPAPI_API_KEY"] = SERP_API_KEY
else:
    st.warning("SERP API 키가 설정되지 않았습니다. SerpAPI 검색 기능은 비활성화됩니다.")

# PDF 로더 및 청크 분할
@st.cache_resource
def load_and_process_pdf():
    try:
        loader = PyPDFLoader("minsa.pdf")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=300,
            length_function=len,
        )
        return text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"PDF 로딩 중 오류 발생: {str(e)}")
        st.stop()

chunks = load_and_process_pdf()

# 저장 경로 지정
persist_directory = "./chroma_db"

def is_vectorstore_ready():
    try:
        with open("vectorstore_status.json", "r") as f:
            status = json.load(f)
        return status.get("ready", False)
    except FileNotFoundError:
        return False

def mark_vectorstore_ready():
    with open("vectorstore_status.json", "w") as f:
        json.dump({"ready": True}, f)

# Chroma DB 설정
def setup_vectorstore():
    if is_vectorstore_ready():
        embeddings = HuggingFaceEmbeddings()
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    try:
        embeddings = HuggingFaceEmbeddings()
        texts = [chunk.page_content for chunk in chunks]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        vectorstore = Chroma.from_texts(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))], persist_directory=persist_directory)
        
        progress_bar.progress(1.0)
        status_text.text("벡터 저장소 설정 완료!")
        
        mark_vectorstore_ready()
        return vectorstore
    except Exception as e:
        st.error(f"벡터 저장소 설정 중 오류 발생: {str(e)}")
        st.stop()

vectorstore = setup_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# OpenAI 챗봇 설정
try:
    chat = ChatOpenAI(temperature=0)
except Exception as e:
    st.error(f"ChatOpenAI 초기화 중 오류 발생: {str(e)}")
    st.stop()

# 도구 설정
retriever_tool = create_retriever_tool(
    retriever,
    "civil_law_search",
    "민사소송법과 관련된 질문에 대해 검색합니다. 사용자 질문이 민사소송법과 관련되어 있다면 항상 이 도구를 사용하세요!",
)

tavily_tool = TavilySearchResults()

tools = [tavily_tool, retriever_tool]

# SerpAPI 초기화 (조건부)
try:
    from langchain_community.utilities import SerpAPIWrapper
    search = SerpAPIWrapper()
    serp_tool = Tool(
        name="serp_search",
        func=search.run,
        description="현재 이벤트나 최신 정보에 대한 질문에 답할 때 유용합니다."
    )
    tools.append(serp_tool)
except ImportError:
    st.warning("SerpAPI 초기화 실패: google-search-results 패키지를 설치해주세요. SerpAPI 검색 기능은 비활성화됩니다.")
except Exception as e:
    st.warning(f"SerpAPI 초기화 중 오류 발생: {str(e)}. SerpAPI 검색 기능은 비활성화됩니다.")

# 에이전트 설정
try:
    prompt = hub.pull("hwchase17/openai-functions-agent")
    agent = create_tool_calling_agent(chat, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
except Exception as e:
    st.error(f"에이전트 설정 중 오류 발생: {str(e)}")
    agent_executor = None

# Streamlit 앱 설정
st.title("민사소송법 전문가 챗봇")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 프롬프트 비용 관리
MAX_MESSAGES_BEFORE_DELETION = 4

if prompt := st.chat_input("민사소송법에 대해 무엇이든 물어보세요!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    if len(st.session_state.messages) >= MAX_MESSAGES_BEFORE_DELETION:
        del st.session_state.messages[:2]

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        if agent_executor is not None:
            try:
                with st.spinner("답변 생성 중..."):
                    result = agent_executor.invoke(
                        {"input": prompt, "chat_history": st.session_state.messages},
                        return_only_outputs=True,
                    )

                for chunk in result["output"].split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                    
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"답변 생성 중 오류 발생: {str(e)}")
        else:
            st.error("에이전트가 초기화되지 않았습니다. 시스템 관리자에게 문의하세요.")

print("_______________________")
print(st.session_state.messages)