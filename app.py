import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
from langchain_core.prompts import PromptTemplate
import yaml

# API KEY 정보로드
load_dotenv()

# 스트림릿 설정
st.title("아네리봇")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

def load_prompts(prompt_template_name : str):
    file_path = 'prompt.yaml'
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)[prompt_template_name]

# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

# 초기 추천 질의 생성 함수
def generate_recommended_queries():
    response = generate2("질문을 5개 만들어주고 각 질문은 \n\n으로 개행해줘. 질문은 #context로 답변 할 수 있는 질문만 만들어줘.")
    doc2query = "".join(response)  # 응답을 문자열로 결합
    add_message("assistant", f"\n아네리 봇은 아래와 같은 내용들에 답변을 할 수 있습니다.\n\n\n{doc2query}")

# 생성 메서드 1
def generate2(query):
    output_parser = StrOutputParser()

    llm = ChatOpenAI(
        temperature=0.1,
        model_name="gpt-4o-mini",
    )
    prompt = PromptTemplate.from_template(load_prompts('basic'))
    chain = prompt | llm | output_parser
    input = {"query": query}
    response = chain.invoke(input)
    return response

# 생성 메서드 2
def generate(query):
    output_parser = StrOutputParser()
    llm = ChatOpenAI(
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        temperature=0.5,
    )
    llm = ChatOpenAI(
        temperature=0.1,
        model_name="gpt-4o-mini",
    )
    prompt = PromptTemplate.from_template(load_prompts('basic'))
    chain = prompt | llm | output_parser
    input = {"query": query}
    response = chain.invoke(input)

    # 스트리밍 호출
    with st.chat_message("assistant"):
        container = st.empty()
        ai_answer = ""
        try:
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)
        except Exception as e:
            st.error(f"Error during response streaming: {e}")
            return

    add_message("user", query)
    add_message("assistant", ai_answer)

# 추천 질의가 처음 한번만 생성되도록 설정
if "recommended_queries_generated" not in st.session_state:
    generate_recommended_queries()
    st.session_state["recommended_queries_generated"] = True

# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = []
    st.session_state["recommended_queries_generated"] = False  # 초기화 시 다시 생성 가능하도록 설정

# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("나는 아네리에 대해 모르는게 없는 아네리봇!")

# 만약에 사용자 입력이 들어오면...
if user_input:
    st.chat_message("user").write(user_input)
    generate(user_input)
