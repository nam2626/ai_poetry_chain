from dotenv import load_dotenv
import os

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser  
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableLambda

import streamlit as st

load_dotenv()
OPENAPI_API_KEY = os.getenv("OPENAI_API_KEY")
# OpenAI 초기화
openai = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.7)

# 프롬프트 템플릿 정의
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a poet who writes emotional poetry. You write poetry on the topics I talk about. result language is Korean."),
    ("human", "{input_data}")
])

# --- LCEL 기반 Runnable 체인 구성 ---
chain = prompt | openai | StrOutputParser()  # 시 생성

content = ""

# 전체 SequentialChain 구성
poetry_chain = RunnableSequence(
    chain
)

# 제목
st.title("인공지능 시인")

# 시 주제 입력 필드
content = st.text_input("시의 주제를 입력하세요:", placeholder="T1의 3연패를 칭찬하는 시를 작성해 주세요.")
st.write(f"입력된 주제: {content}")

# 체인 실행
try:
    # 시 작성 요청
    if st.button("시 작성하기"):
        with st.spinner("시를 작성하는 중입니다..."):
            poetry_result = poetry_chain.invoke({"input_data": content})
    
    st.subheader("작성된 시")
    st.write(f"{poetry_result}")
    
except Exception as e:
    print("Error during poetry generation:", e)
