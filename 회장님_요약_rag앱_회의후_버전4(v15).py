import streamlit as st
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
import openai
from io import BytesIO

# OpenAI 키 세팅
openai.api_key = st.secrets["OPENAI_API_KEY"]

# 벡터스토어 로드
import os

@st.cache_resource
def load_vectorstore():
    try:
        base_dir = os.path.dirname(__file__)
        vs_path = os.path.join(base_dir, "ja-in_vectorstore")
        if not os.path.isdir(vs_path):
            st.warning(f"Vectorstore 폴더 없음: {vs_path}")
            return None
        return FAISS.load_local(
            vs_path,
            OpenAIEmbeddings(openai_api_key=openai.api_key),
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.exception(e)  # 화면에 트레이스백 표시
        return None

vectorstore = load_vectorstore()

# LLM 인스턴스
llm = ChatOpenAI(model="gpt-4o", temperature=0.4)

# 핵심 키워드 엑셀 업로드
st.title("회장님 답변 요약 → 재구성 자동화")
keyword_file = st.file_uploader("🔑 핵심 키워드 파일 업로드", type=["xlsx"])
uploaded_file = st.file_uploader("📄 Q&A 엑셀 업로드", type="xlsx")

if not uploaded_file or not keyword_file:
    st.stop()

# 핵심 키워드 로딩
keyword_df = pd.read_excel(keyword_file, engine="openpyxl")
core_keywords = keyword_df.iloc[:, 0].dropna().astype(str).tolist()

# --- 1단계: 요약 함수 ---
def generate_summary(question, background, answer):
    query = f"{question}\n{background}"
    docs = vectorstore.similarity_search(query, k=3)
    reference = "\n".join([doc.page_content for doc in docs])

    system_prompt = (
        "회장님의 실제 발화 내용을 바탕으로 정돈된 요약을 생성한다.\n"
        "- 회장님의 어투와 워딩은 최대한 유지한다.\n"
        "- 흐름만 자연스럽게 정돈하되, 내용은 절대로 압축하지 않는다.\n"
        "- 원문에서 말한 모든 핵심 메시지와 설명은 그대로 살아 있어야 한다.\n"
        "- 전체 분량은 원문의 90% 이상으로 유지해야 하며, 원문 중심으로 작성한다.\n"
        "- '관련 참고자료'는 회장님의 철학을 보조적으로 반영하며, 실제 발화와 충돌할 경우 원문을 우선한다.\n"
        "- 단, 회장님의 철학에 어긋나거나 맥락상 오해 소지가 큰 표현이 있는 경우에만 자인사상 문서를 참고하여 자연스럽게 보완한다.\n"
        "- 출력은 모두 '~다'체로 한다."
        f"- 주의: 다음은 회장님의 철학에서 매우 중요한 용어이다. 전사 오류로 인해 비슷한 표현으로 잘못 표기되었더라도, 문맥상 같으면 올바른 용어로 교정하라:\n{', '.join(core_keywords)}"
    )

    user_prompt = f"질문:\n{question}\n\n질문 배경:\n{background}\n\n답변:\n{answer}\n\n관련 참고자료:\n{reference}"

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    return response.content

# --- 2단계: 재구성 함수 ---
def regenerate_summary(original_summary, original_answer):
    regen_prompt = f"""
    아래는 회장님의 답변 요약입니다. 문장 간 연결 흐름을 자연스럽게 정리하고, 원문에서 중요한 예시가 생략된 경우 다시 반영해 정돈해주세요. 
    회장님의 말투와 톤은 그대로 유지하고, 내용 축약 없이 최대한 풍부하게 보완하되 전체 분량은 요약문의 100~120% 이내로 맞춰주세요.
    출력은 '~다'체로 해주세요.

    다음 용어들이 중요한 철학 용어이니 전사 오류로 잘못 표기된 경우 문맥을 고려하여 바로잡아 주세요:
    {', '.join(core_keywords)}

    [요약문]
    {original_summary}

    [답변 원문]
    {original_answer}

    [출력 형식 예시]
    - 주제: 자인사상과 성찰
    - 핵심 메시지: 리더의 성장은 내면 성찰과 세상을 향한 확장적 시야에서 비롯된다.
    - 내용:
    리더는 자신을 돌아보고, 세상과의 관계를 깊이 이해해야 한다. 이해를 위해서는 '자인사상'을 아는 것이 중요하다. 이를 통해 구성원이 자신을 넘어서 타인을 포용할 수 있게 도울 수 있으며, 자인에서 하는 말들을 실천함으로써 선행하는 리더가 될 수 있다. 예를 들어, "성장은 동일시의 확장 과정이다"라는 말처럼...

    위 형식으로 재구성해줘.
    """
    chat = ChatOpenAI(model="gpt-4o", temperature=0.6)
    response = chat.invoke([HumanMessage(content=regen_prompt)])
    return response.content

# --- 실행 로직 ---
df = pd.read_excel(uploaded_file, engine-"openpyxl")

if {'사전질문', '질문배경', '답변원문'}.issubset(df.columns):
    with st.spinner("요약 중..."):
        summaries = []
        progress = st.progress(0)
        for i, row in df.iterrows():
            summary = generate_summary(row['사전질문'], row['질문배경'], row['답변원문'])
            summaries.append(summary)
            progress.progress((i + 1) / len(df))
        df['자동요약'] = summaries
        st.success("✅ 요약 완료")

    with st.spinner("재구성 중..."):
        regens = []
        progress2 = st.progress(0)
        for i, row in df.iterrows():
            regen = regenerate_summary(row['자동요약'], row['답변원문'])
            regens.append(regen)
            progress2.progress((i + 1) / len(df))
        df['재구성요약'] = regens
        st.success("✅ 재구성 완료")

    st.dataframe(df[['사전질문', '자동요약', '재구성요약']])

    # 다운로드
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    st.download_button("📥 최종 엑셀 다운로드", data=output.getvalue(), file_name="요약_재구성_결과.xlsx")

else:
    st.error("필수 컬럼(사전질문, 질문배경, 답변원문)이 누락되어 있습니다.")



