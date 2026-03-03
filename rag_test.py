import time  # 시간 측정을 위해 추가
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── 1. 문서 로드 & 청크 분할 ──────────────────────────────────────────────────
print("📄 문서 로딩 중...")
loader = TextLoader("sample_data.txt", encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"✅ {len(chunks)}개 청크로 분할 완료")

# ── 2. 벡터 DB 생성 ──────────────────────────────────────────────────────────
print("🔢 벡터 DB 생성 중...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
print("✅ 벡터 DB 저장 완료")

# ── 3. LLM 초기화 ────────────────────────────────────────────────────────────
llm = OllamaLLM(model="llama3.2")

# ── 4. 질문 루프 ─────────────────────────────────────────────────────────────
while True:
    query = input("\n❓ 질문: ").strip()
    if query.lower() == 'q':
        break
    if not query:
        continue

    print("🔍 데이터 분석 중...")

    # --- [Step 1] 관련 문서 검색 시간 측정 (Retrieval) ---
    start_retrieval = time.time()
    results = db.similarity_search(query, k=3)
    context = "\n---\n".join([r.page_content for r in results])
    end_retrieval = time.time()
    
    retrieval_duration = end_retrieval - start_retrieval

    # --- [Step 2] LLM 답변 생성 시간 측정 (Generation) ---
    prompt = f"""당신은 회사 데이터 분석 전문가입니다.
[지침]
1. 아래 제공된 [참고 데이터]를 우선적으로 참고하여 답변하세요.
2. 만약 [참고 데이터]에 직접적인 답이 없다면, "제공된 문서에는 없지만 일반적인 지식으로는~"과 같이 답변하거나, 
   정말로 알 수 없는 전문적인 회사 정보라면 "데이터에 없는 내용입니다"라고 하세요.

[참고 데이터]
{context}

[질문]
{query}

[답변]"""

    start_generation = time.time()
    answer = llm.invoke(prompt)
    end_generation = time.time()

    generation_duration = end_generation - start_generation

    # ── 결과 및 시간 지표 출력 ──────────────────────────────────────────────
    print(f"\n{answer}")
    
    print("\n" + "-"*30)
    print(f"⏱️ 데이터 조회 시간: {retrieval_duration:.3f}초")
    print(f"⏱️ 답변 생성 시간: {generation_duration:.3f}초")
    print(f"🚀 총 소요 시간: {retrieval_duration + generation_duration:.3f}초")
    print("-"*30)

    print("\n[참고한 데이터 조각]")
    for i, r in enumerate(results, 1):
        print(f"   [{i}] {r.page_content[:80]}...")