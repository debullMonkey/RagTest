import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── 설정 ─────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL       = "gemma3:4b"
FAISS_INDEX_DIR = "./faiss_index"

# ── 1. 문서 로드 & 청크 분할 ──────────────────────────────────────────────────
print("📄 문서 로딩 중...")
loader = TextLoader("sample_data.txt", encoding="utf-8")
docs   = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks   = splitter.split_documents(docs)
print(f"✅ {len(chunks)}개 청크로 분할 완료")

# ── 2. HuggingFace 임베딩 초기화 ──────────────────────────────────────────────
print(f"🤗 HuggingFace 임베딩 로딩 중... ({EMBEDDING_MODEL})")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# ── 3. FAISS 벡터 DB 생성 ──────────────────────────────────────────────────────
print("🗄️  FAISS 벡터 DB 생성 중...")
db = FAISS.from_documents(chunks, embeddings)
db.save_local(FAISS_INDEX_DIR)
print(f"✅ FAISS 인덱스 저장 완료 ({FAISS_INDEX_DIR})")

# ── 4. LLM 초기화 (Gemma3) ────────────────────────────────────────────────────
print(f"🤖 LLM 초기화 중... ({LLM_MODEL})")
llm = OllamaLLM(model=LLM_MODEL)

print("\n" + "="*50)
print(f" 임베딩 : {EMBEDDING_MODEL}")
print(f" LLM    : {LLM_MODEL}")
print(f" VectorDB: FAISS")
print("="*50)

# ── 5. 질문 루프 ─────────────────────────────────────────────────────────────
while True:
    query = input("\n❓ 질문 (종료: q): ").strip()
    if query.lower() == "q":
        break
    if not query:
        continue

    print("🔍 검색 중...")

    # Retrieval
    t0 = time.time()
    results = db.similarity_search(query, k=3)
    context = "\n---\n".join([r.page_content for r in results])
    retrieval_time = time.time() - t0

    # Generation
    prompt = f"""당신은 회사 데이터 분석 전문가입니다.

[지침]
1. 아래 [참고 데이터]를 우선 참고하여 답변하세요.
2. 참고 데이터에 직접적인 답이 없으면 "제공된 문서에는 없지만 일반적으로는~" 형식으로 답변하세요.
3. 회사 내부 전문 정보로 알 수 없는 경우 "데이터에 없는 내용입니다"라고 하세요.

[참고 데이터]
{context}

[질문]
{query}

[답변]"""

    t1 = time.time()
    answer = llm.invoke(prompt)
    generation_time = time.time() - t1

    # 출력
    print(f"\n{answer}")
    print("\n" + "-"*40)
    print(f"⏱️  검색 시간    : {retrieval_time:.3f}초")
    print(f"⏱️  생성 시간    : {generation_time:.3f}초")
    print(f"🚀  총 소요 시간 : {retrieval_time + generation_time:.3f}초")
    print("-"*40)
    print("\n[참고한 데이터 조각]")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r.page_content[:80]}...")
