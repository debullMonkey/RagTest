# Ollama 설치 및 실행 방법 (Windows 11)

---

## 1단계: Ollama 설치

**공식 사이트에서 다운로드:**
```
https://ollama.com/download
```
- Windows 클릭 → `OllamaSetup.exe` 다운로드
- 설치 파일 실행 (기본 설정으로 Next → Install)

설치 완료 후 **시스템 트레이**(우측 하단)에 Ollama 아이콘이 생깁니다.

---

## 2단계: 설치 확인

```bash
ollama --version
```
버전 숫자가 출력되면 정상 설치 완료.

---

## 3단계: 필요한 모델 다운로드

터미널(cmd 또는 PowerShell)에서 실행:

```bash
# 임베딩 모델 (RAG에서 문서 벡터화용)
ollama pull nomic-embed-text

# LLM 모델 (답변 생성용) - 약 2GB
ollama pull llama3.2
```

> 다운로드 시간은 인터넷 속도에 따라 다릅니다.

---

## 4단계: Ollama 서버 실행 확인

Ollama는 설치 후 **백그라운드에서 자동 실행**됩니다.
수동으로 확인/실행하려면:

```bash
# 서버 상태 확인
ollama list

# 서버 직접 실행 (자동 실행 안될 때)
ollama serve
```

브라우저에서 `http://localhost:11434` 접속 시
```
Ollama is running
```
이 뜨면 정상입니다.

---

## 5단계: RAG 실행

```bash
cd C:\Users\HOME\Desktop\rag
python rag_test.py
```

---

## 자주 발생하는 문제

| 문제 | 해결 |
|------|------|
| `ollama` 명령어 없음 | 터미널 재시작 (환경변수 반영) |
| 연결 오류 | `ollama serve` 실행 후 재시도 |
| 모델 없음 오류 | `ollama pull llama3.2` 재실행 |
| GPU 미인식 | NVIDIA 드라이버 최신 버전으로 업데이트 |
