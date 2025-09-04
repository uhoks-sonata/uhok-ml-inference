# uhok-ml-inference/Dockerfile
FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models/hf_cache

# 필수 OS 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements.txt 복사 및 설치
COPY requirements.txt .

# 일반 패키지 먼저 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir fastapi==0.116.1 uvicorn[standard]==0.35.0 pydantic==2.11.7 httpx==0.28.1 python-multipart==0.0.20

# PyTorch CPU 전용 설치
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu

# 나머지 ML 패키지 설치
RUN pip install --no-cache-dir sentence-transformers==5.0.0 huggingface_hub==0.34.3 transformers==4.54.1 scikit-learn==1.7.1 scipy numpy==2.3.2 pandas==2.3.1

# 애플리케이션 코드 복사
COPY . .

# (선택사항) 이미지 빌드 시 모델 미리 다운로드하여 콜드스타트 제거
# 주석 해제하면 이미지 크기가 커지지만 첫 요청이 빨라집니다
# ARG MODEL_ID=paraphrase-multilingual-MiniLM-L12-v2
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('$MODEL_ID')"

EXPOSE 8001

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
