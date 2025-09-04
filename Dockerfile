# uhok-ml-inference/Dockerfile
FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models/hf_cache

# 필수 OS 패키지 (numpy/scipy 등 기본 빌드 최소화)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-ml.txt .

# ⚠️ CPU 전용 PyTorch 인덱스
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-ml.txt --index-url https://download.pytorch.org/whl/cpu

COPY . .

# (옵션) 이미지 빌드 시 모델 미리 받아서 콜드스타트 제거
#   → 이미지 커지니 비용/속도 트레이드오프. 필요하면 주석 해제.
# ARG MODEL_ID=sentence-transformers/all-MiniLM-L6-v2
# RUN python - <<'PY'
# from sentence_transformers import SentenceTransformer
# import os; mid=os.environ.get("MODEL_ID","sentence-transformers/all-MiniLM-L6-v2")
# SentenceTransformer(mid)
# PY

EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
