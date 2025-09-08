# syntax=docker/dockerfile:1.7

#############################
# 1) Builder: 휠 설치 전용
#############################
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 빌드 툴은 빌더에만 존재 (최종 이미지엔 X)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
  && rm -rf /var/lib/apt/lists/*

# (선택) 추가 의존성이 있다면 requirements.txt 먼저 설치해 캐시 활용
#   * 가장 좋은 방법은 아래 "중요"에 적은 대로 모든 버전을 requirements.txt에 명시하는 것
COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt || true

# 명시 버전 패키지 설치 (필요 시 requirements.txt로 이전 권장)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --prefix=/install \
      fastapi==0.116.1 uvicorn[standard]==0.35.0 pydantic==2.11.7 \
      httpx==0.28.1 python-multipart==0.0.20

# Torch CPU 전용 휠
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --prefix=/install \
      torch==2.7.1 torchvision==0.22.1 \
      --index-url https://download.pytorch.org/whl/cpu

# ML 스택
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --prefix=/install \
      sentence-transformers==5.0.0 huggingface_hub==0.34.3 \
      transformers==4.54.1 scikit-learn==1.7.1 \
      scipy numpy==2.3.2 pandas==2.3.1


#############################
# 2) Runtime: 경량 실행
#############################
FROM python:3.11-slim AS runtime

# 런타임 변수/캐시 경로
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models/hf_cache \
    TRANSFORMERS_CACHE=/models/hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/models/hf_cache \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

# 과학 연산/토치용 런타임 라이브러리 + 헬스체크용 curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libstdc++6 curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 빌더에서 설치한 site-packages만 복사 (휠/툴 X)
COPY --from=builder /install /usr/local

# 애플리케이션 소스
COPY . .

# 비루트 유저
RUN useradd -m appuser && chown -R appuser:appuser /app /models || true
USER appuser

# HF 캐시를 볼륨으로 분리(콜드스타트/네트워크 절약)
VOLUME ["/models/hf_cache"]

EXPOSE 8001

# 헬스체크 (/health 엔드포인트 가정)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
