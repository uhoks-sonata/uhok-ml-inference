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

# requirements.txt 복사 및 모든 의존성 설치
COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


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

# 1단계: 비루트 유저 생성
RUN useradd -m -u 1000 appuser

# 2단계: 필요한 디렉토리 생성 및 권한 설정
RUN mkdir -p /app /models/hf_cache && \
    chown -R appuser:appuser /app /models

# 3단계: 비루트 유저로 전환
USER appuser

# HF 캐시를 볼륨으로 분리(콜드스타트/네트워크 절약)
# 네임드 볼륨 선언(compose가 따로 지정하지 않았을 때만 쓰임). 
# 지금은 compose가 명시해 두었으니 compose 설정이 우선
VOLUME ["/models/hf_cache"]

EXPOSE 8001

# 헬스체크 (/health 엔드포인트 가정)
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
