# ML 추론 서비스 Dockerfile (Python 3.13.5 최적화)
FROM python:3.13.5-slim

# 시스템 패키지 설치 (FAISS 제거로 최소화)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 포트 노출
EXPOSE 8080

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

# 애플리케이션 실행
CMD ["python", "-m", "app.main"]
