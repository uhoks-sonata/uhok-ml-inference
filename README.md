# UHOK ML Inference Service

레시피 추천을 위한 임베딩 생성 ML 서비스입니다.

## 🎯 목적

- **비용 최적화**: 무거운 ML 모델을 별도 서비스로 분리하여 EC2 비용 절약
- **확장성**: ML 서비스를 독립적으로 스케일링 가능
- **유지보수성**: 모델 업데이트 시 백엔드 서비스 영향 최소화

## 🔧 코드 구조 및 기능

### 핵심 컴포넌트

#### 1. **app/main.py** - FastAPI 애플리케이션 진입점
- **기능**: FastAPI 서버 설정 및 CORS 미들웨어 구성
- **주요 역할**:
  - 서비스 메타데이터 정의 (제목, 설명, 버전)
  - CORS 설정으로 크로스 오리진 요청 허용
  - API 라우터 등록 (`/api/v1` 프리픽스)
  - 헬스체크 엔드포인트 (`/health`) 제공
  - uvicorn 서버 실행 (포트 8001)

#### 2. **app/api.py** - REST API 엔드포인트
- **기능**: 임베딩 생성 및 배치 처리 API 제공
- **주요 엔드포인트**:
  - `POST /api/v1/embed`: 단일 텍스트 임베딩 생성
  - `POST /api/v1/embed-batch`: 다중 텍스트 배치 임베딩 생성
  - `GET /api/v1/model-info`: 현재 모델 정보 조회
- **데이터 모델**:
  - `EmbedRequest`: 단일 임베딩 요청 (텍스트, 정규화 여부)
  - `EmbedBatchRequest`: 배치 임베딩 요청 (텍스트 리스트, 최대 100개)
  - `EmbedResponse`: 임베딩 응답 (벡터, 차원, 버전)
  - `EmbedBatchResponse`: 배치 임베딩 응답 (벡터 리스트, 메타데이터)
- **성능 모니터링**: 각 요청의 실행 시간 측정 및 로깅

#### 3. **app/deps.py** - ML 모델 의존성 관리
- **기능**: SentenceTransformer 모델의 로딩, 캐싱, 임베딩 생성
- **주요 함수**:
  - `get_model()`: 전역 모델 캐시 관리 (싱글톤 패턴)
  - `encode_text()`: 텍스트를 384차원 임베딩으로 변환
  - `get_model_info()`: 모델 메타데이터 반환
- **모델 정보**:
  - 모델명: `paraphrase-multilingual-MiniLM-L12-v2`
  - 차원: 384차원
  - 디바이스: CPU 전용
  - 버전: sentence-transformers-5.0.0
- **동시성 제어**: asyncio.Lock을 사용한 스레드 안전한 모델 로딩

#### 4. **test_ml_service.py** - 통합 테스트 스크립트
- **기능**: ML 서비스의 모든 API 엔드포인트 테스트
- **테스트 항목**:
  - 헬스체크 테스트
  - 단일 텍스트 임베딩 테스트 (5개 샘플 텍스트)
  - 배치 임베딩 테스트 (최대 5개 텍스트)
  - 모델 정보 조회 테스트
- **성능 측정**: 각 요청의 실행 시간 측정 및 통계 제공
- **에러 처리**: 타임아웃 및 HTTP 에러 처리

### 기술 스택

#### **웹 프레임워크**
- **FastAPI**: 고성능 비동기 웹 프레임워크
- **Uvicorn**: ASGI 서버 (표준 WSGI 대신 비동기 지원)
- **Pydantic**: 데이터 검증 및 직렬화

#### **ML/AI 라이브러리**
- **SentenceTransformers**: 문장 임베딩 생성 (핵심 라이브러리)
- **PyTorch**: 딥러닝 프레임워크 (CPU 전용)
- **HuggingFace Hub**: 모델 다운로드 및 관리
- **Transformers**: HuggingFace 트랜스포머 모델 라이브러리

#### **데이터 처리**
- **NumPy**: 수치 계산 및 배열 처리
- **httpx**: 비동기 HTTP 클라이언트 (테스트용)

### 핵심 기능

#### **1. 임베딩 생성**
- 한국어 텍스트를 384차원 벡터로 변환
- 정규화 옵션 지원 (L2 정규화)
- 단일 및 배치 처리 모두 지원

#### **2. 모델 관리**
- 전역 캐시를 통한 효율적인 메모리 사용
- 첫 요청 시에만 모델 로딩 (콜드스타트 최소화)
- 동시성 안전한 모델 접근

#### **3. API 설계**
- RESTful API 설계 원칙 준수
- 명확한 요청/응답 스키마 정의
- 상세한 에러 메시지 및 로깅

#### **4. 성능 최적화**
- CPU 전용 PyTorch 사용으로 메모리 효율성
- 비동기 처리로 동시 요청 처리
- 모델 캐싱으로 반복 로딩 방지

## 🏗️ 아키텍처

```
[Backend Service] --HTTP--> [ML Inference Service]
     ↓                              ↓
[PostgreSQL]                 [SentenceTransformer]
[pgvector]                   [paraphrase-multilingual-MiniLM-L12-v2]
```

## 🚀 빠른 시작

### Docker로 실행
```bash
# 이미지 빌드
docker build -t uhok-ml-inference .

# 컨테이너 실행
docker run -p 8001:8001 uhok-ml-inference
```

### Docker Compose로 실행

#### 로컬 개발 환경
```bash
cd uhok-ml-inference
docker-compose -f docker-compose.ml.yml up -d
```

#### 통합 환경 (uhok-deploy와 함께)
```bash
cd uhok-deploy
docker-compose --profile with-ml up -d
```

## 📡 API 사용법

### 헬스 체크
```bash
curl http://localhost:8001/health
```

### 단일 텍스트 임베딩
```bash
curl -X POST http://localhost:8001/api/v1/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "갈비탕", "normalize": true}'
```

### 배치 텍스트 임베딩
```bash
curl -X POST http://localhost:8001/api/v1/embed-batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["갈비탕", "김치찌개", "된장찌개"], "normalize": true}'
```

## 🔧 개발 환경 설정

### 로컬 개발

#### Python 직접 실행
```bash
# 의존성 설치
pip install -r requirements.txt

# 개발 서버 실행
python -m app.main
```

#### Docker Compose 사용 (권장)
```bash
# ML 서비스만 독립 실행
cd uhok-ml-inference
docker-compose -f docker-compose.ml.yml up --build

# 백그라운드 실행
docker-compose -f docker-compose.ml.yml up -d

# 로그 확인
docker-compose -f docker-compose.ml.yml logs -f

# 서비스 중지
docker-compose -f docker-compose.ml.yml down
```

### 환경 변수
```bash
# HuggingFace 모델 캐시 디렉토리
export HF_HOME=/models/hf_cache

# Python 경로 설정
export PYTHONPATH=/app
```

## 📊 성능 특성

- **모델**: paraphrase-multilingual-MiniLM-L12-v2 (384차원)
- **처리량**: CPU 기반, 단일 워커
- **지연시간**: 첫 요청 시 모델 로딩 시간 포함
- **메모리**: 약 1-2GB (모델 + 런타임)

## 🧪 테스트

### 기본 테스트
```bash
# 헬스 체크
curl http://localhost:8001/health

# 임베딩 생성 테스트
curl -X POST http://localhost:8001/api/v1/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "갈비탕", "normalize": true}'

# 배치 임베딩 테스트
curl -X POST http://localhost:8001/api/v1/embed-batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["갈비탕", "김치찌개", "된장찌개"], "normalize": true}'

# 모델 정보 조회
curl http://localhost:8001/api/v1/model-info
```

### 통합 테스트 스크립트
```bash
# uhok-deploy 디렉토리에서 실행
python test_ml_integration.py
```

## 🔄 백엔드 연동

백엔드에서는 다음과 같이 원격 ML 서비스를 호출합니다:

```python
# 환경 변수 설정
ML_INFERENCE_URL=http://ml-inference:8001
ML_TIMEOUT=30.0  # 모델 로딩 시간 고려하여 충분한 타임아웃 설정
ML_RETRIES=2

# 원격 임베딩 호출
async with httpx.AsyncClient(timeout=ML_TIMEOUT) as client:
    response = await client.post(
        f"{ML_INFERENCE_URL}/api/v1/embed",
        json={"text": query, "normalize": True}
    )
    response.raise_for_status()
    result = response.json()
    embedding = result["embedding"]
    dim = result["dim"]
```

### 에러 처리
```python
try:
    response = await client.post(
        f"{ML_INFERENCE_URL}/api/v1/embed",
        json={"text": query, "normalize": True}
    )
    response.raise_for_status()
    return response.json()["embedding"]
except httpx.TimeoutException:
    logger.error("ML 서비스 타임아웃")
    return None
except httpx.HTTPStatusError as e:
    logger.error(f"ML 서비스 HTTP 에러: {e.response.status_code}")
    return None
except Exception as e:
    logger.error(f"ML 서비스 호출 실패: {e}")
    return None
```

## 📈 모니터링

### 로그 확인
```bash
# Docker Compose로 실행 중인 경우 (독립 실행)
docker-compose -f docker-compose.ml.yml logs -f

# 통합 환경에서 실행 중인 경우
docker-compose logs -f ml-inference

# 직접 실행 중인 경우
python -m app.main
```

### 성능 모니터링
```bash
# 메모리 사용량 확인
docker stats uhok-ml-inference

# CPU 사용량 확인
docker exec uhok-ml-inference top
```

## 🚨 주의사항

1. **첫 요청 지연**: 모델 로딩으로 인한 콜드스타트 (약 10-30초)
2. **메모리 사용량**: 모델 크기로 인한 높은 메모리 사용 (1-2GB)
3. **네트워크 의존성**: 백엔드와 ML 서비스 간 네트워크 연결 필요
4. **에러 처리**: ML 서비스 장애 시 폴백 메커니즘 필요
5. **타임아웃 설정**: 모델 로딩 시간을 고려한 충분한 타임아웃 설정 필요
6. **동시성**: 단일 모델 인스턴스로 인한 처리량 제한

## 🔧 문제 해결

### 모델 캐시 관리
```bash
# HuggingFace 캐시 디렉토리 확인
ls -la ~/.cache/huggingface/

# 캐시 정리 (필요시)
rm -rf ~/.cache/huggingface/
```

### 네트워크 연결 확인
```bash
# 독립 실행 환경에서 연결 테스트
curl http://localhost:8001/health

# 통합 환경에서 백엔드에서 ML 서비스 연결 테스트
docker-compose exec backend ping ml-inference

# 포트 확인
docker-compose exec backend telnet ml-inference 8001
```

### 메모리 부족 해결
```bash
# 컨테이너 메모리 제한 설정
docker run -m 4g -p 8001:8001 uhok-ml-inference

# 또는 docker-compose.ml.yml에서
services:
  ml-inference:
    deploy:
      resources:
        limits:
          memory: 4G
```

## 🔄 버전 관리

### 버전 업그레이드
```bash
# 1. docker-compose.ml.yml에서 이미지 버전 수정
# image: uhok-ml-inference:1.0.1 → uhok-ml-inference:1.0.2

# 2. 새 이미지 빌드
docker-compose -f docker-compose.ml.yml build --no-cache

# 3. 서비스 재시작
docker-compose -f docker-compose.ml.yml down
docker-compose -f docker-compose.ml.yml up -d
```

### 롤백
```bash
# 이전 버전으로 롤백
# docker-compose.ml.yml에서 이전 버전으로 수정 후
docker-compose -f docker-compose.ml.yml down
docker-compose -f docker-compose.ml.yml up -d
```

## 📚 API 문서

자세한 API 문서는 서비스 실행 후 다음 URL에서 확인할 수 있습니다:
- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 지원

문제가 발생하거나 질문이 있으시면:
1. 이슈를 생성해주세요
2. 로그를 확인해주세요: `docker-compose logs -f ml-inference`
3. 헬스체크를 확인해주세요: `curl http://localhost:8001/health`