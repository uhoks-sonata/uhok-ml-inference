# UHOK ML Inference Service

레시피 추천을 위한 임베딩 생성 ML 서비스입니다.

## 🎯 목적

- **비용 최적화**: 무거운 ML 모델을 별도 서비스로 분리하여 EC2 비용 절약
- **확장성**: ML 서비스를 독립적으로 스케일링 가능
- **유지보수성**: 모델 업데이트 시 백엔드 서비스 영향 최소화

## 🏗️ 아키텍처

```
[Backend Service] --HTTP--> [ML Inference Service]
     ↓                              ↓
[PostgreSQL]                 [SentenceTransformer]
[pgvector]                   [paraphrase-multilingual-MiniLM-L12-v2]
```

## 🚀 API 엔드포인트

### 헬스 체크
```http
GET /health
```

### 단일 텍스트 임베딩
```http
POST /api/v1/embed
Content-Type: application/json

{
  "text": "갈비탕",
  "normalize": true
}
```

### 배치 텍스트 임베딩
```http
POST /api/v1/embed-batch
Content-Type: application/json

{
  "texts": ["갈비탕", "김치찌개", "된장찌개"],
  "normalize": true
}
```

### 모델 정보 조회
```http
GET /api/v1/model-info
```

## 🐳 Docker 실행

### 로컬 개발
```bash
# 이미지 빌드
docker build -t uhok-ml-inference .

# 컨테이너 실행
docker run -p 8001:8001 uhok-ml-inference
```

### Docker Compose
```yaml
services:
  ml-inference:
    build: ./uhok-ml-inference
    ports:
      - "8001:8001"
    environment:
      - HF_HOME=/models/hf_cache
    volumes:
      - ml_cache:/models/hf_cache
```

## 📊 성능 특성

- **모델**: paraphrase-multilingual-MiniLM-L12-v2 (384차원)
- **처리량**: CPU 기반, 단일 워커
- **지연시간**: 첫 요청 시 모델 로딩 시간 포함
- **메모리**: 약 1-2GB (모델 + 런타임)

## 🔧 설정

### 환경 변수
- `HF_HOME`: HuggingFace 모델 캐시 디렉토리
- `PYTHONPATH`: Python 경로 설정

### 로깅
- 구조화된 JSON 로그 출력
- 요청/응답 시간 측정
- 에러 상세 정보 포함

## 🧪 테스트

```bash
# 헬스 체크
curl http://localhost:8001/health

# 임베딩 생성 테스트
curl -X POST http://localhost:8001/api/v1/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "갈비탕", "normalize": true}'
```

## 📈 모니터링

- **헬스체크**: `/health` 엔드포인트
- **메트릭**: 요청 수, 응답 시간, 에러율
- **로그**: 구조화된 JSON 형태

## 🔄 백엔드 연동

백엔드에서는 다음과 같이 원격 ML 서비스를 호출합니다:

```python
# 환경 변수 설정
ML_INFERENCE_URL=http://ml-inference:8001
ML_TIMEOUT=3.0
ML_RETRIES=2

# 원격 임베딩 호출
async with httpx.AsyncClient(timeout=ML_TIMEOUT) as client:
    response = await client.post(
        f"{ML_INFERENCE_URL}/api/v1/embed",
        json={"text": query, "normalize": True}
    )
    embedding = response.json()["embedding"]
```

## 🚨 주의사항

1. **첫 요청 지연**: 모델 로딩으로 인한 콜드스타트 (약 10-30초)
2. **메모리 사용량**: 모델 크기로 인한 높은 메모리 사용
3. **네트워크 의존성**: 백엔드와 ML 서비스 간 네트워크 연결 필요
4. **에러 처리**: ML 서비스 장애 시 폴백 메커니즘 필요
