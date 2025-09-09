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

## 🚀 빠른 시작

### Docker로 실행
```bash
# 이미지 빌드
docker build -t uhok-ml-inference .

# 컨테이너 실행
docker run -p 8001:8001 uhok-ml-inference
```

### Docker Compose로 실행
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
```bash
# 의존성 설치
pip install -r requirements.txt

# 개발 서버 실행
python -m app.main
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
# Docker Compose로 실행 중인 경우
docker compose logs -f ml-inference

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
# 백엔드에서 ML 서비스 연결 테스트
docker-compose exec backend ping ml-inference

# 포트 확인
docker-compose exec backend telnet ml-inference 8001
```

### 메모리 부족 해결
```bash
# 컨테이너 메모리 제한 설정
docker run -m 4g -p 8001:8001 uhok-ml-inference

# 또는 docker-compose.yml에서
services:
  ml-inference:
    deploy:
      resources:
        limits:
          memory: 4G
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