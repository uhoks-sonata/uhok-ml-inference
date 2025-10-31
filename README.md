# UHOK ML Inference Service

레시피 추천을 위한 임베딩 생성 및 벡터 유사도 검색 ML 서비스입니다.
**※연동되는 백엔드 버전: v3.0.0**

## 🎯 목적

- **비용 최적화**: 무거운 ML 모델 및 벡터 검색 로직을 별도 서비스로 분리하여 EC2 비용 절약
- **확장성**: ML 서비스를 독립적으로 스케일링 가능
- **유지보수성**: 모델 및 검색 로직 업데이트 시 백엔드 서비스 영향 최소화
- **역할 분리**: ML 관련 로직(임베딩, 검색)을 ML 서비스가 전담

## 🔧 기술 스택

- **FastAPI**: 고성능 비동기 웹 프레임워크
- **SentenceTransformers**: 문장 임베딩 생성 (paraphrase-multilingual-MiniLM-L12-v2)
- **PyTorch**: 딥러닝 프레임워크 (CPU 전용)
- **SQLAlchemy**: 비동기 ORM 및 데이터베이스 연결/쿼리
- **psycopg**: PostgreSQL 비동기 드라이버
- **pgvector**: PostgreSQL 벡터 확장 및 SQLAlchemy 지원
- **Docker**: 컨테이너화된 배포

## 🏗️ 아키텍처

```
[Backend Service] --HTTP--> [ML Inference Service] --SQL--> [PostgreSQL]
                                    ↓
                             [SentenceTransformer]
                             [paraphrase-multilingual-MiniLM-L12-v2]
```

## 🚀 빠른 시작

### 로컬 개발
```bash
# 의존성 설치
pip install -r requirements.txt

# .env 파일 생성 및 DB 연결 정보 설정 (예: POSTGRES_RECOMMEND_URL="postgresql+psycopg_async://user:password@localhost:5432/REC_DB")
# 개발 서버 실행
python -m app.main
```

### Docker로 실행
```bash
# 이미지 빌드
docker build -t uhok-ml-inference .

# 컨테이너 실행 (DB 연결 정보 필요)
docker run -p 8001:8001 --env-file .env uhok-ml-inference
```

### Docker Compose로 실행 (권장)
```bash
# uhok-deploy의 ml 폴더에서 실행
cd uhok-deploy/ml
docker-compose -f docker-compose.ml.yml up -d
```

## 📊 성능 특성

- **모델**: paraphrase-multilingual-MiniLM-L12-v2 (384차원)
- **처리량**: CPU 기반, 단일 워커
- **지연시간**: 첫 요청 시 모델 로딩 시간 포함
- **메모리**: 약 1-2GB (모델 + 런타임)

## 🗺️ API 엔드포인트

- `POST /api/v1/embed`: 단일 텍스트 임베딩 생성
- `POST /api/v1/embed-batch`: 배치 텍스트 임베딩 생성
- `POST /api/v1/search`: 쿼리 텍스트에 대한 유사 레시피 검색 (임베딩 생성 및 DB 검색 포함)
- `GET /api/v1/model-info`: 현재 로드된 모델 정보 반환
- `GET /health`: 서비스 헬스 체크

## 🔄 백엔드 연동

백엔드에서는 다음과 같이 원격 ML 서비스의 검색 API를 호출합니다:

```python
# 환경 변수 설정
ML_INFERENCE_URL=http://ml-inference:8001
ML_TIMEOUT=30.0  # 모델 로딩 및 DB 검색 시간 고려하여 충분한 타임아웃 설정
ML_RETRIES=2

# 원격 유사도 검색 호출
async with httpx.AsyncClient(timeout=ML_TIMEOUT) as client:
    response = await client.post(
        f"{ML_INFERENCE_URL}/api/v1/search",
        json={
            "query": "매콤한 닭볶음탕",
            "top_k": 10,
            "exclude_ids": [123, 456] # 선택 사항
        }
    )
    response.raise_for_status()
    result = response.json()
    search_results = result["results"] # [{'recipe_id': 789, 'distance': 0.123}, ...]
```

### 에러 처리
```python
try:
    response = await client.post(
        f"{ML_INFERENCE_URL}/api/v1/search",
        json={
            "query": "매콤한 닭볶음탕",
            "top_k": 10
        }
    )
    response.raise_for_status()
    return response.json()["results"]
except httpx.TimeoutException:
    logger.error("ML 서비스 타임아웃")
    return []
except httpx.HTTPStatusError as e:
    logger.error(f"ML 서비스 HTTP 에러: {e.response.status_code} - {e.response.text}")
    return []
except Exception as e:
    logger.error(f"ML 서비스 호출 실패: {e}")
    return []
```
