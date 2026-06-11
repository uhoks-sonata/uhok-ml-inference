# 서비스 아키텍처 문서

## 목차

1. [서비스 개요](#1-서비스-개요)
2. [디렉토리 구조](#2-디렉토리-구조)
3. [레이어 구조](#3-레이어-구조)
4. [API 엔드포인트 목록](#4-api-엔드포인트-목록)
5. [의존성 흐름](#5-의존성-흐름)
6. [데이터 흐름](#6-데이터-흐름)
7. [기술 스택](#7-기술-스택)
8. [상위 서비스와의 관계](#8-상위-서비스와의-관계)

---

## 1. 서비스 개요

`uhok-ml-inference`는 레시피 추천 시스템에서 **ML 추론만 전담하는 독립 마이크로서비스**입니다.

백엔드(Spring Boot 또는 다른 서버)가 무거운 ML 모델을 직접 로드하지 않고, 이 서비스에 HTTP 요청을 보내 임베딩 생성과 벡터 유사도 검색 결과를 받아갑니다.

```
[클라이언트]
    │
    ▼
[uhok-backend]  ─── HTTP POST /api/v1/search ───▶  [uhok-ml-inference]
                                                            │
                                                            ├── SentenceTransformer (CPU)
                                                            └── PostgreSQL + pgvector
```

### 분리의 목적

| 항목 | 분리 전 | 분리 후 |
|------|---------|---------|
| 백엔드 메모리 | 2~3GB (모델 포함) | ~500MB |
| 백엔드 인스턴스 | ML 처리 가능한 대형 | 더 작은 인스턴스 가능 |
| 모델 업데이트 | 백엔드 재배포 필요 | ML 서비스만 재배포 |
| 스케일링 | 백엔드/ML 함께 | 독립적으로 수평 확장 가능 |

---

## 2. 디렉토리 구조

```
uhok-ml-inference/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 앱 진입점, 라우터 등록, CORS 설정
│   ├── deps.py              # 공유 의존성: 모델 캐시, DB 세션, encode_text
│   ├── models/
│   │   ├── embedding.py     # 임베딩 API Pydantic 스키마
│   │   └── search.py        # 검색 API Pydantic 스키마
│   ├── routers/
│   │   ├── embedding.py     # POST /embed, POST /embed-batch
│   │   ├── search.py        # POST /search
│   │   └── management.py    # GET /model-info
│   └── crud/
│       └── search_crud.py   # pgvector SQL 쿼리 실행
├── docs/                    # 이 문서들
├── Dockerfile               # 멀티스테이지 빌드 (builder + runtime, python:3.11-slim)
├── requirements.txt
├── .env                     # DB 연결 URL (비밀, git 제외)
└── test_ml_service.py
```

---

## 3. 레이어 구조

```
┌─────────────────────────────────────────┐
│           HTTP 요청/응답 레이어           │
│  main.py (FastAPI, CORS, 라우터 등록)    │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│             라우터 레이어                │
│  routers/embedding.py  (임베딩 API)      │
│  routers/search.py     (검색 API)        │
│  routers/management.py (모델 정보 API)   │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│           데이터 검증 레이어              │
│  models/embedding.py  (Pydantic 스키마)  │
│  models/search.py     (Pydantic 스키마)  │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│           의존성/인프라 레이어            │
│  deps.py                                │
│    - get_model()       모델 싱글턴 캐시  │
│    - encode_text()     텍스트 → 벡터     │
│    - get_db_session()  DB 세션 관리      │
└─────────────────────────────────────────┘
                    │
        ┌───────────┴────────────┐
        ▼                        ▼
┌───────────────┐      ┌─────────────────────────┐
│  ML 모델 레이어 │      │     DB 접근 레이어        │
│  deps.py      │      │  crud/search_crud.py     │
│               │      │                         │
│ SentenceTransf│      │  pgvector SQL 쿼리 실행   │
│ ormer (CPU)   │      │  (<-> L2 거리 연산)       │
└───────────────┘      └─────────────────────────┘
                                 │
                       ┌─────────────────┐
                       │   PostgreSQL     │
                       │   + pgvector     │
                       └─────────────────┘
```

---

## 4. API 엔드포인트 목록

### 헬스체크

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/health` | 서비스 상태 및 모델 정보 반환 |

응답:
```json
{
  "status": "ok",
  "model": "paraphrase-multilingual-MiniLM-L12-v2",
  "dim": 384,
  "version": "sentence-transformers-5.0.0"
}
```

### 임베딩 (`/api/v1`)

| 메서드 | 경로 | 설명 | 코드 |
|--------|------|------|------|
| POST | `/api/v1/embed` | 단건 텍스트 임베딩 생성 | `routers/embedding.py:12` |
| POST | `/api/v1/embed-batch` | 최대 100건 배치 임베딩 생성 | `routers/embedding.py:35` |

### 검색 (`/api/v1`)

| 메서드 | 경로 | 설명 | 코드 |
|--------|------|------|------|
| POST | `/api/v1/search` | 벡터 유사도 기반 레시피 검색 | `routers/search.py:14` |

### 관리 (`/api/v1`)

| 메서드 | 경로 | 설명 | 코드 |
|--------|------|------|------|
| GET | `/api/v1/model-info` | 로드된 모델 상세 정보 | `routers/management.py:10` |

---

## 5. 의존성 흐름

FastAPI의 `Depends()` 메커니즘으로 DB 세션과 모델을 주입합니다.

```
search_similar_recipes(request, db=Depends(get_db_session))
        │
        ├── db: AsyncSession
        │     └── get_db_session()
        │           └── AsyncSessionLocal (SQLAlchemy)
        │                 └── PostgreSQL (asyncpg 드라이버)
        │
        └── encode_text(request.query, normalize=True)
              └── get_model()
                    └── SentenceTransformer (전역 캐시, asyncio.Lock 보호)
```

`get_db_session()`은 `yield` 제너레이터입니다. FastAPI가 요청 시작 시 세션을 열고, 응답 전송 후 자동으로 닫아줍니다.

```python
async def get_db_session():
    async with AsyncSessionLocal() as session:
        yield session   # ← 요청 처리 중
    # ← 여기서 자동 close (요청 종료 후)
```

---

## 6. 데이터 흐름

### 검색 요청 전체 데이터 흐름

```
클라이언트
  POST /api/v1/search
  { "query": "된장찌개", "top_k": 25, "exclude_ids": [1, 2] }
        │
        ▼
FastAPI (Pydantic 검증)
  SearchRequest 파싱 및 타입 검증
        │
        ▼
routers/search.py:search_similar_recipes()
        │
        ├─▶ deps.py:encode_text("된장찌개", normalize=True)
        │         │
        │         ├─▶ deps.py:get_model()   # 캐시 확인 or 로드
        │         │
        │         └─▶ SentenceTransformer.encode()
        │               → numpy array (384,)
        │               → .tolist() → List[float] (384개)
        │
        ├─▶ crud/search_crud.py:search_similar_in_db()
        │         │
        │         └─▶ PostgreSQL
        │               SELECT RECIPE_ID, VECTOR_NAME <-> :qv AS distance
        │               FROM RECIPE_VECTOR_TABLE
        │               WHERE RECIPE_ID NOT IN (1, 2)
        │               ORDER BY distance ASC LIMIT 25
        │               → List[(recipe_id, distance)]
        │
        └─▶ SearchResponse 조립
              { "results": [{ "recipe_id": ..., "distance": ... }, ...] }
        │
        ▼
클라이언트 응답 (JSON)
```

---

## 7. 기술 스택

| 레이어 | 기술 | 버전 | 역할 |
|--------|------|------|------|
| 웹 프레임워크 | FastAPI | 0.116.1 | HTTP 라우팅, 의존성 주입 |
| ASGI 서버 | uvicorn | 0.35.0 | 비동기 HTTP 서버 |
| 데이터 검증 | Pydantic v2 | 2.11.7 | 요청/응답 스키마 검증 |
| 환경변수 | pydantic-settings | 2.10.1 | `.env` 파일 파싱 |
| ML 모델 | sentence-transformers | 5.0.0 | 텍스트 → 벡터 변환 |
| 딥러닝 런타임 | PyTorch | 2.7.1 (CPU) | Transformer 추론 |
| ORM | SQLAlchemy (async) | 2.0.42 | 비동기 DB 접근 |
| DB 드라이버 | psycopg[binary] | 3.2.9 | PostgreSQL 비동기 드라이버 (`psycopg_async`) |
| 벡터 확장 | pgvector | 0.3.6 | PostgreSQL 벡터 타입 및 연산 |
| DB | PostgreSQL | - | 레시피 벡터 저장 및 검색 |
| 컨테이너 | Docker | - | 멀티스테이지 빌드 (`builder` + `runtime`) |

---

## 8. 상위 서비스와의 관계

이 서비스는 **순수 ML 추론 레이어**만 담당합니다. 비즈니스 로직은 호출자(uhok-backend)가 처리합니다.

```
uhok-backend (비즈니스 로직)
    │
    │  1. 검색 쿼리 + 제외 ID 전송
    ▼
uhok-ml-inference (ML 추론)
    │
    │  2. 유사 레시피 ID + 거리 반환
    ▼
uhok-backend
    │
    │  3. 인기도 가중치, 개인화, 카테고리 필터 적용
    │  4. 최종 재랭킹
    ▼
클라이언트
```

이 서비스가 반환하는 `distance`는 원시 유사도 점수입니다. 최종 추천 순위는 상위 백엔드에서 추가 로직으로 결정됩니다.

### 환경변수 연동 (INTEGRATION_GUIDE.md 참고)

백엔드에서 이 서비스를 호출하는 방식은 환경변수 `ML_MODE`로 제어됩니다.

```bash
ML_MODE=local          # 백엔드 내부에서 직접 모델 실행 (개발용)
ML_MODE=remote_embed   # 이 서비스에 HTTP로 임베딩 요청 (운영용)
ML_INFERENCE_URL=http://ml-inference:8001
```

### DB 연결 URL 형식

`.env`의 `POSTGRES_RECOMMEND_URL`은 `psycopg_async` 드라이버 스킴을 사용합니다.

```bash
# .env
POSTGRES_RECOMMEND_URL="postgresql+psycopg_async://user:password@host:5432/dbname"
```

`asyncpg`가 아닌 `psycopg_async` (psycopg3의 비동기 드라이버)를 사용합니다. URL 스킴이 다르므로 주의하세요.

### Dockerfile 주요 설정

```dockerfile
# 런타임 성능 환경변수 (Dockerfile에 하드코딩)
OMP_NUM_THREADS=1           # PyTorch CPU 스레드 수 제한 (컨테이너 간 충돌 방지)
MKL_NUM_THREADS=1           # Intel MKL 스레드 수 제한
TOKENIZERS_PARALLELISM=false # HuggingFace 토크나이저 병렬처리 비활성화 (asyncio와 충돌 방지)

HF_HOME=/models/hf_cache    # 모델 캐시 경로 (볼륨 마운트)
```

실행 명령은 `--workers 1`로 단일 워커만 사용합니다. 멀티 워커 시 각 워커가 독립적으로 ~1GB 모델을 로드하므로 메모리 비용이 급증합니다.
