# 검색 및 추천 로직 상세 문서

## 목차

1. [개요](#1-개요)
2. [전체 흐름](#2-전체-흐름)
3. [Step 1 — 쿼리 임베딩 생성](#3-step-1--쿼리-임베딩-생성)
4. [Step 2 — 벡터 유사도 검색](#4-step-2--벡터-유사도-검색)
5. [Step 3 — 응답 포맷팅](#5-step-3--응답-포맷팅)
6. [exclude_ids와 추천 다양성](#6-exclude_ids와-추천-다양성)
7. [성능 측정 구조](#7-성능-측정-구조)
8. [설계 원칙과 한계](#8-설계-원칙과-한계)

---

## 1. 개요

`uhok-ml-inference`의 검색/추천 기능은 **의미 기반 벡터 유사도 검색(Semantic Search)** 방식으로 동작합니다.

키워드 매칭 방식(LIKE, Full-Text Search)과 달리, 사용자 쿼리의 **의미**를 벡터로 변환한 뒤 DB에 저장된 레시피 벡터들과 수학적 거리를 비교합니다.

```
"된장찌개 레시피" ──▶ [0.12, -0.34, 0.87, ...] (384차원 벡터)
                                │
                        pgvector <-> 연산
                                │
                 DB에 저장된 수만 개의 레시피 벡터와 거리 계산
                                │
                    거리 오름차순 상위 25개 반환
```

---

## 2. 전체 흐름

### API 엔드포인트

```
POST /api/v1/search
```

### 요청 스키마

```python
# app/models/search.py
class SearchRequest(BaseModel):
    query: str                          # 검색할 쿼리 텍스트 (필수)
    top_k: int = Field(25)              # 반환할 상위 결과 수 (기본 25)
    exclude_ids: Optional[List[int]]    # 검색에서 제외할 레시피 ID 목록
```

### 요청 예시

```json
{
  "query": "된장찌개 만드는 법",
  "top_k": 25,
  "exclude_ids": [101, 202, 303]
}
```

### 처리 순서 (`app/routers/search.py:15-49`)

```
POST /api/v1/search 수신
        │
        ▼
① encode_text(query, normalize=True)
   → SentenceTransformer로 384차원 벡터 생성
        │
        ▼
② search_similar_in_db(db, query_vector, top_k, exclude_ids)
   → PostgreSQL + pgvector로 유사도 검색
        │
        ▼
③ SearchResponse(results=[...]) 반환
```

---

## 3. Step 1 — 쿼리 임베딩 생성

### 코드 위치

- `app/routers/search.py:26`
- `app/deps.py:94-100`

### 호출 흐름

```python
# search.py:26
query_vector = await encode_text(request.query, normalize=True)

# deps.py:94-100
async def encode_text(text: str, normalize: bool = True) -> list:
    model = await get_model()
    embedding = model.encode(text, normalize_embeddings=normalize)
    return embedding.tolist()
```

### 모델 내부 처리 과정

```
입력 텍스트: "된장찌개 만드는 법"
        │
        ▼
토크나이저 (WordPiece/BPE)
→ ["된", "##장", "##찌", "##개", "만드는", "법"] + [CLS], [SEP]
        │
        ▼
Transformer 인코더 (MiniLM 12레이어)
→ 각 토큰별 문맥 이해 (Attention 메커니즘)
→ 각 토큰별 히든 스테이트 생성
        │
        ▼
Mean Pooling
→ 모든 토큰 벡터의 평균 → 단일 문장 벡터 (384차원)
        │
        ▼
L2 정규화 (normalize_embeddings=True)
→ 벡터 크기(magnitude)를 1.0으로 정규화
→ [0.12, -0.34, 0.87, ...] (384차원, ||v|| = 1.0)
```

### normalize=True의 수학적 의미

정규화란 벡터의 크기를 1로 맞추는 연산입니다.

```
정규화 전: v = [3.0, 4.0]   → ||v|| = √(9 + 16) = 5.0
정규화 후: v = [0.6, 0.8]   → ||v|| = √(0.36 + 0.64) = 1.0
```

정규화된 벡터에서 **L2 거리**와 **코사인 유사도**는 동일한 순위를 만듭니다.

```
코사인 유사도 = a·b / (||a|| × ||b||)

||a|| = ||b|| = 1이면:
코사인 유사도 = a·b

L2 거리² = ||a - b||² = ||a||² - 2(a·b) + ||b||² = 2 - 2(a·b)

따라서: L2 거리 ↓ ⟺ 코사인 유사도 ↑  (동일한 순위)
```

즉, DB에서 `<->` (L2 거리) 연산자로 조회하면서 실질적으로는 **의미적 유사도(코사인)** 기반 검색을 수행합니다.

### 모델 싱글턴 캐시

모델은 프로세스 당 1회만 로드됩니다. `app/deps.py:47-83` 의 Double-Checked Locking 패턴으로 동시 요청에서 중복 로드를 방지합니다. 자세한 내용은 [embedding.md](./embedding.md)를 참고하세요.

---

## 4. Step 2 — 벡터 유사도 검색

### 코드 위치

- `app/crud/search_crud.py:14-55`

### 핵심 SQL

#### exclude_ids 없을 때

```sql
SELECT "RECIPE_ID" AS recipe_id,
       "VECTOR_NAME" <-> :qv AS distance
FROM "RECIPE_VECTOR_TABLE"
ORDER BY distance ASC
LIMIT :k
```

#### exclude_ids 있을 때

```sql
SELECT "RECIPE_ID" AS recipe_id,
       "VECTOR_NAME" <-> :qv AS distance
FROM "RECIPE_VECTOR_TABLE"
WHERE "RECIPE_ID" NOT IN :ex_ids
ORDER BY distance ASC
LIMIT :k
```

### `<->` 연산자 — L2 유클리드 거리

pgvector가 제공하는 벡터 거리 연산자입니다.

```
distance(a, b) = √( Σᵢ (aᵢ - bᵢ)² )   (i = 1 ... 384)
```

| 값 | 의미 |
|----|------|
| distance = 0.0 | 완전히 동일한 벡터 |
| distance 작음 | 두 텍스트의 의미가 유사 |
| distance 큼 | 두 텍스트의 의미가 상이 |

pgvector가 제공하는 거리 연산자 3종 비교:

| 연산자 | 거리 종류 | 이 프로젝트 사용 여부 |
|--------|-----------|----------------------|
| `<->` | L2 유클리드 거리 | **사용** |
| `<#>` | 내적의 음수 | 미사용 |
| `<=>` | 코사인 거리 | 미사용 (정규화로 동등) |

### 파라미터 바인딩 상세

```python
# search_crud.py:30-39
params = {
    "qv": query_vector,
    "ex_ids": tuple(exclude_ids),  # list → tuple 변환 (SQLAlchemy 호환)
    "k": top_k,
}
sql = sql.bindparams(
    bindparam("qv", type_=Vector(EMBEDDING_DIM)),  # 384차원 타입 명시
    bindparam("ex_ids", expanding=True),            # IN 절 동적 확장
    bindparam("k")
)
```

**`type_=Vector(384)`**: SQLAlchemy가 Python `List[float]`를 pgvector 바이너리 포맷으로 직렬화합니다. 이 타입 힌트 없이는 쿼리가 실패합니다.

**`expanding=True`**: `IN (:ex_ids)` 를 입력 리스트 크기에 맞게 자동 확장합니다.

```
입력: exclude_ids = [101, 202, 303]

expanding 없음: WHERE "RECIPE_ID" NOT IN (:ex_ids)         ← 오류
expanding 있음: WHERE "RECIPE_ID" NOT IN (:ex_ids_0, :ex_ids_1, :ex_ids_2)  ← 정상
```

각 값이 별도 바인딩 파라미터로 처리되므로 SQL injection도 방지됩니다.

### DB 테이블 구조 (추정)

```sql
-- RECIPE_VECTOR_TABLE 추정 스키마
CREATE TABLE "RECIPE_VECTOR_TABLE" (
    "RECIPE_ID"   INTEGER PRIMARY KEY,
    "VECTOR_NAME" vector(384)           -- pgvector 타입, 384차원
);

-- 검색 성능을 위한 인덱스 (IVFFlat 또는 HNSW)
CREATE INDEX ON "RECIPE_VECTOR_TABLE"
    USING ivfflat ("VECTOR_NAME" vector_l2_ops);
```

---

## 5. Step 3 — 응답 포맷팅

### 코드 위치

- `app/routers/search.py:39-44`
- `app/models/search.py`

### 응답 스키마

```python
class SearchResultItem(BaseModel):
    recipe_id: int
    distance: float     # L2 거리. 0에 가까울수록 더 유사

class SearchResponse(BaseModel):
    results: List[SearchResultItem]
```

### 응답 예시

```json
{
  "results": [
    { "recipe_id": 1023, "distance": 0.23 },
    { "recipe_id": 2045, "distance": 0.31 },
    { "recipe_id": 3102, "distance": 0.45 },
    ...
  ]
}
```

`distance`는 이 서비스에서 최종 점수로 소비되지 않고 호출자(상위 백엔드)에게 그대로 전달됩니다. 호출자는 이 값을 인기도 가중치, 개인화 점수 등과 결합해 최종 재랭킹에 활용할 수 있습니다.

---

## 6. exclude_ids와 추천 다양성

### 목적

무한 스크롤 또는 "더 보기" 패턴에서 이미 노출한 레시피가 다시 추천되지 않도록 합니다.

### 동작 원리

```
1차 호출
  요청: { "query": "된장찌개", "top_k": 5, "exclude_ids": [] }
  응답: [레시피 A, B, C, D, E]

2차 호출
  요청: { "query": "된장찌개", "top_k": 5, "exclude_ids": [A, B, C, D, E] }
  응답: [레시피 F, G, H, I, J]   ← A~E 제외된 그 다음 유사 레시피

3차 호출
  요청: { "query": "된장찌개", "top_k": 5, "exclude_ids": [A, B, ..., J] }
  응답: [레시피 K, L, M, N, O]
```

### 호출자 책임

exclude_ids 목록의 누적 관리는 이 서비스가 아닌 **상위 백엔드**가 담당합니다. 이 서비스는 전달받은 ID 목록을 `NOT IN` 절로 단순 필터링만 수행합니다.

### 주의사항

exclude_ids 목록이 커질수록 SQL `IN` 절이 길어져 쿼리 성능에 영향을 줄 수 있습니다. `expanding=True` 바인딩으로 파라미터화는 되어 있으나, 매우 많은 ID(수백~수천 개)가 넘어오는 경우 임시 테이블이나 `ANY(ARRAY[...])` 방식 전환을 고려할 수 있습니다.

---

## 7. 성능 측정 구조

`app/routers/search.py` 에는 3개 구간의 시간 측정이 내장되어 있습니다.

```python
start_time = time.time()

# 구간 1: 임베딩 생성
embedding_start = time.time()
query_vector = await encode_text(...)
logger.info(f"쿼리 임베딩 생성 완료: {time.time() - embedding_start:.3f}초")

# 구간 2: DB 검색
db_start = time.time()
results = await search_similar_in_db(...)
logger.info(f"DB 유사도 검색 완료: {time.time() - db_start:.3f}초, 결과 {len(results)}건")

# 전체
logger.info(f"유사도 검색 성공: 총 {time.time() - start_time:.3f}초 소요")
```

이 구조로 병목이 **모델 추론**인지 **DB 조회**인지 로그에서 바로 구분할 수 있습니다.

일반적인 예상 수치:

| 구간 | 예상 소요 시간 |
|------|---------------|
| 임베딩 생성 (모델 캐시 후) | 50~200ms (CPU) |
| DB 벡터 검색 (pgvector 인덱스) | 10~50ms |
| 전체 응답 | 60~250ms |

---

## 8. 설계 원칙과 한계

### 설계 원칙

**단일 책임**: 이 서비스는 순수 벡터 유사도 검색만 담당합니다. 개인화, 인기도 가중치, 카테고리 필터링 같은 비즈니스 규칙은 호출자(상위 백엔드)에서 처리합니다.

**Stateless**: 서비스 자체는 사용자 상태를 저장하지 않습니다. 세션, 히스토리 관리는 모두 호출자 책임입니다.

### 현재 한계

**모델 추론 블로킹**: `model.encode()`는 동기 CPU 연산이므로, `await`로 감싸여 있어도 실제로는 이벤트 루프를 블로킹합니다. 고트래픽 상황에서는 `loop.run_in_executor()`로 스레드풀에 위임하는 것이 적합합니다.

```python
# 현재 (이벤트 루프 블로킹)
embedding = model.encode(text, normalize_embeddings=normalize)

# 개선안 (스레드풀 위임)
import asyncio
loop = asyncio.get_event_loop()
embedding = await loop.run_in_executor(
    None, lambda: model.encode(text, normalize_embeddings=normalize)
)
```

**pgvector 인덱스 미명시**: 쿼리 자체에는 인덱스 힌트가 없으므로, DB 측에서 `IVFFlat` 또는 `HNSW` 인덱스가 생성되어 있어야 실용적인 검색 성능이 나옵니다.

**배치 임베딩 순차 처리**: `embed-batch` 엔드포인트(`app/routers/embedding.py:43`)는 리스트를 루프로 처리하므로 실질적 병렬화가 없습니다. 대량 배치에는 `model.encode(texts)` 배치 API 활용이 효율적입니다.
