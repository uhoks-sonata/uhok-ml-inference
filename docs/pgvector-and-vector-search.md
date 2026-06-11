# pgvector 및 벡터 유사도 검색 심층 문서

## 목차

1. [벡터 검색이란](#1-벡터-검색이란)
2. [pgvector 개요](#2-pgvector-개요)
3. [이 프로젝트에서의 사용 방식](#3-이-프로젝트에서의-사용-방식)
4. [거리 연산자 비교](#4-거리-연산자-비교)
5. [정규화와 L2 거리](#5-정규화와-l2-거리)
6. [파라미터 바인딩 상세](#6-파라미터-바인딩-상세)
7. [인덱스 전략](#7-인덱스-전략)
8. [성능 특성](#8-성능-특성)

---

## 1. 벡터 검색이란

전통적인 키워드 검색(LIKE, Full-Text Search)은 텍스트의 **표면적 형태**를 비교합니다.

```
쿼리: "된장국"
LIKE 검색 결과: "된장국 레시피" ← 매칭
                "된장찌개 끓이기" ← 미매칭 (다른 단어)
```

벡터 검색은 텍스트의 **의미**를 비교합니다.

```
쿼리: "된장국"
벡터: [0.12, -0.34, 0.87, ...]  (384차원)

DB의 "된장찌개 끓이기" 벡터: [0.11, -0.32, 0.85, ...]  (거리: 0.05 ← 가까움)
DB의 "자전거 수리법" 벡터:    [-0.78, 0.21, -0.43, ...] (거리: 2.31 ← 멂)
```

---

## 2. pgvector 개요

pgvector는 PostgreSQL의 공식 Extension으로, 벡터 데이터 타입과 유사도 연산자를 DB 레벨에서 제공합니다.

### 제공 기능

```sql
-- 벡터 타입
CREATE TABLE items (
    id     SERIAL PRIMARY KEY,
    vector vector(384)   -- 384차원 float4 배열
);

-- 벡터 삽입
INSERT INTO items (vector) VALUES ('[0.1, 0.2, 0.3, ...]');

-- 유사도 검색
SELECT id, vector <-> '[0.1, 0.2, ...]' AS distance
FROM items
ORDER BY distance
LIMIT 10;
```

### 내부 저장 형식

`vector(384)` 타입은 내부적으로 `float4` (32비트 부동소수점) 배열로 저장됩니다.

```
384차원 벡터 = 384 × 4 bytes = 1,536 bytes/레코드
100만 레시피 = 약 1.5GB (벡터 컬럼만)
```

---

## 3. 이 프로젝트에서의 사용 방식

### 테이블 구조

```sql
-- RECIPE_VECTOR_TABLE (추정 스키마)
CREATE TABLE "RECIPE_VECTOR_TABLE" (
    "RECIPE_ID"   INTEGER PRIMARY KEY,
    "VECTOR_NAME" vector(384)
);
```

상수 정의 (`app/crud/search_crud.py:11-12`):

```python
EMBEDDING_DIM = 384
VECTOR_COL = '"VECTOR_NAME"'
```

### 실행 쿼리 (exclude_ids 없을 때)

```sql
SELECT "RECIPE_ID" AS recipe_id,
       "VECTOR_NAME" <-> :qv AS distance
FROM "RECIPE_VECTOR_TABLE"
ORDER BY distance ASC
LIMIT :k
```

### 실행 쿼리 (exclude_ids 있을 때)

```sql
SELECT "RECIPE_ID" AS recipe_id,
       "VECTOR_NAME" <-> :qv AS distance
FROM "RECIPE_VECTOR_TABLE"
WHERE "RECIPE_ID" NOT IN :ex_ids
ORDER BY distance ASC
LIMIT :k
```

---

## 4. 거리 연산자 비교

pgvector는 3가지 거리 연산자를 제공합니다.

### `<->` L2 유클리드 거리 (이 프로젝트 사용)

```
distance(a, b) = √( Σᵢ (aᵢ - bᵢ)² )
```

- 기하학적 거리. 두 점 사이의 직선 거리.
- 값 범위: `[0, ∞)` — 0일수록 동일, 클수록 상이
- 정규화된 벡터에서는 코사인 유사도와 동일한 순위

### `<=>` 코사인 거리

```
distance(a, b) = 1 - (a·b) / (||a|| × ||b||)
```

- 두 벡터의 **방향 차이**만 측정. 크기는 무시.
- 값 범위: `[0, 2]`
- 텍스트 길이가 다른 문서 비교에 적합

### `<#>` 내적의 음수 (Negative Inner Product)

```
distance(a, b) = -(a·b) = -Σᵢ (aᵢ × bᵢ)
```

- 값이 작을수록(음수가 클수록) 유사
- 정규화된 벡터에서는 코사인 거리와 동등
- 추천 시스템에서 종종 사용

### 비교 요약

| 연산자 | 측정 기준 | 정규화 벡터에서 | 이 프로젝트 |
|--------|----------|----------------|------------|
| `<->` | 절대적 거리 | 코사인과 동순위 | **사용** |
| `<=>` | 방향 차이 | L2와 동순위 | 미사용 |
| `<#>` | 내적 | 코사인과 동순위 | 미사용 |

---

## 5. 정규화와 L2 거리

이 프로젝트가 `<->` (L2 거리)를 쓰면서 `normalize=True`를 기본값으로 하는 이유입니다.

### 수학적 증명

단위 벡터 (||a|| = ||b|| = 1) 에서:

```
L2 거리² = ||a - b||²
         = ||a||² - 2(a·b) + ||b||²
         = 1 - 2(a·b) + 1
         = 2 - 2(a·b)
         = 2 × (1 - 코사인 유사도)

따라서:
L2 거리 = √(2 × (1 - 코사인 유사도))
```

L2 거리가 작아질수록 코사인 유사도는 커집니다. 순위(rank)가 동일합니다.

### 실제 예시

```
쿼리 벡터 q = [1.0, 0.0]  (정규화됨)

벡터 a = [0.99, 0.14]  (코사인 유사도 = 0.99)
벡터 b = [0.70, 0.71]  (코사인 유사도 = 0.70)

L2 거리(q, a) = √((1-0.99)² + (0-0.14)²) = √(0.0001 + 0.0196) ≈ 0.14
L2 거리(q, b) = √((1-0.70)² + (0-0.71)²) = √(0.09 + 0.50) ≈ 0.77

→ a가 q에 더 가까움 (L2 기준, 코사인 기준 둘 다 동일)
```

### 주의사항

DB에 저장된 레시피 벡터와 검색 쿼리 벡터의 **정규화 여부가 반드시 일치**해야 합니다. 한쪽만 정규화된 상태에서 비교하면 거리 값이 왜곡됩니다.

---

## 6. 파라미터 바인딩 상세

### Vector 타입 바인딩

```python
from pgvector.sqlalchemy import Vector

bindparam("qv", type_=Vector(EMBEDDING_DIM))
```

`type_=Vector(384)`가 없으면 SQLAlchemy는 `List[float]`를 일반 배열로 처리하여 pgvector 포맷 불일치 오류가 발생합니다. 이 타입 힌트가 Python `List[float]` → pgvector 바이너리 직렬화를 담당합니다.

### expanding=True 바인딩

```python
bindparam("ex_ids", expanding=True)
```

`expanding=True` 없이 `IN (:ex_ids)` 에 리스트를 바인딩하면 오류가 납니다.

```python
# 입력
exclude_ids = [101, 202, 303]

# expanding=False (기본, 오류)
WHERE "RECIPE_ID" NOT IN (:ex_ids)
# → NOT IN ([101, 202, 303])  ← 단일 값 취급, 오류

# expanding=True (정상)
WHERE "RECIPE_ID" NOT IN (:ex_ids_0, :ex_ids_1, :ex_ids_2)
# → NOT IN (101, 202, 303)   ← 각각 독립 바인딩, 정상
```

각 값이 별도 바인딩 파라미터로 처리되므로 SQL injection도 방지됩니다.

### 쿼리 분기 이유

```python
if exclude_ids:
    # NOT IN 절 포함 쿼리
else:
    # NOT IN 절 없는 단순 쿼리
```

`NOT IN ()`에 빈 리스트를 넣으면 SQL 문법 오류가 발생하기 때문에 분기 처리가 필요합니다.

```sql
-- 빈 리스트: 오류
WHERE "RECIPE_ID" NOT IN ()

-- 올바른 처리: NOT IN 절 자체를 제거
SELECT ... FROM ... ORDER BY distance LIMIT :k
```

---

## 7. 인덱스 전략

현재 코드에 인덱스 생성 DDL은 없습니다. 실용적인 검색 성능을 위해서는 DB 측에서 벡터 인덱스가 필요합니다.

### IVFFlat 인덱스

```sql
CREATE INDEX ON "RECIPE_VECTOR_TABLE"
    USING ivfflat ("VECTOR_NAME" vector_l2_ops)
    WITH (lists = 100);
```

- 벡터 공간을 `lists`개 클러스터로 분할
- 검색 시 가까운 클러스터만 탐색 → 속도 향상, 정확도 약간 손실
- `lists` 권장값: 레코드 수의 √N ~ N/1000

### HNSW 인덱스 (pgvector 0.5+)

```sql
CREATE INDEX ON "RECIPE_VECTOR_TABLE"
    USING hnsw ("VECTOR_NAME" vector_l2_ops)
    WITH (m = 16, ef_construction = 64);
```

- 계층적 그래프 기반 근사 최근접 이웃 탐색
- IVFFlat보다 빠른 검색 속도, 더 많은 메모리 사용
- 대용량 데이터셋에 권장

### 인덱스 없을 때 (Sequential Scan)

인덱스 없이 전체 테이블을 순차 스캔합니다. 레시피 수가 수백만 이상이면 응답 시간이 급격히 증가합니다.

| 레코드 수 | Sequential Scan | IVFFlat | HNSW |
|---------|----------------|---------|------|
| 10만 | ~100ms | ~5ms | ~3ms |
| 100만 | ~1s | ~20ms | ~10ms |
| 1000만 | ~10s | ~50ms | ~20ms |

---

## 8. 성능 특성

### 검색 응답 시간 구성

```
전체 응답 시간 = 임베딩 생성 + DB 검색 + 네트워크

임베딩 생성 (CPU): 50~200ms   ← 주 병목
DB 검색 (인덱스): 10~50ms
네트워크 (내부): 1~5ms
```

### 동시 요청 처리

`model.encode()`는 동기 CPU 연산이므로 비동기 서버(uvicorn)에서도 이벤트 루프를 블로킹합니다.

```
동시 요청 10개 → 순차 처리 (각 100ms) → 마지막 요청 응답 시간: ~1초
```

고트래픽 대응 방안:

```python
# 스레드풀로 CPU 바운드 작업 오프로드
import asyncio

async def encode_text(text: str, normalize: bool = True) -> list:
    model = await get_model()
    loop = asyncio.get_event_loop()
    embedding = await loop.run_in_executor(
        None,
        lambda: model.encode(text, normalize_embeddings=normalize)
    )
    return embedding.tolist()
```

### 메모리 사용량

| 구성 요소 | 메모리 |
|---------|--------|
| SentenceTransformer 모델 | ~500MB |
| PyTorch 런타임 | ~200MB |
| FastAPI + uvicorn | ~50MB |
| **전체 서비스** | **~1~1.5GB** |
