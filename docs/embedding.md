# 임베딩 생성 로직 상세 문서

## 목차

1. [개요](#1-개요)
2. [모델: paraphrase-multilingual-MiniLM-L12-v2](#2-모델-paraphrase-multilingual-minilm-l12-v2)
3. [싱글턴 캐시 — Double-Checked Locking](#3-싱글턴-캐시--double-checked-locking)
4. [임베딩 API](#4-임베딩-api)
5. [정규화 옵션](#5-정규화-옵션)

---

## 1. 개요

임베딩(Embedding)은 텍스트의 의미를 고정 차원의 숫자 벡터로 표현하는 기술입니다. 의미적으로 유사한 텍스트는 벡터 공간에서 가까운 위치에 놓입니다.

```
"된장찌개"       → [0.12, -0.34, 0.87, ...]  (384차원)
"된장국 끓이기"  → [0.11, -0.32, 0.85, ...]  (384차원, 위와 가까움)
"자전거 수리"    → [-0.78, 0.21, -0.43, ...] (384차원, 위와 멂)
```

이 서비스는 `SentenceTransformers` 라이브러리의 `paraphrase-multilingual-MiniLM-L12-v2` 모델로 텍스트를 384차원 벡터로 변환합니다.

---

## 2. 모델: paraphrase-multilingual-MiniLM-L12-v2

### 모델 스펙

| 항목 | 값 |
|------|-----|
| 출력 차원 | 384 |
| 지원 언어 | 50개 이상 (한국어 포함) |
| 아키텍처 | MiniLM 12레이어 (BERT 경량화) |
| 파라미터 수 | 약 1억 1,800만 |
| 특화 | 의미적 유사도 (paraphrase detection) |
| 실행 장치 | CPU (이 서비스) |

### 아키텍처 내부 흐름

```
입력 텍스트 (최대 512 토큰)
        │
        ▼
WordPiece 토크나이저
→ 텍스트를 서브워드 단위로 분할
→ 각 토큰을 정수 ID로 변환
→ [CLS] + 토큰 시퀀스 + [SEP] 형태로 구성
        │
        ▼
토큰 임베딩 레이어
→ 각 토큰 ID → 384차원 초기 벡터
→ 위치 임베딩(Positional Encoding) 추가
        │
        ▼
Transformer 인코더 × 12레이어
→ 각 레이어: Self-Attention + Feed-Forward + LayerNorm
→ Self-Attention: 각 토큰이 다른 토큰들과의 관계를 학습
  ("된장"이 "찌개"와 함께 나올 때의 의미 파악)
        │
        ▼
Mean Pooling
→ 모든 토큰의 히든 스테이트를 평균
→ 가변 길이 시퀀스 → 고정 384차원 벡터 (문장 레벨)
        │
        ▼
(선택) L2 정규화
→ 벡터 크기를 1.0으로 정규화
→ 출력: 384차원 단위 벡터
```

### 왜 이 모델인가

- **다국어 지원**: 한국어 레시피 텍스트를 별도 파인튜닝 없이 처리 가능
- **경량화**: MiniLM은 원본 BERT 대비 속도/메모리 효율이 높음
- **paraphrase 학습**: "끓이는 법"과 "만드는 방법"이 유사하다고 인식 가능
- **384차원**: 768차원 대비 저장 공간과 연산량 절반, 품질은 유사

---

## 3. 싱글턴 캐시 — Double-Checked Locking

### 코드 위치

`app/deps.py:43-83`

### 전체 코드

```python
_model: Optional[SentenceTransformer] = None
_model_lock = asyncio.Lock()
_model_info: Optional[Dict[str, Any]] = None

async def get_model() -> SentenceTransformer:
    global _model, _model_info

    # ① 1차 체크: Lock 없이 빠른 경로 (모델이 이미 로드된 99.9% 케이스)
    if _model is not None:
        logger.debug("캐시된 SentenceTransformer 모델 사용 중")
        return _model

    start_time = time.time()

    # ② Lock 획득: 최초 1회 로딩 시에만 진입
    async with _model_lock:
        # ③ 2차 체크: Lock 획득 전 다른 코루틴이 먼저 로드했을 수 있음
        if _model is None:
            _model = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2",
                device="cpu"
            )
            _model_info = {
                "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
                "dimension": 384,
                "version": "sentence-transformers-5.0.0",
                "device": "cpu"
            }
        else:
            logger.debug("다른 코루틴에서 모델 로드 완료")

    return _model
```

### 왜 Double-Checked인가

Lock 없이 1차 체크만 하면 동시성 문제가 발생할 수 있습니다.

```
코루틴 A: _model is None 확인 → True
코루틴 B: _model is None 확인 → True  (A가 아직 로드 전)
코루틴 A: 모델 로드 시작
코루틴 B: 모델 로드 시작 (중복 로드!)
```

Lock만 있고 2차 체크가 없으면:

```
코루틴 A: Lock 획득 → 모델 로드 → Lock 해제
코루틴 B: Lock 획득 → 모델 로드 (중복 로드! — 2차 체크 없으면)
```

Double-Checked Locking으로 두 문제를 모두 해결합니다:

```
코루틴 A: ①번 통과 → Lock 대기
코루틴 B: ①번 통과 → Lock 획득 → ③번 True → 모델 로드 → Lock 해제
코루틴 A: Lock 획득 → ③번 False (_model이 이미 있음) → 중복 로드 건너뜀
이후 모든 요청: ①번에서 즉시 반환 (Lock 진입 없음)
```

### asyncio.Lock의 범위와 실제 필요성

`asyncio.Lock()`은 **같은 이벤트 루프(프로세스) 안에서만** 유효합니다. gunicorn 등으로 멀티 워커를 운용하면 각 워커 프로세스가 독립적으로 모델을 1번씩 로드합니다.

asyncio는 단일 스레드 이벤트 루프이므로 코루틴 간 진정한 동시 실행은 없습니다. 그러나 `model.encode()`가 실행되기 전까지는 `await` 지점에서 다른 코루틴에게 제어권이 넘어갈 수 있습니다. Lock은 "모델 로드 중인데 또 다른 로드 요청이 들어오는" 구간을 보호합니다. 현재 코드는 모델 로드 중 `await`가 없어 실제로 race condition이 발생하기 어렵지만, 방어적 설계로 유지합니다.

### 콜드 스타트 시간

SentenceTransformer 모델 로드는 처음 1회에 10~30초가 소요됩니다. 이후 캐시된 모델을 재사용하므로 추가 비용 없습니다.

---

## 4. 임베딩 API

### 단건 임베딩 — `POST /api/v1/embed`

**코드**: `app/routers/embedding.py:12-33`

```python
class EmbedRequest(BaseModel):
    text: str   = Field(..., min_length=1, max_length=1000)
    normalize: bool = Field(True)

class EmbedResponse(BaseModel):
    embedding: List[float]   # 384차원 float 배열
    dim: int                 # 384
    version: str             # 모델 버전 문자열
```

요청:
```json
{ "text": "된장찌개 만드는 법", "normalize": true }
```

응답:
```json
{
  "embedding": [0.12, -0.34, 0.87, ...],
  "dim": 384,
  "version": "sentence-transformers-5.0.0"
}
```

### 배치 임베딩 — `POST /api/v1/embed-batch`

**코드**: `app/routers/embedding.py:35-57`

```python
class EmbedBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)
    normalize: bool  = Field(True)

class EmbedBatchResponse(BaseModel):
    embeddings: List[List[float]]   # 텍스트 수 × 384 행렬
    dim: int
    version: str
    count: int
```

요청:
```json
{
  "texts": ["된장찌개 만드는 법", "김치볶음밥 레시피"],
  "normalize": true
}
```

응답:
```json
{
  "embeddings": [
    [0.12, -0.34, ...],
    [0.45, 0.21, ...]
  ],
  "dim": 384,
  "version": "sentence-transformers-5.0.0",
  "count": 2
}
```

**주의**: 현재 배치 처리는 루프 방식입니다 (`app/routers/embedding.py:43`).

```python
# 현재 구현: 텍스트를 하나씩 순차 인코딩
embeddings = [await encode_text(text, request.normalize) for text in request.texts]
```

`model.encode(texts)` 배치 API를 사용하면 내부 패딩 최적화로 더 효율적입니다.

```python
# 개선안: 배치 한 번에 인코딩 (내부적으로 최적 패딩 처리)
model = await get_model()
embeddings_np = model.encode(request.texts, normalize_embeddings=request.normalize)
embeddings = [e.tolist() for e in embeddings_np]
```

레시피 대량 임베딩(DB 초기 적재 등) 작업에는 이 방식이 훨씬 효율적입니다.

---

## 5. 정규화 옵션

`normalize` 파라미터는 `encode_text`와 임베딩 API 양쪽에 존재합니다.

| normalize | 벡터 크기 | 용도 |
|-----------|----------|------|
| `True` (기본값) | 항상 1.0 | 유사도 검색, DB 저장, 검색 쿼리 |
| `False` | 텍스트 길이에 따라 가변 | 벡터 크기 자체가 의미를 가지는 경우 |

검색/추천 목적이라면 항상 `normalize=True`를 사용해야 DB에 저장된 벡터(정규화됨)와 쿼리 벡터(정규화됨) 간의 거리 계산이 정확하게 동작합니다. DB 저장 벡터와 쿼리 벡터의 정규화 여부가 다르면 검색 결과가 왜곡됩니다.
