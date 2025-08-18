# UHOK ML 추론 서비스

레시피 추천을 위한 ML 추론 서비스입니다.

## 🚀 기능

- **텍스트 임베딩 생성**: Sentence Transformers를 사용한 다국어 텍스트 임베딩
- **벡터 유사도 검색**: scikit-learn 기반 벡터 검색 (백엔드와 동일)
- **FAISS 성능 비교**: FAISS 기반 벡터 검색 (성능 비교용)
- **배치 처리**: 대량 텍스트의 효율적인 임베딩 생성
- **성능 측정**: 상세한 시간 측정 및 통계

## 📁 구조

```
uhok-ml-inference/
├── app/                    # FastAPI 애플리케이션
│   ├── main.py            # 메인 애플리케이션
│   ├── api.py             # API 엔드포인트
│   └── deps.py            # 의존성 주입
├── models/                 # ML 모델
│   └── sentence_transformers.py
├── store/                  # 벡터 스토어
│   ├── vector_store.py    # scikit-learn 기반 (백엔드와 동일)
│   └── vector_store_faiss.py  # FAISS 기반 (성능 비교용)
├── Dockerfile
├── requirements.txt
└── README.md
```

## 🛠️ 설치 및 실행

### 로컬 실행
```bash
# 의존성 설치
pip install -r requirements.txt

# 서비스 실행
python -m app.main
```

### Docker 실행
```bash
# 이미지 빌드
docker build -t uhok-ml-inference .

# 컨테이너 실행
docker run -p 8080:8080 uhok-ml-inference
```

## 📡 API 엔드포인트

### 헬스체크
- `GET /api/health` - 서비스 상태 확인

### 임베딩 생성
- `POST /api/v1/embed` - 단일 텍스트 임베딩
- `POST /api/v1/embed-batch` - 배치 텍스트 임베딩

### 벡터 검색
- `POST /api/v1/search` - 유사도 검색 (scikit-learn)

### 성능 비교
- `POST /api/v1/performance-comparison` - scikit-learn vs FAISS 성능 비교
- `GET /api/v1/performance-stats` - 현재 성능 통계 조회

## 🔧 설정

### 환경 변수
- `ML_MODEL_NAME`: 사용할 모델 이름 (기본값: paraphrase-multilingual-MiniLM-L12-v2)
- `ML_DEVICE`: 사용할 디바이스 (cpu/cuda, 자동 감지)

### 모델 정보
- **모델**: paraphrase-multilingual-MiniLM-L12-v2
- **차원**: 384
- **언어**: 다국어 (한국어 포함)
- **정규화**: L2 정규화 지원

## 📊 성능

- **임베딩 생성**: ~10ms (CPU), ~5ms (GPU)
- **검색 속도**: 
  - scikit-learn: ~5-50ms (벡터 수에 따라)
  - FAISS: ~1-10ms (벡터 수에 따라)
- **동시 처리**: 비동기 지원
- **성능 측정**: 상세한 통계 (평균, P95, P99 등)

## 🔄 백엔드 연동

이 서비스는 UHOK 백엔드와 연동되어 사용됩니다:

1. **전략 A**: 원격 임베딩 + 로컬 pgvector
2. **전략 B**: 원격 벡터 스토어

## 🚨 주의사항

- 첫 실행 시 모델 다운로드로 인한 지연 발생
- GPU 사용 시 CUDA 메모리 관리 필요
- 대용량 인덱스 사용 시 디스크 공간 확보 필요
- FAISS 사용 시 C++ 컴파일 환경 필요 (Docker에서 자동 처리)

## 🧪 성능 테스트

### 자동 테스트 스크립트
```bash
# 성능 비교 테스트 실행
python performance_test_script.py
```

### API 테스트
```bash
# 성능 비교 API 호출
curl -X POST "http://localhost:8080/api/v1/performance-comparison" \
  -H "Content-Type: application/json" \
  -d '{"query": "김치찌개", "top_k": 25, "iterations": 10}'
```
