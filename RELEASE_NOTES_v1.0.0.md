# Release Notes

## [1.0.0] - 2024-01-15

### 🎉 첫 번째 정식 릴리스

이번 릴리스는 UHOK ML Inference Service의 첫 번째 정식 버전입니다. 레시피 추천을 위한 고성능 임베딩 생성 서비스를 제공합니다.

### ✨ 새로운 기능

#### 핵심 기능
- **텍스트 임베딩 생성**: 한국어 레시피 텍스트를 384차원 벡터로 변환
- **배치 처리**: 단일/다중 텍스트 동시 처리 지원
- **RESTful API**: FastAPI 기반의 표준화된 API 인터페이스
- **헬스 체크**: 서비스 상태 및 모델 정보 실시간 확인

#### AI/ML 기능
- **모델**: `paraphrase-multilingual-MiniLM-L12-v2` (SentenceTransformers 5.0.0)
- **차원**: 384차원 벡터 임베딩
- **언어 지원**: 한국어 최적화, 다국어 지원
- **정규화**: 선택적 벡터 정규화 지원

#### 아키텍처
- **마이크로서비스**: 백엔드와 분리된 독립적 서비스
- **Docker 지원**: 컨테이너화된 배포 및 실행
- **비동기 처리**: FastAPI + Uvicorn 기반 고성능 처리
- **메모리 효율성**: CPU 전용 PyTorch로 리소스 최적화

### 🔧 기술 스택

- **FastAPI 0.116.1**: 고성능 웹 API 프레임워크
- **PyTorch 2.7.1**: CPU 전용 딥러닝 프레임워크
- **SentenceTransformers 5.0.0**: 문장 임베딩 라이브러리
- **Transformers 4.54.1**: HuggingFace 트랜스포머 모델
- **NumPy 2.3.2**: 수치 계산

### 📡 API 엔드포인트

#### 헬스 체크
```http
GET /health
```

#### 단일 텍스트 임베딩
```http
POST /api/v1/embed
Content-Type: application/json

{
  "text": "갈비탕",
  "normalize": true
}
```

#### 배치 텍스트 임베딩
```http
POST /api/v1/embed-batch
Content-Type: application/json

{
  "texts": ["갈비탕", "김치찌개", "된장찌개"],
  "normalize": true
}
```

#### 모델 정보 조회
```http
GET /api/v1/model-info
```

### 🐳 배포

#### Docker 실행
```bash
# 이미지 빌드
docker build -t uhok-ml-inference .

# 컨테이너 실행
docker run -p 8001:8001 uhok-ml-inference
```

#### Docker Compose 통합
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

### 📊 성능 특성

- **첫 요청**: 10-30초 (모델 로딩 시간)
- **이후 요청**: 100-300ms (CPU 기반)
- **메모리**: 1-2GB (모델 + 런타임)
- **배치 처리**: 3개 텍스트 동시 처리 지원

### 🔗 백엔드 통합

#### 환경 변수 설정
```bash
ML_MODE=remote_embed
ML_INFERENCE_URL=http://ml-inference:8001
ML_TIMEOUT=3.0
ML_RETRIES=2
```

#### 통합 방식
- HTTP API를 통한 임베딩 생성
- 타임아웃, 재시도, 폴백 메커니즘
- 구조화된 JSON 로그 출력
- 헬스체크 및 메트릭 수집

### 🧪 테스트

#### 단위 테스트
```bash
python test_ml_service.py
```

#### 통합 테스트
```bash
# 헬스체크
curl http://localhost:8001/health

# 임베딩 생성
curl -X POST http://localhost:8001/api/v1/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "갈비탕", "normalize": true}'
```

### 📈 모니터링

- **구조화된 JSON 로그**: 파싱 가능한 로그 형식
- **요청 추적**: 요청 ID 기반 로그 추적
- **성능 메트릭**: 응답 시간, 처리량 측정
- **에러 로깅**: 상세한 에러 정보 및 스택 트레이스

### 🚨 알려진 제한사항

#### 성능 제한
- **콜드스타트**: 첫 요청 시 모델 로딩으로 인한 지연
- **단일 워커**: 동시 처리 제한 (순차 처리)
- **메모리 사용량**: 모델 크기로 인한 높은 메모리 사용

#### 운영 제한
- **네트워크 의존성**: 백엔드와의 네트워크 연결 필수
- **에러 처리**: ML 서비스 장애 시 백엔드 추천 기능 중단
- **스케일링**: 수동 스케일링 (자동 오토스케일링 미지원)

### 🔄 마이그레이션 가이드

#### 기존 시스템에서 마이그레이션
1. 백엔드 설정 변경: `ML_MODE=remote_embed` 설정
2. ML 서비스 배포: Docker Compose로 ML 서비스 실행
3. 네트워크 설정: 내부 통신 설정 확인
4. 테스트 실행: 통합 테스트 및 성능 검증
5. 점진적 전환: 트래픽을 점진적으로 전환

#### 롤백 계획
```bash
# 긴급 롤백 (로컬 모드로 복귀)
echo "ML_MODE=local" > uhok-backend/.env
docker-compose restart backend
```

### 🛠️ 개발자 가이드

#### 로컬 개발 환경 설정
```bash
# 의존성 설치
pip install -r requirements.txt

# 개발 서버 실행
python -m app.main
```

#### 디버깅
```bash
# 로그 확인
docker-compose logs -f ml-inference

# 메모리 사용량 확인
docker stats uhok-ml-inference
```

### 📞 지원

#### 문제 해결
1. **로그 확인**: `docker-compose logs -f ml-inference`
2. **헬스체크**: `curl http://localhost:8001/health`
3. **네트워크 확인**: `docker-compose exec backend ping ml-inference`
4. **환경변수 확인**: `docker-compose exec backend env | grep ML_`

#### 문서
- **API 문서**: `/docs` 엔드포인트 (Swagger UI)
- **통합 가이드**: `INTEGRATION_GUIDE.md`
- **환경 설정**: `ENVIRONMENT_SETUP.md`

### 🎯 향후 계획

#### v1.1.0 (예정)
- GPU 지원: CUDA 기반 가속 처리
- 배치 크기 최적화: 더 큰 배치 처리 지원
- 캐싱: 임베딩 결과 캐싱 기능

#### v1.2.0 (예정)
- 다중 모델: 여러 모델 동시 지원
- A/B 테스트: 모델 성능 비교 기능
- 메트릭 수집: Prometheus/Grafana 연동

### 🏆 기여자

- **개발팀**: UHOK 개발팀
- **ML팀**: 머신러닝 모델 최적화
- **DevOps팀**: 인프라 및 배포 자동화

---

**UHOK ML Inference Service v1.0.0** - 레시피 추천을 위한 고성능 임베딩 생성 서비스
