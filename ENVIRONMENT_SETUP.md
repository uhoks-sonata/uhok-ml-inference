# ML 서비스 환경 설정 가이드

## 백엔드 환경 변수 설정

백엔드 서비스의 `.env` 파일에 다음 설정을 추가하세요:

```bash
# ==================== [ML 서비스 설정] ====================
# ML 모드 설정: local, remote_embed, remote_store
ML_MODE=local                    # 개발 시: local, 운영 시: remote_embed

# 원격 ML 서비스 URL (ML_MODE가 remote_*일 때 사용)
ML_INFERENCE_URL=http://ml-inference:8001

# ML 서비스 타임아웃 및 재시도 설정
ML_TIMEOUT=3.0
ML_RETRIES=2
```

## 모드별 설정

### 1. 로컬 모드 (개발용)
```bash
ML_MODE=local
```
- 백엔드에서 직접 모델 로딩
- 개발 및 테스트 시 사용
- 모델이 백엔드 컨테이너에 포함됨

### 2. 원격 임베딩 모드 (운영용)
```bash
ML_MODE=remote_embed
ML_INFERENCE_URL=http://ml-inference:8001
ML_TIMEOUT=3.0
ML_RETRIES=2
```
- ML 서비스에서 임베딩 생성
- 백엔드에서 PostgreSQL pgvector로 검색
- 운영 환경에서 권장

### 3. 원격 벡터 스토어 모드 (고급)
```bash
ML_MODE=remote_store
ML_INFERENCE_URL=http://ml-inference:8001
ML_TIMEOUT=3.0
ML_RETRIES=2
```
- ML 서비스에서 직접 벡터 검색
- 백엔드에서 ID만 받아 상세 정보 조회
- 향후 확장 시 사용

## Docker Compose 실행

### 기본 실행 (로컬 모드)
```bash
cd uhok-deploy
docker-compose up -d
```

### ML 서비스 포함 실행 (원격 모드)
```bash
cd uhok-deploy
docker-compose --profile with-ml up -d
```

### ML 서비스만 실행
```bash
cd uhok-deploy
docker-compose up -d ml-inference
```

## 네트워크 설정

ML 서비스는 내부 네트워크(`app_net`)에서만 접근 가능합니다:
- 외부 포트: 노출되지 않음 (보안)
- 내부 포트: `8001`
- 백엔드에서 `http://ml-inference:8001`로 접근

## 헬스체크

ML 서비스 상태 확인:
```bash
# 컨테이너 내부에서
curl http://ml-inference:8001/health

# 외부에서 (포트 포워딩 시)
curl http://localhost:8001/health
```

## 모니터링

### 로그 확인
```bash
# ML 서비스 로그
docker-compose logs -f ml-inference

# 백엔드 로그 (ML 호출 관련)
docker-compose logs -f backend | grep -i "ml\|embedding"
```

### 성능 모니터링
- ML 서비스: 요청 수, 응답 시간, 에러율
- 백엔드: ML 호출 성공/실패율, 타임아웃 발생률

## 문제 해결

### 1. ML 서비스 연결 실패
```bash
# 네트워크 연결 확인
docker-compose exec backend ping ml-inference

# 서비스 상태 확인
docker-compose ps ml-inference
```

### 2. 모델 로딩 실패
```bash
# 모델 캐시 확인
docker-compose exec ml-inference ls -la /models/hf_cache

# 모델 수동 다운로드
docker-compose exec ml-inference python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"
```

### 3. 메모리 부족
- ML 서비스 메모리 제한 설정
- 모델 캐시 볼륨 크기 조정
- 백엔드에서 ML 호출 빈도 조절

## 운영 환경 배포

### EC2 배포 시 고려사항
1. **인스턴스 타입**: ML 서비스용 별도 인스턴스 권장
2. **메모리**: 최소 4GB (모델 + 런타임)
3. **네트워크**: VPC 내부 통신 설정
4. **보안 그룹**: 내부 통신만 허용
5. **로드 밸런서**: ALB/NLB로 트래픽 분산

### 비용 최적화
- ML 서비스: `t3.medium` 이상
- 백엔드: `t3.small` (모델 제거로 경량화)
- 스팟 인스턴스 활용 고려
