"""
ML 추론 서비스 API 엔드포인트
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import time

from app.deps import get_embedding_model, get_vector_store
# from store.vector_store_faiss import VectorStoreFAISS  # Python 3.13.5 호환성 문제로 제거

router = APIRouter()

# 요청/응답 모델
class EmbedRequest(BaseModel):
    text: str
    normalize: bool = True

class EmbedResponse(BaseModel):
    embedding: List[float]
    dim: int
    version: str

class EmbedBatchRequest(BaseModel):
    texts: List[str]
    normalize: bool = True

class EmbedBatchResponse(BaseModel):
    embeddings: List[List[float]]
    dim: int
    version: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 25
    exclude_ids: Optional[List[int]] = None

class SearchResult(BaseModel):
    recipe_id: int
    distance: float

class SearchResponse(BaseModel):
    results: List[SearchResult]

class PerformanceComparisonRequest(BaseModel):
    query: str
    top_k: int = 25
    exclude_ids: Optional[List[int]] = None
    iterations: int = 10  # 성능 측정 반복 횟수

class PerformanceComparisonResponse(BaseModel):
    scikit_learn_stats: dict
    faiss_stats: dict
    comparison: dict

@router.get("/health")
async def health_check():
    """헬스체크 엔드포인트"""
    model = get_embedding_model()
    return {
        "status": "ok",
        "model": model.get_model_name(),
        "dim": model.get_dimension(),
        "version": model.get_version(),
        "timestamp": time.time()
    }

@router.post("/v1/embed", response_model=EmbedResponse)
async def create_embedding(request: EmbedRequest):
    """단일 텍스트 임베딩 생성"""
    try:
        model = get_embedding_model()
        embedding = await model.embed_text(request.text, normalize=request.normalize)
        
        return EmbedResponse(
            embedding=embedding,
            dim=model.get_dimension(),
            version=model.get_version()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"임베딩 생성 실패: {str(e)}")

@router.post("/v1/embed-batch", response_model=EmbedBatchResponse)
async def create_embeddings_batch(request: EmbedBatchRequest):
    """배치 텍스트 임베딩 생성"""
    try:
        model = get_embedding_model()
        embeddings = await model.embed_texts_batch(request.texts, normalize=request.normalize)
        
        return EmbedBatchResponse(
            embeddings=embeddings,
            dim=model.get_dimension(),
            version=model.get_version()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"배치 임베딩 생성 실패: {str(e)}")

@router.post("/v1/search", response_model=SearchResponse)
async def search_similar(request: SearchRequest):
    """벡터 유사도 검색 (전략 B용)"""
    try:
        vector_store = get_vector_store()
        results = await vector_store.search_similar(
            query=request.query,
            top_k=request.top_k,
            exclude_ids=request.exclude_ids or []
        )
        
        return SearchResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 실패: {str(e)}")

@router.post("/v1/performance-comparison", response_model=PerformanceComparisonResponse)
async def compare_performance(request: PerformanceComparisonRequest):
    """scikit-learn 성능 측정 (FAISS 제거됨)"""
    try:
        # scikit-learn 벡터 스토어만 사용
        scikit_store = VectorStore()
        await scikit_store.initialize()
        
        # 성능 통계 초기화
        scikit_store.reset_performance_stats()
        
        # 성능 측정 실행
        for i in range(request.iterations):
            await scikit_store.search_similar(
                query=request.query,
                top_k=request.top_k,
                exclude_ids=request.exclude_ids or []
            )
        
        # 성능 통계 수집
        scikit_stats = scikit_store.get_performance_stats()
        
        # FAISS는 제거되었으므로 더미 데이터
        faiss_stats = {
            "status": "removed",
            "message": "FAISS는 Python 3.13.5 호환성 문제로 제거됨"
        }
        
        # 성능 비교 분석
        comparison = {
            "scikit_learn_faster": True,
            "speedup_ratio": 1.0,
            "recommendation": "scikit-learn",
            "note": "FAISS 제거로 scikit-learn만 사용"
        }
        
        # 리소스 정리
        await scikit_store.cleanup()
        
        return PerformanceComparisonResponse(
            scikit_learn_stats=scikit_stats,
            faiss_stats=faiss_stats,
            comparison=comparison
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"성능 측정 실패: {str(e)}")

@router.get("/v1/performance-stats")
async def get_performance_stats():
    """현재 성능 통계 조회 (FAISS 제거됨)"""
    try:
        # scikit-learn 벡터 스토어만 사용
        scikit_store = VectorStore()
        await scikit_store.initialize()
        
        scikit_stats = scikit_store.get_performance_stats()
        
        # FAISS는 제거되었으므로 더미 데이터
        faiss_stats = {
            "status": "removed",
            "message": "FAISS는 Python 3.13.5 호환성 문제로 제거됨"
        }
        
        # 리소스 정리
        await scikit_store.cleanup()
        
        return {
            "scikit_learn": scikit_stats,
            "faiss": faiss_stats,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"성능 통계 조회 실패: {str(e)}")
