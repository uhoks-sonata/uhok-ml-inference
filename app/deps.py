"""
ML 추론 서비스 의존성 주입 및 리소스 관리
"""

from typing import Optional
from models.sentence_transformers import SentenceTransformerModel
from store.vector_store import VectorStore

# 전역 변수로 모델과 벡터 스토어 인스턴스 관리
_embedding_model: Optional[SentenceTransformerModel] = None
_vector_store: Optional[VectorStore] = None

async def init_resources():
    """서비스 시작 시 리소스 초기화"""
    global _embedding_model, _vector_store
    
    # 임베딩 모델 초기화
    _embedding_model = SentenceTransformerModel()
    await _embedding_model.initialize()
    
    # 벡터 스토어 초기화 (전략 B용)
    _vector_store = VectorStore()
    await _vector_store.initialize()

async def cleanup_resources():
    """서비스 종료 시 리소스 정리"""
    global _embedding_model, _vector_store
    
    if _embedding_model:
        await _embedding_model.cleanup()
        _embedding_model = None
    
    if _vector_store:
        await _vector_store.cleanup()
        _vector_store = None

def get_embedding_model() -> SentenceTransformerModel:
    """임베딩 모델 인스턴스 반환"""
    if _embedding_model is None:
        raise RuntimeError("임베딩 모델이 초기화되지 않았습니다")
    return _embedding_model

def get_vector_store() -> VectorStore:
    """벡터 스토어 인스턴스 반환"""
    if _vector_store is None:
        raise RuntimeError("벡터 스토어가 초기화되지 않았습니다")
    return _vector_store
