"""
ML 모델 의존성 관리
모델 로딩, 캐싱, 정보 제공을 담당합니다.
"""

import asyncio
from typing import Optional, Dict, Any
import time
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

# 전역 모델 캐시
_model: Optional[SentenceTransformer] = None
_model_lock = asyncio.Lock()
_model_info: Optional[Dict[str, Any]] = None

async def get_model() -> SentenceTransformer:
    """
    SentenceTransformer 임베딩 모델을 전역 캐시하여 반환합니다.
    - 최초 1회만 로드하며 동시 호출은 Lock으로 보호합니다.
    - 모델: paraphrase-multilingual-MiniLM-L12-v2 (384차원)
    """
    global _model, _model_info
    
    if _model is not None:
        logger.debug("캐시된 SentenceTransformer 모델 사용 중")
        return _model
    
    # 모델 로딩 시간 체크 시작
    start_time = time.time()
    
    async with _model_lock:
        if _model is None:
            logger.info("SentenceTransformer 모델 로드 중: paraphrase-multilingual-MiniLM-L12-v2")
            try:
                _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
                
                # 모델 정보 저장
                _model_info = {
                    "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
                    "dimension": 384,
                    "version": "sentence-transformers-5.0.0",
                    "device": "cpu"
                }
                
                # 모델 로딩 시간 체크 완료 및 로깅
                loading_time = time.time() - start_time
                logger.info(f"SentenceTransformer 모델 로드 완료: 로딩시간={loading_time:.3f}초")
            except Exception as e:
                loading_time = time.time() - start_time
                logger.error(f"SentenceTransformer 모델 로드 실패: 로딩시간={loading_time:.3f}초, error={str(e)}")
                raise
        else:
            logger.debug("다른 코루틴에서 모델 로드 완료")
    
    return _model

async def get_model_info() -> Dict[str, Any]:
    """모델 정보를 반환합니다."""
    global _model_info
    
    if _model_info is None:
        # 모델이 로드되지 않았다면 로드
        await get_model()
    
    return _model_info

async def encode_text(text: str, normalize: bool = True) -> list:
    """
    텍스트를 임베딩으로 변환합니다.
    
    Args:
        text: 임베딩할 텍스트
        normalize: 정규화 여부
        
    Returns:
        임베딩 벡터 (list of float)
    """
    model = await get_model()
    embedding = model.encode(text, normalize_embeddings=normalize)
    return embedding.tolist()
