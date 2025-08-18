"""
Sentence Transformers 모델 관리 (백엔드 core.py 참고)
"""

import asyncio
from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from common.logger import get_logger

logger = get_logger("sentence_transformers_model")

_model: Optional[SentenceTransformer] = None
_model_lock = asyncio.Lock()

async def get_model() -> SentenceTransformer:
    """
    SentenceTransformer 임베딩 모델을 전역 캐시하여 반환한다.
    - 최초 1회만 로드하며 동시 호출은 Lock으로 보호한다.
    - 모델: paraphrase-multilingual-MiniLM-L12-v2 (384차원)
    """
    global _model
    if _model is not None:
        logger.debug("캐시된 SentenceTransformer 모델 사용 중")
        return _model
    
    # 모델 로딩 시간 체크 시작
    start_time = asyncio.get_event_loop().time()
    
    async with _model_lock:
        if _model is None:
            logger.info("SentenceTransformer 모델 로드 중: paraphrase-multilingual-MiniLM-L12-v2")
            try:
                # CPU/GPU 자동 선택
                device = "cuda" if torch.cuda.is_available() else "cpu"
                _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)
                
                # 웜업 실행
                warmup_texts = ["테스트", "레시피", "요리"]
                _ = _model.encode(warmup_texts, normalize_embeddings=True)
                
                # 모델 로딩 시간 체크 완료 및 로깅
                loading_time = asyncio.get_event_loop().time() - start_time
                logger.info(f"SentenceTransformer 모델 로드 완료: 로딩시간={loading_time:.3f}초, device={device}")
            except Exception as e:
                loading_time = asyncio.get_event_loop().time() - start_time
                logger.error(f"SentenceTransformer 모델 로드 실패: 로딩시간={loading_time:.3f}초, error={str(e)}")
                raise
        else:
            logger.debug("다른 코루틴에서 모델 로드 완료")
    return _model

class SentenceTransformerModel:
    """Sentence Transformers 모델 래퍼 클래스 (백엔드와 동일한 인터페이스)"""
    
    def __init__(self):
        self.model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
        self.dimension: int = 384
        self.version: str = "sbert-multilingual-v1"
        
    async def initialize(self):
        """모델 초기화 및 로딩"""
        try:
            # 백엔드와 동일한 방식으로 모델 로드
            await get_model()
            logger.info(f"모델 초기화 완료: {self.model_name}")
        except Exception as e:
            logger.error(f"모델 초기화 실패: {e}")
            raise
    
    async def embed_text(self, text: str, normalize: bool = True) -> List[float]:
        """단일 텍스트 임베딩 생성"""
        model = await get_model()
        
        # 텍스트 전처리
        processed_text = self._preprocess_text(text)
        
        # 임베딩 생성
        embedding = model.encode(
            processed_text, 
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )
        
        return embedding.tolist()
    
    async def embed_texts_batch(self, texts: List[str], normalize: bool = True) -> List[List[float]]:
        """배치 텍스트 임베딩 생성"""
        model = await get_model()
        
        # 텍스트 전처리
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # 배치 임베딩 생성
        embeddings = model.encode(
            processed_texts,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            batch_size=32  # 배치 크기 조정 가능
        )
        
        return embeddings.tolist()
    
    def _preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        # 기본 정규화
        text = text.strip()
        if not text:
            text = "빈 텍스트"
        return text
    
    def get_model_name(self) -> str:
        """모델 이름 반환"""
        return self.model_name
    
    def get_dimension(self) -> int:
        """임베딩 차원 반환"""
        return self.dimension
    
    def get_version(self) -> str:
        """모델 버전 반환"""
        return self.version
    
    async def cleanup(self):
        """리소스 정리"""
        global _model
        if _model is not None:
            del _model
            _model = None
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
