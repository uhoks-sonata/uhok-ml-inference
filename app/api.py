"""
ML Inference API 엔드포인트
임베딩 생성 및 배치 처리를 제공합니다.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional
import time
import logging
from .deps import encode_text, get_model_info

logger = logging.getLogger(__name__)

router = APIRouter()

class EmbedRequest(BaseModel):
    """단일 텍스트 임베딩 요청"""
    text: str = Field(..., description="임베딩할 텍스트", min_length=1, max_length=1000)
    normalize: bool = Field(True, description="임베딩 정규화 여부")

class EmbedBatchRequest(BaseModel):
    """배치 텍스트 임베딩 요청"""
    texts: List[str] = Field(..., description="임베딩할 텍스트 리스트", min_items=1, max_items=100)
    normalize: bool = Field(True, description="임베딩 정규화 여부")

class EmbedResponse(BaseModel):
    """임베딩 응답"""
    embedding: List[float] = Field(..., description="임베딩 벡터")
    dim: int = Field(..., description="벡터 차원")
    version: str = Field(..., description="모델 버전")

class EmbedBatchResponse(BaseModel):
    """배치 임베딩 응답"""
    embeddings: List[List[float]] = Field(..., description="임베딩 벡터 리스트")
    dim: int = Field(..., description="벡터 차원")
    version: str = Field(..., description="모델 버전")
    count: int = Field(..., description="처리된 텍스트 수")

@router.post("/embed", response_model=EmbedResponse)
async def create_embedding(request: EmbedRequest):
    """
    단일 텍스트에 대한 임베딩을 생성합니다.
    
    Args:
        request: 임베딩 요청 (텍스트, 정규화 여부)
        
    Returns:
        임베딩 벡터와 메타데이터
    """
    start_time = time.time()
    
    try:
        logger.info(f"임베딩 생성 요청: text='{request.text[:50]}...', normalize={request.normalize}")
        
        # 임베딩 생성
        embedding = await encode_text(request.text, request.normalize)
        
        # 모델 정보 가져오기
        model_info = await get_model_info()
        
        # 응답 생성
        response = EmbedResponse(
            embedding=embedding,
            dim=model_info["dimension"],
            version=model_info["version"]
        )
        
        execution_time = time.time() - start_time
        logger.info(f"임베딩 생성 완료: 실행시간={execution_time:.3f}초, 차원={len(embedding)}")
        
        return response
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"임베딩 생성 실패: 실행시간={execution_time:.3f}초, error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

@router.post("/embed-batch", response_model=EmbedBatchResponse)
async def create_embeddings_batch(request: EmbedBatchRequest):
    """
    여러 텍스트에 대한 배치 임베딩을 생성합니다.
    
    Args:
        request: 배치 임베딩 요청 (텍스트 리스트, 정규화 여부)
        
    Returns:
        임베딩 벡터 리스트와 메타데이터
    """
    start_time = time.time()
    
    try:
        logger.info(f"배치 임베딩 생성 요청: count={len(request.texts)}, normalize={request.normalize}")
        
        # 배치 임베딩 생성
        embeddings = []
        for text in request.texts:
            embedding = await encode_text(text, request.normalize)
            embeddings.append(embedding)
        
        # 모델 정보 가져오기
        model_info = await get_model_info()
        
        # 응답 생성
        response = EmbedBatchResponse(
            embeddings=embeddings,
            dim=model_info["dimension"],
            version=model_info["version"],
            count=len(embeddings)
        )
        
        execution_time = time.time() - start_time
        logger.info(f"배치 임베딩 생성 완료: 실행시간={execution_time:.3f}초, 처리수={len(embeddings)}")
        
        return response
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"배치 임베딩 생성 실패: 실행시간={execution_time:.3f}초, error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch embedding generation failed: {str(e)}")

@router.get("/model-info")
async def get_model_information():
    """
    현재 로드된 모델의 정보를 반환합니다.
    
    Returns:
        모델 정보 (이름, 차원, 버전 등)
    """
    try:
        model_info = await get_model_info()
        return model_info
    except Exception as e:
        logger.error(f"모델 정보 조회 실패: error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")
