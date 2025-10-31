"""임베딩 생성 API 라우터"""
import time
import logging
from fastapi import APIRouter, HTTPException

from ..models.embedding import EmbedRequest, EmbedResponse, EmbedBatchRequest, EmbedBatchResponse
from ..deps import encode_text, get_model_info

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/embed", response_model=EmbedResponse)
async def create_embedding(request: EmbedRequest):
    """
    단일 텍스트에 대한 임베딩을 생성합니다.
    """
    start_time = time.time()
    try:
        logger.info(f"임베딩 생성 요청: text='{request.text[:50]}...', normalize={request.normalize}")
        embedding = await encode_text(request.text, request.normalize)
        model_info = await get_model_info()
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
    """
    start_time = time.time()
    try:
        logger.info(f"배치 임베딩 생성 요청: count={len(request.texts)}, normalize={request.normalize}")
        embeddings = [await encode_text(text, request.normalize) for text in request.texts]
        model_info = await get_model_info()
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
