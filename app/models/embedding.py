"""임베딩 API 관련 Pydantic 모델"""
from pydantic import BaseModel, Field
from typing import List

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
