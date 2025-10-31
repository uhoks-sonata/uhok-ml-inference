"""유사도 검색 API 관련 Pydantic 모델"""
from pydantic import BaseModel, Field
from typing import List, Optional

class SearchRequest(BaseModel):
    """유사도 검색 요청"""
    query: str = Field(..., description="검색할 쿼리 텍스트")
    top_k: int = Field(25, description="반환할 상위 결과 수")
    exclude_ids: Optional[List[int]] = Field(None, description="검색에서 제외할 ID 리스트")

class SearchResultItem(BaseModel):
    """검색 결과 항목"""
    recipe_id: int
    distance: float

class SearchResponse(BaseModel):
    """유사도 검색 응답"""
    results: List[SearchResultItem] = Field(..., description="검색 결과 리스트")
