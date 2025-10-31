"""유사도 검색 API 라우터"""
import time
import logging
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.search import SearchRequest, SearchResponse, SearchResultItem
from ..crud.search_crud import search_similar_in_db
from ..deps import get_db_session, encode_text

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/search", response_model=SearchResponse)
async def search_similar_recipes(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db_session)
):
    """
    쿼리 텍스트와 유사한 레시피를 검색합니다.
    """
    start_time = time.time()
    try:
        # 1. 쿼리 임베딩 생성
        embedding_start_time = time.time()
        query_vector = await encode_text(request.query, normalize=True)
        embedding_time = time.time() - embedding_start_time
        logger.info(f"쿼리 임베딩 생성 완료: {embedding_time:.3f}초")

        # 2. DB에서 유사도 검색
        db_start_time = time.time()
        search_results_raw = await search_similar_in_db(
            db, query_vector, request.top_k, request.exclude_ids
        )
        db_time = time.time() - db_start_time
        logger.info(f"DB 유사도 검색 완료: {db_time:.3f}초, 결과 {len(search_results_raw)}건")

        # 3. 결과 포맷팅
        search_results = [SearchResultItem(recipe_id=r_id, distance=dist) for r_id, dist in search_results_raw]
        
        total_time = time.time() - start_time
        logger.info(f"유사도 검색 성공: 총 {total_time:.3f}초 소요")
        
        return SearchResponse(results=search_results)

    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"유사도 검색 실패: 총 {total_time:.3f}초, error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {str(e)}")
