"""유사도 검색 관련 CRUD 작업"""
import logging
from typing import List, Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, bindparam
from pgvector.sqlalchemy import Vector

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384
VECTOR_COL = '"VECTOR_NAME"'

async def search_similar_in_db(
    db: AsyncSession,
    query_vector: List[float],
    top_k: int,
    exclude_ids: Optional[List[int]] = None,
) -> List[Tuple[int, float]]:
    """DB에서 벡터 유사도 검색을 수행합니다."""
    if exclude_ids:
        sql = text(f'''
            SELECT "RECIPE_ID" AS recipe_id,
                   {VECTOR_COL} <-> :qv AS distance
            FROM "RECIPE_VECTOR_TABLE"
            WHERE "RECIPE_ID" NOT IN :ex_ids
            ORDER BY distance ASC
            LIMIT :k
        ''')
        params = {
            "qv": query_vector,
            "ex_ids": tuple(exclude_ids),
            "k": top_k,
        }
        sql = sql.bindparams(
            bindparam("qv", type_=Vector(EMBEDDING_DIM)),
            bindparam("ex_ids", expanding=True),
            bindparam("k")
        )
    else:
        sql = text(f'''
            SELECT "RECIPE_ID" AS recipe_id,
                   {VECTOR_COL} <-> :qv AS distance
            FROM "RECIPE_VECTOR_TABLE"
            ORDER BY distance ASC
            LIMIT :k
        ''')
        params = {"qv": query_vector, "k": top_k}
        sql = sql.bindparams(
            bindparam("qv", type_=Vector(EMBEDDING_DIM)),
            bindparam("k")
        )

    result_rows = (await db.execute(sql, params)).all()
    return [(row.recipe_id, row.distance) for row in result_rows]
