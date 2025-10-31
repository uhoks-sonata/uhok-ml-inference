"""모델 관리 API 라우터"""
import logging
from fastapi import APIRouter, HTTPException

from ..deps import get_model_info

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/model-info")
async def get_model_information():
    """
    현재 로드된 모델의 정보를 반환합니다.
    """
    try:
        model_info = await get_model_info()
        return model_info
    except Exception as e:
        logger.error(f"모델 정보 조회 실패: error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")
