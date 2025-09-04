"""
uhok-ml-inference 서비스 메인 애플리케이션
FastAPI 기반으로 임베딩 생성 API를 제공합니다.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from .api import router
from .deps import get_model_info

app = FastAPI(
    title="UHOK ML Inference Service",
    description="레시피 추천을 위한 임베딩 생성 서비스",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 운영에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
app.include_router(router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    try:
        model_info = await get_model_info()
        return {
            "status": "ok",
            "model": model_info["model_name"],
            "dim": model_info["dimension"],
            "version": model_info["version"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Service unhealthy: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )
