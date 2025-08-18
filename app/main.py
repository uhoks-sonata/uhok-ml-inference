"""
ML 추론 서비스 메인 애플리케이션
"""

import uvicorn
from fastapi import FastAPI
from app.api import router
from app.deps import init_resources, cleanup_resources

app = FastAPI(
    title="UHOK ML Inference Service",
    description="레시피 추천을 위한 ML 추론 서비스",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """서비스 시작 시 리소스 초기화"""
    await init_resources()

@app.on_event("shutdown")
async def shutdown_event():
    """서비스 종료 시 리소스 정리"""
    await cleanup_resources()

# API 라우터 등록
app.include_router(router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        workers=1
    )
