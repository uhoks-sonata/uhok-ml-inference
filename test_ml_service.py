#!/usr/bin/env python3
"""
ML 서비스 테스트 스크립트
로컬에서 ML 서비스의 동작을 확인합니다.
"""

import asyncio
import httpx
import json
import time
from typing import Dict, Any

# 테스트 설정
ML_SERVICE_URL = "http://localhost:8001"
TEST_TEXTS = [
    "갈비탕",
    "김치찌개", 
    "된장찌개",
    "불고기",
    "비빔밥"
]

async def test_health_check() -> bool:
    """헬스체크 테스트"""
    print("🔍 헬스체크 테스트...")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{ML_SERVICE_URL}/health")
            response.raise_for_status()
            
            data = response.json()
            print(f"✅ 헬스체크 성공: {data}")
            return True
    except Exception as e:
        print(f"❌ 헬스체크 실패: {e}")
        return False

async def test_single_embedding(text: str) -> Dict[str, Any]:
    """단일 텍스트 임베딩 테스트"""
    print(f"🔍 단일 임베딩 테스트: '{text}'")
    try:
        start_time = time.time()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{ML_SERVICE_URL}/api/v1/embed",
                json={"text": text, "normalize": True}
            )
            response.raise_for_status()
            
            data = response.json()
            execution_time = time.time() - start_time
            
            print(f"✅ 임베딩 성공: 차원={data['dim']}, 시간={execution_time:.3f}초")
            return {
                "success": True,
                "dimension": data["dim"],
                "execution_time": execution_time,
                "embedding_preview": data["embedding"][:5]  # 처음 5개 값만
            }
    except Exception as e:
        print(f"❌ 임베딩 실패: {e}")
        return {"success": False, "error": str(e)}

async def test_batch_embedding(texts: list) -> Dict[str, Any]:
    """배치 임베딩 테스트"""
    print(f"🔍 배치 임베딩 테스트: {len(texts)}개 텍스트")
    try:
        start_time = time.time()
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{ML_SERVICE_URL}/api/v1/embed-batch",
                json={"texts": texts, "normalize": True}
            )
            response.raise_for_status()
            
            data = response.json()
            execution_time = time.time() - start_time
            
            print(f"✅ 배치 임베딩 성공: 처리수={data['count']}, 시간={execution_time:.3f}초")
            return {
                "success": True,
                "count": data["count"],
                "dimension": data["dim"],
                "execution_time": execution_time
            }
    except Exception as e:
        print(f"❌ 배치 임베딩 실패: {e}")
        return {"success": False, "error": str(e)}

async def test_model_info() -> Dict[str, Any]:
    """모델 정보 조회 테스트"""
    print("🔍 모델 정보 조회 테스트...")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{ML_SERVICE_URL}/api/v1/model-info")
            response.raise_for_status()
            
            data = response.json()
            print(f"✅ 모델 정보 조회 성공: {data}")
            return {"success": True, "info": data}
    except Exception as e:
        print(f"❌ 모델 정보 조회 실패: {e}")
        return {"success": False, "error": str(e)}

async def main():
    """메인 테스트 함수"""
    print("🚀 ML 서비스 테스트 시작")
    print(f"📍 서비스 URL: {ML_SERVICE_URL}")
    print("=" * 50)
    
    # 1. 헬스체크
    health_ok = await test_health_check()
    if not health_ok:
        print("❌ 헬스체크 실패로 테스트 중단")
        return
    
    print()
    
    # 2. 모델 정보 조회
    await test_model_info()
    print()
    
    # 3. 단일 임베딩 테스트
    single_results = []
    for text in TEST_TEXTS[:3]:  # 처음 3개만 테스트
        result = await test_single_embedding(text)
        single_results.append(result)
        print()
    
    # 4. 배치 임베딩 테스트
    batch_result = await test_batch_embedding(TEST_TEXTS)
    print()
    
    # 5. 결과 요약
    print("📊 테스트 결과 요약")
    print("=" * 50)
    
    successful_single = sum(1 for r in single_results if r.get("success", False))
    print(f"단일 임베딩: {successful_single}/{len(single_results)} 성공")
    
    if batch_result.get("success", False):
        print(f"배치 임베딩: 성공 ({batch_result['count']}개 처리)")
    else:
        print(f"배치 임베딩: 실패")
    
    # 성능 통계
    single_times = [r.get("execution_time", 0) for r in single_results if r.get("success", False)]
    if single_times:
        avg_time = sum(single_times) / len(single_times)
        print(f"평균 임베딩 시간: {avg_time:.3f}초")
    
    if batch_result.get("success", False):
        batch_time = batch_result.get("execution_time", 0)
        batch_count = batch_result.get("count", 1)
        print(f"배치 평균 시간: {batch_time/batch_count:.3f}초/개")
    
    print("\n✅ 테스트 완료!")

if __name__ == "__main__":
    asyncio.run(main())
