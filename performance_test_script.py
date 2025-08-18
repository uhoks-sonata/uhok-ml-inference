#!/usr/bin/env python3
"""
scikit-learn vs FAISS 성능 비교 테스트 스크립트
"""

import asyncio
import time
import json
from typing import List, Dict
import numpy as np

# 벡터 스토어 임포트 (FAISS 제거됨)
from store.vector_store import VectorStore
# from store.vector_store_faiss import VectorStoreFAISS  # Python 3.13.5 호환성 문제로 제거

async def generate_test_data(num_vectors: int = 1000, dimension: int = 384):
    """테스트용 벡터 데이터 생성"""
    print(f"테스트 데이터 생성 중: {num_vectors}개 벡터, {dimension}차원")
    
    # 랜덤 벡터 생성
    vectors = np.random.randn(num_vectors, dimension).astype('float32')
    
    # 정규화
    for i in range(num_vectors):
        vectors[i] = vectors[i] / np.linalg.norm(vectors[i])
    
    # 레시피 ID 생성
    recipe_ids = list(range(1, num_vectors + 1))
    
    return recipe_ids, vectors

async def test_scikit_learn(recipe_ids: List[int], vectors: np.ndarray, queries: List[str], top_k: int = 25):
    """scikit-learn 벡터 스토어 성능 테스트"""
    print("\n=== scikit-learn 벡터 스토어 테스트 ===")
    
    store = VectorStore()
    await store.initialize()
    
    # 벡터 추가
    start_time = time.time()
    await store.add_vectors(recipe_ids, vectors)
    add_time = time.time() - start_time
    print(f"벡터 추가 시간: {add_time:.3f}초")
    
    # 검색 성능 테스트
    search_times = []
    for i, query in enumerate(queries):
        start_time = time.time()
        results = await store.search_similar(query, top_k=top_k)
        search_time = time.time() - start_time
        search_times.append(search_time)
        
        if i < 3:  # 처음 3개만 상세 출력
            print(f"  쿼리 '{query}': {search_time:.3f}초, 결과 {len(results)}개")
    
    # 통계 계산
    avg_search_time = np.mean(search_times)
    min_search_time = np.min(search_times)
    max_search_time = np.max(search_times)
    p95_search_time = np.percentile(search_times, 95)
    
    print(f"검색 성능 통계:")
    print(f"  평균: {avg_search_time:.3f}초")
    print(f"  최소: {min_search_time:.3f}초")
    print(f"  최대: {max_search_time:.3f}초")
    print(f"  P95: {p95_search_time:.3f}초")
    
    await store.cleanup()
    
    return {
        "add_time": add_time,
        "search_times": search_times,
        "avg_search_time": avg_search_time,
        "min_search_time": min_search_time,
        "max_search_time": max_search_time,
        "p95_search_time": p95_search_time
    }

async def test_faiss(recipe_ids: List[int], vectors: np.ndarray, queries: List[str], top_k: int = 25):
    """FAISS 벡터 스토어 성능 테스트"""
    print("\n=== FAISS 벡터 스토어 테스트 ===")
    
    store = VectorStoreFAISS()
    await store.initialize()
    
    # 벡터 추가
    start_time = time.time()
    await store.add_vectors(recipe_ids, vectors)
    add_time = time.time() - start_time
    print(f"벡터 추가 시간: {add_time:.3f}초")
    
    # 검색 성능 테스트
    search_times = []
    for i, query in enumerate(queries):
        start_time = time.time()
        results = await store.search_similar(query, top_k=top_k)
        search_time = time.time() - start_time
        search_times.append(search_time)
        
        if i < 3:  # 처음 3개만 상세 출력
            print(f"  쿼리 '{query}': {search_time:.3f}초, 결과 {len(results)}개")
    
    # 통계 계산
    avg_search_time = np.mean(search_times)
    min_search_time = np.min(search_times)
    max_search_time = np.max(search_times)
    p95_search_time = np.percentile(search_times, 95)
    
    print(f"검색 성능 통계:")
    print(f"  평균: {avg_search_time:.3f}초")
    print(f"  최소: {min_search_time:.3f}초")
    print(f"  최대: {max_search_time:.3f}초")
    print(f"  P95: {p95_search_time:.3f}초")
    
    await store.cleanup()
    
    return {
        "add_time": add_time,
        "search_times": search_times,
        "avg_search_time": avg_search_time,
        "min_search_time": min_search_time,
        "max_search_time": max_search_time,
        "p95_search_time": p95_search_time
    }

async def main():
    """메인 테스트 함수"""
    print("🚀 scikit-learn vs FAISS 성능 비교 테스트 시작")
    
    # 테스트 파라미터
    num_vectors = 1000
    dimension = 384
    num_queries = 20
    top_k = 25
    
    # 테스트 데이터 생성
    recipe_ids, vectors = await generate_test_data(num_vectors, dimension)
    
    # 테스트 쿼리 생성
    test_queries = [
        "김치찌개", "된장찌개", "순두부찌개", "부대찌개",
        "제육볶음", "닭볶음탕", "돼지고기볶음", "소고기볶음",
        "김밥", "라면", "비빔밥", "덮밥",
        "스테이크", "파스타", "피자", "햄버거",
        "샐러드", "스프", "스튜", "카레"
    ][:num_queries]
    
    print(f"테스트 쿼리: {len(test_queries)}개")
    
    # scikit-learn 테스트
    scikit_results = await test_scikit_learn(recipe_ids, vectors, test_queries, top_k)
    
    # FAISS 테스트 (제거됨)
    # faiss_results = await test_faiss(recipe_ids, vectors, test_queries, top_k)
    
    # 성능 분석 (scikit-learn만)
    print("\n" + "="*50)
    print("📊 scikit-learn 성능 결과")
    print("="*50)
    
    print(f"벡터 추가 시간: {scikit_results['add_time']:.3f}초")
    print(f"검색 시간 (평균): {scikit_results['avg_search_time']:.3f}초")
    print(f"검색 시간 (P95): {scikit_results['p95_search_time']:.3f}초")
    
    print(f"\n✅ scikit-learn 성능 측정 완료!")
    print("   FAISS는 Python 3.13.5 호환성 문제로 제거됨")
    
    # 결과 저장
    results = {
        "test_config": {
            "num_vectors": num_vectors,
            "dimension": dimension,
            "num_queries": num_queries,
            "top_k": top_k
        },
        "scikit_learn": scikit_results,
        "faiss": {
            "status": "removed",
            "message": "Python 3.13.5 호환성 문제로 제거됨"
        },
        "comparison": {
            "recommendation": "scikit-learn",
            "note": "FAISS 제거로 scikit-learn만 사용"
        },
        "timestamp": time.time()
    }
    
    # JSON 파일로 저장
    with open("performance_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 결과가 'performance_test_results.json' 파일에 저장되었습니다.")

if __name__ == "__main__":
    asyncio.run(main())
