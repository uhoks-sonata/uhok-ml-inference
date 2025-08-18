"""
벡터 스토어 관리 (전략 B용) - 백엔드 core.py 참고
"""

import asyncio
import time
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from common.logger import get_logger

logger = get_logger("vector_store")

class VectorStore:
    """백엔드 core.py와 동일한 방식의 벡터 스토어"""
    
    def __init__(self):
        self.vectors: Optional[np.ndarray] = None
        self.recipe_ids: List[int] = []
        self.dimension: int = 384
        self.index_path: str = "store/recipe_vectors.npy"
        self.metadata_path: str = "store/recipe_metadata.pkl"
        
        # 성능 측정용 통계
        self.performance_stats = {
            "index_build_time": 0.0,
            "search_times": [],
            "add_vector_times": [],
            "total_searches": 0,
            "total_adds": 0
        }
        
    async def initialize(self):
        """벡터 스토어 초기화"""
        start_time = time.time()
        
        try:
            # 기존 인덱스 로드 시도
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                await self._load_existing_index()
                logger.info(f"기존 인덱스 로드 완료: {len(self.recipe_ids)}개 레시피")
            else:
                # 새 인덱스 생성
                await self._create_new_index()
                logger.info("새 인덱스 생성 완료")
                
        except Exception as e:
            logger.error(f"벡터 스토어 초기화 실패: {e}")
            raise
        finally:
            init_time = time.time() - start_time
            logger.info(f"벡터 스토어 초기화 시간: {init_time:.3f}초")
    
    async def _load_existing_index(self):
        """기존 인덱스 로드"""
        # 벡터 데이터 로드
        self.vectors = np.load(self.index_path)
        
        # 메타데이터 로드
        with open(self.metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            self.recipe_ids = metadata.get('recipe_ids', [])
            self.dimension = metadata.get('dimension', 384)
        
        logger.info(f"기존 인덱스 로드 완료: {len(self.recipe_ids)}개 레시피")
    
    async def _create_new_index(self):
        """새 인덱스 생성"""
        # 메타데이터 초기화
        self.recipe_ids = []
        self.vectors = np.array([]).reshape(0, self.dimension)
        
        # 디렉토리 생성
        os.makedirs("store", exist_ok=True)
        logger.info("새 인덱스 생성 완료")
    
    async def search_similar(
        self, 
        query: str, 
        top_k: int = 25, 
        exclude_ids: Optional[List[int]] = None
    ) -> List[dict]:
        """
        쿼리와 유사한 레시피 검색 (백엔드 core.py와 동일한 방식)
        """
        if self.vectors is None:
            raise RuntimeError("벡터 스토어가 초기화되지 않았습니다")
        
        search_start_time = time.time()
        
        try:
            # 쿼리 임베딩 생성 (임시로 랜덤 벡터 사용)
            # 실제로는 SentenceTransformerModel을 사용해야 함
            query_vector = np.random.randn(self.dimension).astype('float32')
            query_vector = query_vector / np.linalg.norm(query_vector)  # 정규화
            
            # 백엔드와 동일한 방식으로 유사도 계산
            similarities = []
            for i, recipe_id in enumerate(self.recipe_ids):
                if exclude_ids and recipe_id in exclude_ids:
                    continue
                
                # 코사인 유사도 계산
                similarity = float(cosine_similarity([query_vector], [self.vectors[i]])[0][0])
                similarities.append((recipe_id, similarity))
            
            # 유사도 순으로 정렬
            similarities.sort(key=lambda x: x[1], ascending=False)
            
            # top_k 결과 반환
            results = []
            for recipe_id, similarity in similarities[:top_k]:
                results.append({
                    "recipe_id": int(recipe_id),
                    "distance": 1.0 - similarity  # 유사도를 거리로 변환
                })
            
            # 성능 통계 업데이트
            search_time = time.time() - search_start_time
            self.performance_stats["search_times"].append(search_time)
            self.performance_stats["total_searches"] += 1
            
            logger.info(f"벡터 검색 완료: {len(results)}개 결과, 검색시간={search_time:.3f}초")
            return results
            
        except Exception as e:
            search_time = time.time() - search_start_time
            logger.error(f"검색 실패: 검색시간={search_time:.3f}초, error={str(e)}")
            raise
    
    async def add_vectors(self, recipe_ids: List[int], vectors: np.ndarray):
        """새 벡터 추가"""
        add_start_time = time.time()
        
        try:
            # 벡터 추가
            if len(self.vectors) == 0:
                self.vectors = vectors
            else:
                self.vectors = np.vstack([self.vectors, vectors])
            
            # 메타데이터 업데이트
            self.recipe_ids.extend(recipe_ids)
            
            # 저장
            await self._save_index()
            
            # 성능 통계 업데이트
            add_time = time.time() - add_start_time
            self.performance_stats["add_vector_times"].append(add_time)
            self.performance_stats["total_adds"] += 1
            
            logger.info(f"벡터 추가 완료: {len(recipe_ids)}개 레시피, 추가시간={add_time:.3f}초")
            
        except Exception as e:
            add_time = time.time() - add_start_time
            logger.error(f"벡터 추가 실패: 추가시간={add_time:.3f}초, error={str(e)}")
            raise
    
    async def _save_index(self):
        """인덱스 및 메타데이터 저장"""
        if self.vectors is None:
            return
        
        # 벡터 데이터 저장
        np.save(self.index_path, self.vectors)
        
        # 메타데이터 저장
        metadata = {
            'recipe_ids': self.recipe_ids,
            'dimension': self.dimension
        }
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
    
    async def cleanup(self):
        """리소스 정리"""
        if self.vectors is not None:
            del self.vectors
            self.vectors = None
        
        self.recipe_ids.clear()
        logger.info("벡터 스토어 리소스 정리 완료")
    
    def get_performance_stats(self) -> dict:
        """성능 통계 반환"""
        stats = self.performance_stats.copy()
        
        # 평균 시간 계산
        if stats["search_times"]:
            stats["avg_search_time"] = np.mean(stats["search_times"])
            stats["min_search_time"] = np.min(stats["search_times"])
            stats["max_search_time"] = np.max(stats["search_times"])
            stats["p95_search_time"] = np.percentile(stats["search_times"], 95)
            stats["p99_search_time"] = np.percentile(stats["search_times"], 99)
        
        if stats["add_vector_times"]:
            stats["avg_add_time"] = np.mean(stats["add_vector_times"])
            stats["min_add_time"] = np.min(stats["add_vector_times"])
            stats["max_add_time"] = np.max(stats["add_vector_times"])
        
        return stats
    
    def reset_performance_stats(self):
        """성능 통계 초기화"""
        self.performance_stats = {
            "index_build_time": 0.0,
            "search_times": [],
            "add_vector_times": [],
            "total_searches": 0,
            "total_adds": 0
        }
        logger.info("성능 통계 초기화 완료")
