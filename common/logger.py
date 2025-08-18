"""
로거 유틸리티 (백엔드와 동일한 인터페이스)
"""

import logging
import sys
from typing import Optional

def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    로거 인스턴스를 반환한다.
    
    Args:
        name: 로거 이름
        level: 로그 레벨 (기본값: INFO)
    
    Returns:
        logging.Logger: 설정된 로거 인스턴스
    """
    if level is None:
        level = logging.INFO
    
    # 로거 생성
    logger = logging.getLogger(name)
    
    # 이미 핸들러가 설정되어 있으면 그대로 반환
    if logger.handlers:
        return logger
    
    # 로그 레벨 설정
    logger.setLevel(level)
    
    # 콘솔 핸들러 생성
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # 포맷터 생성
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(console_handler)
    
    return logger
