# -*- coding: utf-8 -*-
"""
EMA (Exponential Moving Average) 필터 - 순환 각도용

지수 이동평균 기반 각도 필터링
단순하고 효율적인 스무딩 필터
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
from src.pytemplate.core import ang


@dataclass
class EMAConfig:
    """EMA 필터 설정"""
    alpha: float = 0.2              # 스무딩 팩터 (0 < alpha <= 1)
    window_size: int = None         # 윈도우 크기 (alpha 계산용, 옵션)
    adaptive: bool = False          # 적응형 알파 사용 여부


class EMAFilter:
    """지수 이동평균 필터
    
    공식: EMA[n] = α × X[n] + (1-α) × EMA[n-1]
    
    특징:
    - 최근 값에 더 높은 가중치 부여
    - 메모리 효율적 (단일 이전 값만 저장)
    - 순환 각도 처리 최적화
    - 빠른 응답성과 안정성 절충
    """
    
    def __init__(self, config: EMAConfig):
        """EMA 필터 초기화
        
        Args:
            config: EMA 필터 설정
        """
        self.config = config
        self.alpha = config.alpha
        self.ema_value = 0.0
        self.is_initialized = False
        
        # 적응형 알파를 위한 변수들
        self.velocity_history = []
        self.velocity_threshold = 0.1  # rad/s
        self.alpha_min = 0.05
        self.alpha_max = 0.5
        
        # 윈도우 크기로부터 알파 계산
        if config.window_size is not None:
            self.alpha = 2.0 / (config.window_size + 1)
    
    def update(self, measured_angle: float) -> float:
        """EMA 필터 업데이트
        
        Args:
            measured_angle: 측정된 각도 (rad)
            
        Returns:
            ema_angle: EMA 필터링된 각도 (rad)
        """
        if not self.is_initialized:
            # 첫 번째 값으로 초기화
            self.ema_value = measured_angle
            self.is_initialized = True
            return self.ema_value
        
        # 적응형 알파 계산 (옵션)
        current_alpha = self.alpha
        if self.config.adaptive:
            current_alpha = self._calculate_adaptive_alpha(measured_angle)
        
        # 순환 각도 차이 계산
        angle_diff = ang.diffpi(measured_angle, self.ema_value)
        
        # EMA 필터 적용
        self.ema_value += current_alpha * angle_diff
        
        # -π ~ π 범위로 정규화
        self.ema_value = ang.wrap_pi(self.ema_value)
        
        return self.ema_value
    
    def _calculate_adaptive_alpha(self, measured_angle: float) -> float:
        """적응형 알파 계산
        
        각속도가 빠를 때는 높은 알파 (빠른 추적)
        각속도가 느릴 때는 낮은 알파 (안정적 스무딩)
        
        Args:
            measured_angle: 현재 측정 각도
            
        Returns:
            adaptive_alpha: 적응형 알파 값
        """
        if not self.is_initialized:
            return self.alpha
        
        # 각속도 추정
        angle_diff = ang.diffpi(measured_angle, self.ema_value)
        estimated_velocity = abs(angle_diff / 0.01)  # 대략적 dt=0.01 가정
        
        # 속도 이력 관리 (최근 5개 값)
        self.velocity_history.append(estimated_velocity)
        if len(self.velocity_history) > 5:
            self.velocity_history.pop(0)
        
        # 평균 속도 계산
        avg_velocity = np.mean(self.velocity_history)
        
        # 속도에 따른 알파 조정
        if avg_velocity > self.velocity_threshold:
            # 빠른 움직임: 높은 알파 (빠른 추적)
            velocity_ratio = min(avg_velocity / (3 * self.velocity_threshold), 1.0)
            adaptive_alpha = self.alpha_min + velocity_ratio * (self.alpha_max - self.alpha_min)
        else:
            # 느린 움직임: 낮은 알파 (안정적 스무딩)
            adaptive_alpha = self.alpha_min
        
        return adaptive_alpha
    
    def reset(self, initial_angle: float = 0.0):
        """필터 리셋
        
        Args:
            initial_angle: 초기 각도 (rad)
        """
        self.ema_value = initial_angle
        self.is_initialized = False
        self.velocity_history = []
    
    def get_state(self) -> float:
        """현재 EMA 값 반환
        
        Returns:
            ema_value: 현재 EMA 값 (rad)
        """
        return self.ema_value
    
    def get_alpha(self) -> float:
        """현재 알파 값 반환
        
        Returns:
            alpha: 현재 알파 값
        """
        return self.alpha


class DoubleEMAFilter:
    """이중 지수 이동평균 (DEMA) 필터
    
    지연을 감소시키기 위한 향상된 EMA 필터
    """
    
    def __init__(self, config: EMAConfig):
        """DEMA 필터 초기화"""
        self.ema1 = EMAFilter(config)
        self.ema2 = EMAFilter(config)
        self.is_initialized = False
    
    def update(self, measured_angle: float) -> float:
        """DEMA 필터 업데이트
        
        DEMA = 2×EMA1 - EMA2(EMA1)
        """
        if not self.is_initialized:
            self.ema1.reset(measured_angle)
            self.ema2.reset(measured_angle)
            self.is_initialized = True
            return measured_angle
        
        # 첫 번째 EMA
        ema1_value = self.ema1.update(measured_angle)
        
        # 두 번째 EMA (첫 번째 EMA의 EMA)
        ema2_value = self.ema2.update(ema1_value)
        
        # DEMA 계산: 2×EMA1 - EMA2
        dema_diff1 = ang.diffpi(ema1_value, 0)
        dema_diff2 = ang.diffpi(ema2_value, 0)
        
        dema_value = 2 * dema_diff1 - dema_diff2
        dema_value = ang.wrap_pi(dema_value)
        
        return dema_value
    
    def reset(self, initial_angle: float = 0.0):
        """필터 리셋"""
        self.ema1.reset(initial_angle)
        self.ema2.reset(initial_angle)
        self.is_initialized = False


def create_ema_filter(alpha: float = 0.2, adaptive: bool = False) -> EMAFilter:
    """EMA 필터 간편 생성 함수
    
    Args:
        alpha: 스무딩 팩터 (0 < alpha <= 1)
        adaptive: 적응형 알파 사용 여부
        
    Returns:
        EMAFilter 인스턴스
    """
    config = EMAConfig(alpha=alpha, adaptive=adaptive)
    return EMAFilter(config)


def create_ema_from_window(window_size: int) -> EMAFilter:
    """윈도우 크기로부터 EMA 필터 생성
    
    Args:
        window_size: 이동평균 윈도우 크기
        
    Returns:
        EMAFilter 인스턴스
    """
    config = EMAConfig(alpha=2.0/(window_size+1), window_size=window_size)
    return EMAFilter(config)