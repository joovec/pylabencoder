# -*- coding: utf-8 -*-
"""
IIR (Infinite Impulse Response) 필터 - 순환 각도용

1차 IIR 저역통과 필터로 각도 측정값 필터링
칼만 필터와의 성능 비교를 위한 기준 모델
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple
from src.pytemplate.core import ang


@dataclass
class IIRConfig:
    """IIR 필터 설정"""
    alpha: float = 0.1              # 필터 계수 (0 < alpha < 1)
    dt: float = 0.01                # 샘플링 시간 (s)
    cutoff_freq: float = None       # 차단 주파수 (Hz, alpha로부터 계산)


class IIRFilter:
    """1차 IIR 저역통과 필터
    
    공식: y[n] = α × x[n] + (1-α) × y[n-1]
    
    특징:
    - 간단한 1차 필터 구조
    - 실시간 처리 최적화
    - 순환 각도 특성 고려
    - 메모리 효율적
    """
    
    def __init__(self, config: IIRConfig):
        """IIR 필터 초기화
        
        Args:
            config: IIR 필터 설정
        """
        self.config = config
        self.alpha = config.alpha
        self.filtered_value = 0.0
        self.is_initialized = False
        
        # 차단 주파수 계산 (참고용)
        if config.cutoff_freq is None and config.dt is not None:
            # alpha = 2πfcΔt / (1 + 2πfcΔt) 역산
            self.cutoff_freq = self.alpha / (2 * np.pi * config.dt * (1 - self.alpha))
        else:
            self.cutoff_freq = config.cutoff_freq
    
    def update(self, measured_angle: float) -> float:
        """IIR 필터 업데이트
        
        Args:
            measured_angle: 측정된 각도 (rad)
            
        Returns:
            filtered_angle: 필터링된 각도 (rad)
        """
        if not self.is_initialized:
            # 첫 번째 값으로 초기화
            self.filtered_value = measured_angle
            self.is_initialized = True
            return self.filtered_value
        
        # 순환 각도 차이 계산
        angle_diff = ang.diffpi(measured_angle, self.filtered_value)
        
        # IIR 필터 적용
        self.filtered_value += self.alpha * angle_diff
        
        # -π ~ π 범위로 정규화
        self.filtered_value = ang.wrap_pi(self.filtered_value)
        
        return self.filtered_value
    
    def reset(self, initial_angle: float = 0.0):
        """필터 리셋
        
        Args:
            initial_angle: 초기 각도 (rad)
        """
        self.filtered_value = initial_angle
        self.is_initialized = False
    
    def get_state(self) -> float:
        """현재 필터링된 각도 반환
        
        Returns:
            angle: 현재 필터링된 각도 (rad)
        """
        return self.filtered_value
    
    def get_cutoff_frequency(self) -> float:
        """추정된 차단 주파수 반환
        
        Returns:
            cutoff_freq: 차단 주파수 (Hz)
        """
        return self.cutoff_freq
    
    def get_alpha(self) -> float:
        """필터 계수 반환
        
        Returns:
            alpha: 필터 계수
        """
        return self.alpha


def create_iir_filter(alpha: float = 0.1, dt: float = 0.01) -> IIRFilter:
    """IIR 필터 간편 생성 함수
    
    Args:
        alpha: 필터 계수 (0 < alpha < 1)
        dt: 샘플링 시간 (s)
        
    Returns:
        IIRFilter 인스턴스
    """
    config = IIRConfig(alpha=alpha, dt=dt)
    return IIRFilter(config)


def alpha_from_cutoff_freq(cutoff_freq: float, dt: float) -> float:
    """차단 주파수로부터 알파 계수 계산
    
    Args:
        cutoff_freq: 차단 주파수 (Hz)
        dt: 샘플링 시간 (s)
        
    Returns:
        alpha: 필터 계수
    """
    omega_c = 2 * np.pi * cutoff_freq
    return omega_c * dt / (1 + omega_c * dt)