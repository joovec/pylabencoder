# -*- coding: utf-8 -*-
"""
FIR (Finite Impulse Response) 필터 - 순환 각도용

유한 임펄스 응답 저역통과 필터로 각도 측정값 필터링
이동평균 기반의 안정적인 스무딩 필터
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from collections import deque
from src.pytemplate.core import ang


@dataclass
class FIRConfig:
    """FIR 필터 설정"""
    window_size: int = 10           # 윈도우 크기 (필터 차수 + 1)
    filter_type: str = "moving_average"  # "moving_average", "hamming", "blackman", "kaiser"
    beta: float = 8.6               # Kaiser 윈도우용 베타 파라미터
    cutoff_freq: float = None       # 차단 주파수 (Hz) - 참고용
    dt: float = 0.01                # 샘플링 시간 (s)


class FIRFilter:
    """유한 임펄스 응답 필터
    
    공식: y[n] = Σ(h[k] * x[n-k]) for k=0 to N-1
    
    특징:
    - 선형 위상 특성 (지연 일정)
    - 안정성 보장
    - 순환 각도 처리 최적화
    - 다양한 윈도우 함수 지원
    """
    
    def __init__(self, config: FIRConfig):
        """FIR 필터 초기화
        
        Args:
            config: FIR 필터 설정
        """
        self.config = config
        self.window_size = config.window_size
        self.filter_coeffs = self._generate_filter_coefficients()
        
        # 입력 버퍼 (순환 큐)
        self.input_buffer = deque(maxlen=self.window_size)
        self.is_initialized = False
        
        # 지연 보상
        self.group_delay = (self.window_size - 1) / 2
        
        # 차단 주파수 추정 (이동평균의 경우)
        if config.cutoff_freq is None and config.filter_type == "moving_average":
            # 이동평균의 근사적 차단주파수: fc ≈ 0.443 / (N * dt)
            self.cutoff_freq = 0.443 / (self.window_size * config.dt)
        else:
            self.cutoff_freq = config.cutoff_freq
    
    def _generate_filter_coefficients(self) -> np.ndarray:
        """필터 계수 생성"""
        N = self.window_size
        
        if self.config.filter_type == "moving_average":
            # 단순 이동평균 (박스카 윈도우)
            coeffs = np.ones(N) / N
            
        elif self.config.filter_type == "hamming":
            # 해밍 윈도우 기반 저역통과 필터
            hamming = np.hamming(N)
            coeffs = hamming / np.sum(hamming)
            
        elif self.config.filter_type == "blackman":
            # 블랙만 윈도우 기반 저역통과 필터
            blackman = np.blackman(N)
            coeffs = blackman / np.sum(blackman)
            
        elif self.config.filter_type == "kaiser":
            # 카이저 윈도우 기반 저역통과 필터
            kaiser = np.kaiser(N, self.config.beta)
            coeffs = kaiser / np.sum(kaiser)
            
        else:
            # 기본값: 이동평균
            coeffs = np.ones(N) / N
        
        return coeffs
    
    def update(self, measured_angle: float) -> float:
        """FIR 필터 업데이트
        
        Args:
            measured_angle: 측정된 각도 (rad)
            
        Returns:
            filtered_angle: 필터링된 각도 (rad)
        """
        # 버퍼에 새 값 추가
        self.input_buffer.append(measured_angle)
        
        # 버퍼가 충분히 차지 않은 경우
        if len(self.input_buffer) < self.window_size:
            if not self.is_initialized:
                self.is_initialized = True
                return measured_angle
            else:
                # 부분적으로 채워진 버퍼로 계산 (패딩 없이)
                return self._compute_partial_output()
        
        # 완전한 FIR 필터링
        return self._compute_full_output()
    
    def _compute_full_output(self) -> float:
        """완전한 FIR 출력 계산 (버퍼가 완전히 찬 경우)"""
        # 입력 버퍼를 배열로 변환 (최신 값이 마지막)
        input_array = np.array(self.input_buffer)
        
        # 순환 각도 차이 기반 필터링
        reference_angle = input_array[-1]  # 최신 값을 기준으로
        
        # 각도 차이들 계산 (순환 차이)
        angle_diffs = np.array([ang.diffpi(angle, reference_angle) for angle in input_array])
        
        # FIR 필터 적용 (역순으로 곱하기 - 최신 값부터)
        filtered_diff = np.dot(self.filter_coeffs, angle_diffs[::-1])
        
        # 기준각에 필터링된 차이 적용
        filtered_angle = reference_angle + filtered_diff
        
        # -π ~ π 범위로 정규화
        return ang.wrap_pi(filtered_angle)
    
    def _compute_partial_output(self) -> float:
        """부분적인 출력 계산 (버퍼가 완전히 차지 않은 경우)"""
        if len(self.input_buffer) == 1:
            return self.input_buffer[0]
        
        # 현재까지 채워진 버퍼로만 계산
        input_array = np.array(self.input_buffer)
        current_size = len(input_array)
        
        # 현재 크기에 맞는 정규화된 계수
        partial_coeffs = self.filter_coeffs[:current_size]
        partial_coeffs = partial_coeffs / np.sum(partial_coeffs)
        
        # 순환 각도 차이 기반 필터링
        reference_angle = input_array[-1]
        angle_diffs = np.array([ang.diffpi(angle, reference_angle) for angle in input_array])
        
        # 부분 FIR 필터 적용
        filtered_diff = np.dot(partial_coeffs, angle_diffs[::-1])
        filtered_angle = reference_angle + filtered_diff
        
        return ang.wrap_pi(filtered_angle)
    
    def reset(self, initial_angle: float = 0.0):
        """필터 리셋
        
        Args:
            initial_angle: 초기 각도 (rad) - 사용되지 않음 (FIR은 초기값 불필요)
        """
        self.input_buffer.clear()
        self.is_initialized = False
    
    def get_coefficients(self) -> np.ndarray:
        """필터 계수 반환
        
        Returns:
            coeffs: 필터 계수 배열
        """
        return self.filter_coeffs.copy()
    
    def get_group_delay(self) -> float:
        """그룹 지연 반환
        
        Returns:
            delay: 그룹 지연 (샘플 단위)
        """
        return self.group_delay
    
    def get_cutoff_frequency(self) -> Optional[float]:
        """추정된 차단 주파수 반환
        
        Returns:
            cutoff_freq: 차단 주파수 (Hz) 또는 None
        """
        return self.cutoff_freq
    
    def get_filter_info(self) -> dict:
        """필터 정보 반환
        
        Returns:
            info: 필터 정보 딕셔너리
        """
        return {
            'type': 'FIR',
            'window_size': self.window_size,
            'filter_type': self.config.filter_type,
            'group_delay': self.group_delay,
            'cutoff_freq': self.cutoff_freq,
            'coefficients': self.filter_coeffs.tolist()
        }


class AdaptiveFIRFilter:
    """적응형 FIR 필터
    
    각속도에 따라 윈도우 크기를 동적으로 조정
    """
    
    def __init__(self, config: FIRConfig):
        """적응형 FIR 필터 초기화"""
        self.base_config = config
        self.min_window = max(3, config.window_size // 2)
        self.max_window = config.window_size
        self.current_window = config.window_size
        
        # 현재 FIR 필터
        self.fir_filter = FIRFilter(config)
        
        # 속도 추정용
        self.prev_angle = 0.0
        self.velocity_history = deque(maxlen=5)
        self.velocity_threshold = 0.5  # rad/s
        
    def update(self, measured_angle: float) -> float:
        """적응형 FIR 필터 업데이트"""
        # 각속도 추정
        if len(self.velocity_history) > 0:
            angle_diff = ang.diffpi(measured_angle, self.prev_angle)
            velocity = abs(angle_diff / self.base_config.dt)
            self.velocity_history.append(velocity)
            
            # 평균 속도 계산
            avg_velocity = np.mean(self.velocity_history)
            
            # 윈도우 크기 조정
            if avg_velocity > self.velocity_threshold:
                # 빠른 움직임: 작은 윈도우 (빠른 응답)
                target_window = self.min_window
            else:
                # 느린 움직임: 큰 윈도우 (안정적 스무딩)
                target_window = self.max_window
            
            # 윈도우 크기 변경이 필요한 경우
            if target_window != self.current_window:
                self.current_window = target_window
                new_config = FIRConfig(
                    window_size=target_window,
                    filter_type=self.base_config.filter_type,
                    beta=self.base_config.beta,
                    dt=self.base_config.dt
                )
                self.fir_filter = FIRFilter(new_config)
        
        self.prev_angle = measured_angle
        return self.fir_filter.update(measured_angle)
    
    def reset(self, initial_angle: float = 0.0):
        """적응형 필터 리셋"""
        self.fir_filter.reset()
        self.velocity_history.clear()
        self.prev_angle = initial_angle
        self.current_window = self.base_config.window_size


def create_fir_filter(window_size: int = 10, 
                     filter_type: str = "moving_average",
                     dt: float = 0.01) -> FIRFilter:
    """FIR 필터 간편 생성 함수
    
    Args:
        window_size: 윈도우 크기
        filter_type: 필터 타입 ("moving_average", "hamming", "blackman", "kaiser")
        dt: 샘플링 시간 (s)
        
    Returns:
        FIRFilter 인스턴스
    """
    config = FIRConfig(window_size=window_size, filter_type=filter_type, dt=dt)
    return FIRFilter(config)


def create_moving_average_filter(window_size: int = 10, dt: float = 0.01) -> FIRFilter:
    """이동평균 필터 생성 (가장 일반적인 FIR 필터)
    
    Args:
        window_size: 이동평균 윈도우 크기
        dt: 샘플링 시간 (s)
        
    Returns:
        FIRFilter 인스턴스
    """
    return create_fir_filter(window_size, "moving_average", dt)