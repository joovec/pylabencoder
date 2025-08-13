# -*- coding: utf-8 -*-
"""
CA (Constant Acceleration) 칼만 필터 - 순환 혁신 방식

3차 상태 벡터: [각도, 각속도, 각가속도]
각도의 모듈러 특성을 고려한 순환 혁신 방식 적용
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple
from src.pytemplate.core import ang


@dataclass
class CAConfig:
    """CA 칼만 필터 설정"""
    dt: float = 0.01            # 샘플링 시간 (s)
    process_noise: float = 0.001 # 프로세스 노이즈 분산
    measurement_noise: float = 0.01 # 측정 노이즈 분산


class CAKalmanFilter:
    """CA (Constant Acceleration) 칼만 필터
    
    상태 벡터: x = [θ, ω, α]^T
    - θ: 각도 (rad)
    - ω: 각속도 (rad/s)
    - α: 각가속도 (rad/s²)
    
    순환 혁신 방식:
    - 측정값과 예측값의 각도 차이를 순환적으로 계산
    - 각도 오버플로 및 누적 오차 방지
    """
    
    def __init__(self, config: CAConfig):
        """칼만 필터 초기화
        
        Args:
            config: CA 칼만 필터 설정
        """
        self.config = config
        self.dt = config.dt
        
        # 상태 벡터 [각도, 각속도, 각가속도]
        self.x = np.array([0.0, 0.0, 0.0])
        
        # 공분산 행렬
        self.P = np.eye(3) * 1.0
        
        # 상태 전이 행렬 (CA 모델)
        dt2 = self.dt * self.dt
        self.F = np.array([
            [1.0, self.dt, dt2/2],
            [0.0, 1.0, self.dt],
            [0.0, 0.0, 1.0]
        ])
        
        # 프로세스 노이즈 공분산 행렬
        dt3 = dt2 * self.dt
        dt4 = dt3 * self.dt
        dt5 = dt4 * self.dt
        
        self.Q = np.array([
            [dt5/20, dt4/8,  dt3/6],
            [dt4/8,  dt3/3,  dt2/2],
            [dt3/6,  dt2/2,  self.dt]
        ]) * config.process_noise
        
        # 측정 행렬 (각도만 측정)
        self.H = np.array([[1.0, 0.0, 0.0]])
        
        # 측정 노이즈 분산
        self.R = np.array([[config.measurement_noise]])
        
        self.is_initialized = False
    
    def predict(self) -> Tuple[float, float, float]:
        """예측 단계
        
        Returns:
            predicted_angle: 예측된 각도 (rad)
            predicted_velocity: 예측된 각속도 (rad/s)
            predicted_acceleration: 예측된 각가속도 (rad/s²)
        """
        # 상태 예측
        self.x = self.F @ self.x
        
        # 각도를 -π ~ π 범위로 정규화
        self.x[0] = ang.wrap_pi(self.x[0])
        
        # 공분산 예측
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x[0], self.x[1], self.x[2]
    
    def update(self, measured_angle: float) -> Tuple[float, float, float]:
        """업데이트 단계 (순환 혁신 방식)
        
        Args:
            measured_angle: 측정된 각도 (rad)
            
        Returns:
            estimated_angle: 추정된 각도 (rad)
            estimated_velocity: 추정된 각속도 (rad/s)
            estimated_acceleration: 추정된 각가속도 (rad/s²)
        """
        if not self.is_initialized:
            # 첫 측정값으로 초기화
            self.x[0] = measured_angle
            self.x[1] = 0.0
            self.x[2] = 0.0
            self.is_initialized = True
            return self.x[0], self.x[1], self.x[2]
        
        # 예측 단계
        predicted_angle, predicted_velocity, predicted_acceleration = self.predict()
        
        # 순환 혁신 계산
        innovation = ang.diffpi(measured_angle, predicted_angle)
        
        # 혁신 공분산
        S = self.H @ self.P @ self.H.T + self.R
        
        # 칼만 게인
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 상태 업데이트
        self.x = self.x + K.flatten() * innovation
        
        # 각도 정규화
        self.x[0] = ang.wrap_pi(self.x[0])
        
        # 공분산 업데이트
        I_KH = np.eye(3) - K @ self.H
        self.P = I_KH @ self.P
        
        return self.x[0], self.x[1], self.x[2]
    
    def get_state(self) -> Tuple[float, float, float]:
        """현재 상태 반환
        
        Returns:
            angle: 현재 추정된 각도 (rad)
            velocity: 현재 추정된 각속도 (rad/s)
            acceleration: 현재 추정된 각가속도 (rad/s²)
        """
        return self.x[0], self.x[1], self.x[2]
    
    def reset(self, initial_angle: float = 0.0):
        """필터 리셋
        
        Args:
            initial_angle: 초기 각도 (rad)
        """
        self.x = np.array([initial_angle, 0.0, 0.0])
        self.P = np.eye(3) * 1.0
        self.is_initialized = False