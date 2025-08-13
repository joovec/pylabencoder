# -*- coding: utf-8 -*-
"""
CJ (Constant Jerk) 칼만 필터 - 순환 혁신 방식

4차 상태 벡터: [각도, 각속도, 각가속도, 저크]
각도의 모듈러 특성을 고려한 순환 혁신 방식 적용
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple
from src.pytemplate.core import ang


@dataclass
class CJConfig:
    """CJ 칼만 필터 설정"""
    dt: float = 0.01            # 샘플링 시간 (s)
    process_noise: float = 0.001 # 프로세스 노이즈 분산
    measurement_noise: float = 0.01 # 측정 노이즈 분산


class CJKalmanFilter:
    """CJ (Constant Jerk) 칼만 필터
    
    상태 벡터: x = [θ, ω, α, j]^T
    - θ: 각도 (rad)
    - ω: 각속도 (rad/s)
    - α: 각가속도 (rad/s²)
    - j: 저크 (rad/s³)
    
    순환 혁신 방식:
    - 측정값과 예측값의 각도 차이를 순환적으로 계산
    - 각도 오버플로 및 누적 오차 방지
    """
    
    def __init__(self, config: CJConfig):
        """칼만 필터 초기화
        
        Args:
            config: CJ 칼만 필터 설정
        """
        self.config = config
        self.dt = config.dt
        
        # 상태 벡터 [각도, 각속도, 각가속도, 저크]
        self.x = np.array([0.0, 0.0, 0.0, 0.0])
        
        # 공분산 행렬
        self.P = np.eye(4) * 1.0
        
        # 상태 전이 행렬 (CJ 모델)
        dt2 = self.dt * self.dt
        dt3 = dt2 * self.dt
        
        self.F = np.array([
            [1.0, self.dt, dt2/2, dt3/6],
            [0.0, 1.0,     self.dt, dt2/2],
            [0.0, 0.0,     1.0,     self.dt],
            [0.0, 0.0,     0.0,     1.0]
        ])
        
        # 프로세스 노이즈 공분산 행렬 (4차 시스템)
        dt4 = dt3 * self.dt
        dt5 = dt4 * self.dt
        dt6 = dt5 * self.dt
        dt7 = dt6 * self.dt
        
        self.Q = np.array([
            [dt7/252, dt6/72,  dt5/30, dt4/24],
            [dt6/72,  dt5/20,  dt4/8,  dt3/6],
            [dt5/30,  dt4/8,   dt3/3,  dt2/2],
            [dt4/24,  dt3/6,   dt2/2,  self.dt]
        ]) * config.process_noise
        
        # 측정 행렬 (각도만 측정)
        self.H = np.array([[1.0, 0.0, 0.0, 0.0]])
        
        # 측정 노이즈 분산
        self.R = np.array([[config.measurement_noise]])
        
        self.is_initialized = False
    
    def predict(self) -> Tuple[float, float, float, float]:
        """예측 단계
        
        Returns:
            predicted_angle: 예측된 각도 (rad)
            predicted_velocity: 예측된 각속도 (rad/s)
            predicted_acceleration: 예측된 각가속도 (rad/s²)
            predicted_jerk: 예측된 저크 (rad/s³)
        """
        # 상태 예측
        self.x = self.F @ self.x
        
        # 각도를 -π ~ π 범위로 정규화
        self.x[0] = ang.wrap_pi(self.x[0])
        
        # 공분산 예측
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x[0], self.x[1], self.x[2], self.x[3]
    
    def update(self, measured_angle: float) -> Tuple[float, float, float, float]:
        """업데이트 단계 (순환 혁신 방식)
        
        Args:
            measured_angle: 측정된 각도 (rad)
            
        Returns:
            estimated_angle: 추정된 각도 (rad)
            estimated_velocity: 추정된 각속도 (rad/s)
            estimated_acceleration: 추정된 각가속도 (rad/s²)
            estimated_jerk: 추정된 저크 (rad/s³)
        """
        if not self.is_initialized:
            # 첫 측정값으로 초기화
            self.x[0] = measured_angle
            self.x[1] = 0.0
            self.x[2] = 0.0
            self.x[3] = 0.0
            self.is_initialized = True
            return self.x[0], self.x[1], self.x[2], self.x[3]
        
        # 예측 단계
        predicted_angle, predicted_velocity, predicted_acceleration, predicted_jerk = self.predict()
        
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
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ self.P
        
        return self.x[0], self.x[1], self.x[2], self.x[3]
    
    def get_state(self) -> Tuple[float, float, float, float]:
        """현재 상태 반환
        
        Returns:
            angle: 현재 추정된 각도 (rad)
            velocity: 현재 추정된 각속도 (rad/s)
            acceleration: 현재 추정된 각가속도 (rad/s²)
            jerk: 현재 추정된 저크 (rad/s³)
        """
        return self.x[0], self.x[1], self.x[2], self.x[3]
    
    def reset(self, initial_angle: float = 0.0):
        """필터 리셋
        
        Args:
            initial_angle: 초기 각도 (rad)
        """
        self.x = np.array([initial_angle, 0.0, 0.0, 0.0])
        self.P = np.eye(4) * 1.0
        self.is_initialized = False