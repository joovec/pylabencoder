# -*- coding: utf-8 -*-
"""
로봇 팔 엔코더 시나리오 모션 생성기

사다리꼴 각속도 프로파일을 사용한 로봇 팔 관절 움직임 시뮬레이션
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
from src.pytemplate.core import ang


@dataclass
class TrapezoidalMotionParams:
    """사다리꼴 모션 파라미터"""
    duration: float = 10.0           # 전체 시간 (s)
    max_velocity: float = 2.0        # 최대 각속도 (rad/s) 
    acceleration_time: float = 2.0   # 가속/감속 시간 (s)
    noise_std: float = 0.01         # 측정 노이즈 표준편차 (rad)
    quantization_bits: int = 12     # ADC 양자화 비트
    dt: float = 0.01                # 샘플링 시간 (s)


class RobotArmEncoder:
    """로봇 팔 엔코더 시뮬레이터
    
    사다리꼴 각속도 프로파일:
    - 구간 1: 가속 구간 (0 → max_velocity)
    - 구간 2: 등속 구간 (max_velocity 유지)  
    - 구간 3: 감속 구간 (max_velocity → 0)
    
    특징:
    - 매끄러운 속도 변화
    - 실제 서보모터 특성 모사
    - 가우시안 노이즈 + 양자화 시뮬레이션
    """
    
    def __init__(self, params: TrapezoidalMotionParams):
        """로봇 팔 엔코더 초기화
        
        Args:
            params: 사다리꼴 모션 파라미터
        """
        self.params = params
        self.dt = params.dt
        
        # 시간 배열 생성
        self.time_array = np.arange(0, params.duration + params.dt, params.dt)
        self.n_samples = len(self.time_array)
        
        # 사다리꼴 프로파일 계산
        self._calculate_trapezoidal_profile()
    
    def _calculate_trapezoidal_profile(self):
        """사다리꼴 각속도 프로파일 계산"""
        params = self.params
        time = self.time_array
        
        # 프로파일 구간 정의
        t_accel = params.acceleration_time
        t_decel_start = params.duration - params.acceleration_time
        
        # 가속도 계산
        acceleration = params.max_velocity / t_accel
        
        self.velocity_profile = np.zeros(self.n_samples)
        self.angle_profile = np.zeros(self.n_samples)
        self.acceleration_profile = np.zeros(self.n_samples)
        
        angle = 0.0
        
        for i, t in enumerate(time):
            if t <= t_accel:
                # 가속 구간
                velocity = acceleration * t
                accel = acceleration
            elif t <= t_decel_start:
                # 등속 구간
                velocity = params.max_velocity
                accel = 0.0
            else:
                # 감속 구간
                remaining_time = params.duration - t
                velocity = acceleration * remaining_time
                accel = -acceleration
            
            self.velocity_profile[i] = velocity
            self.acceleration_profile[i] = accel
            
            # 각도 적분 (사다리꼴 규칙)
            if i > 0:
                angle += (velocity + self.velocity_profile[i-1]) * self.dt / 2.0
            
            self.angle_profile[i] = angle
    
    def generate_measurements(self, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """측정값 시뮬레이션
        
        Args:
            seed: 랜덤 시드
            
        Returns:
            time: 시간 배열 (s)
            true_angles: 실제 각도 (rad)
            measured_angles: 측정된 각도 (rad) - 노이즈 + 양자화
            true_velocities: 실제 각속도 (rad/s)
            true_accelerations: 실제 각가속도 (rad/s²)
        """
        np.random.seed(seed)
        
        # 가우시안 노이즈 추가
        noise = np.random.normal(0, self.params.noise_std, self.n_samples)
        noisy_angles = self.angle_profile + noise
        
        # 양자화 시뮬레이션
        if self.params.quantization_bits > 0:
            # 각도 범위를 양자화 레벨로 분할
            angle_range = 4 * np.pi  # ±2π 범위
            n_levels = 2 ** self.params.quantization_bits
            quantization_step = angle_range / n_levels
            
            # 양자화 적용
            quantized_angles = np.round(noisy_angles / quantization_step) * quantization_step
        else:
            quantized_angles = noisy_angles
        
        # 각도를 -π ~ π 범위로 래핑
        measured_angles = np.array([ang.wrap_pi(angle) for angle in quantized_angles])
        true_angles_wrapped = np.array([ang.wrap_pi(angle) for angle in self.angle_profile])
        
        return (
            self.time_array.copy(),
            true_angles_wrapped,
            measured_angles,
            self.velocity_profile.copy(),
            self.acceleration_profile.copy()
        )
    
    def get_profile_info(self) -> dict:
        """프로파일 정보 반환
        
        Returns:
            프로파일 정보 딕셔너리
        """
        return {
            'motion_type': 'Trapezoidal Velocity Profile',
            'max_velocity': f"{self.params.max_velocity:.3f} rad/s ({self.params.max_velocity * 30/np.pi:.1f} RPM)",
            'acceleration_time': f"{self.params.acceleration_time:.1f} s",
            'total_duration': f"{self.params.duration:.1f} s",
            'max_acceleration': f"{self.params.max_velocity/self.params.acceleration_time:.3f} rad/s² ({self.params.max_velocity/self.params.acceleration_time * 30/np.pi:.1f} RPM/s)",
            'total_angle': f"{self.angle_profile[-1]:.3f} rad ({np.degrees(self.angle_profile[-1]):.1f}°)",
            'noise_std': f"{self.params.noise_std:.4f} rad ({np.degrees(self.params.noise_std):.3f}°)",
            'quantization_bits': f"{self.params.quantization_bits} bits",
            'sampling_rate': f"{1/self.params.dt:.0f} Hz"
        }


def create_robot_arm_scenario(duration: float = 10.0, 
                            max_velocity: float = 2.0,
                            acceleration_time: float = 2.0,
                            noise_std: float = 0.01,
                            quantization_bits: int = 12) -> RobotArmEncoder:
    """로봇 팔 시나리오 생성 헬퍼 함수
    
    Args:
        duration: 전체 시간 (s)
        max_velocity: 최대 각속도 (rad/s)
        acceleration_time: 가속/감속 시간 (s)
        noise_std: 측정 노이즈 표준편차 (rad)
        quantization_bits: 양자화 비트 수
        
    Returns:
        RobotArmEncoder 인스턴스
    """
    params = TrapezoidalMotionParams(
        duration=duration,
        max_velocity=max_velocity,
        acceleration_time=acceleration_time,
        noise_std=noise_std,
        quantization_bits=quantization_bits
    )
    
    return RobotArmEncoder(params)


def demo_robot_arm_motion():
    """로봇 팔 모션 데모"""
    # 로봇 팔 시나리오 생성
    robot = create_robot_arm_scenario(
        duration=8.0,
        max_velocity=1.5,
        acceleration_time=1.5,
        noise_std=0.02
    )
    
    # 데이터 생성
    time, true_angles, measured_angles, true_velocities, true_accelerations = robot.generate_measurements()
    
    # 프로파일 정보 출력
    info = robot.get_profile_info()
    print("=== 로봇 팔 엔코더 시나리오 ===")
    for key, value in info.items():
        print(f"{key}: {value}")
    
    print(f"\n샘플 수: {len(time)}")
    print(f"최종 각도: {true_angles[-1]:.3f} rad ({np.degrees(true_angles[-1]):.1f}°)")
    print(f"최대 측정 노이즈: {np.std(measured_angles - true_angles):.4f} rad")
    
    return time, true_angles, measured_angles, true_velocities, true_accelerations


if __name__ == "__main__":
    demo_robot_arm_motion()