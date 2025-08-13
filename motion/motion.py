# -*- coding: utf-8 -*-
"""
로봇 모션 시나리오 생성기 - 7가지 운동 패턴

다양한 산업용 로봇 관절 움직임을 시뮬레이션하는 모듈
각 운동 패턴의 특성을 반영한 각도, 각속도, 각가속도 프로파일 생성
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict
from enum import Enum
from src.pytemplate.core import ang


class MotionType(Enum):
    """로봇 운동 시나리오 타입"""
    CONSTANT = "constant"              # 정속 회전
    ACCELERATING = "accelerating"      # 선형 가속
    SINUSOIDAL = "sinusoidal"          # 사인파 속도
    TRAPEZOIDAL = "trapezoidal"        # 사다리꼴 속도
    STEP = "step"                      # 스텝 속도 변화
    RAMP = "ramp"                      # 램프 속도 증가  
    SCURVE = "scurve"                  # S-커브 프로파일


@dataclass
class MotionParams:
    """로봇 모션 파라미터"""
    motion_type: MotionType = MotionType.TRAPEZOIDAL
    duration: float = 10.0           # 전체 시간 (s)
    max_velocity: float = 2.0        # 최대 각속도 (rad/s) 
    acceleration_time: float = 2.0   # 가속/감속 시간 (s)
    frequency: float = 0.2           # 사인파 주파수 (Hz)
    step_time: float = 5.0           # 스텝 변화 시간 (s)
    noise_std: float = 0.01         # 측정 노이즈 표준편차 (rad)
    quantization_bits: int = 12     # ADC 양자화 비트
    dt: float = 0.01                # 샘플링 시간 (s)


class RobotMotionGenerator:
    """로봇 모션 시뮬레이터
    
    7가지 운동 시나리오:
    1. CONSTANT: 정속 회전
    2. ACCELERATING: 선형 가속 운동
    3. SINUSOIDAL: 사인파 속도 변화
    4. TRAPEZOIDAL: 사다리꼴 속도 프로파일
    5. STEP: 스텝 속도 변화
    6. RAMP: 램프 속도 증가
    7. SCURVE: S-커브 프로파일
    
    특징:
    - 다양한 산업용 로봇 운동 패턴
    - 실제 서보모터 특성 모사
    - 가우시안 노이즈 + 양자화 시뮬레이션
    """
    
    def __init__(self, params: MotionParams):
        """로봇 모션 생성기 초기화
        
        Args:
            params: 모션 파라미터
        """
        self.params = params
        self.dt = params.dt
        
        # 시간 배열 생성
        self.time_array = np.arange(0, params.duration + params.dt, params.dt)
        self.n_samples = len(self.time_array)
        
        # 모션 프로파일 계산
        self._calculate_motion_profile()
    
    def _calculate_motion_profile(self):
        """모션 프로파일 계산 (7가지 시나리오)"""
        self.velocity_profile = np.zeros(self.n_samples)
        self.angle_profile = np.zeros(self.n_samples)
        self.acceleration_profile = np.zeros(self.n_samples)
        
        if self.params.motion_type == MotionType.CONSTANT:
            self._calculate_constant_profile()
        elif self.params.motion_type == MotionType.ACCELERATING:
            self._calculate_accelerating_profile()
        elif self.params.motion_type == MotionType.SINUSOIDAL:
            self._calculate_sinusoidal_profile()
        elif self.params.motion_type == MotionType.TRAPEZOIDAL:
            self._calculate_trapezoidal_profile()
        elif self.params.motion_type == MotionType.STEP:
            self._calculate_step_profile()
        elif self.params.motion_type == MotionType.RAMP:
            self._calculate_ramp_profile()
        elif self.params.motion_type == MotionType.SCURVE:
            self._calculate_scurve_profile()
        else:
            raise ValueError(f"지원하지 않는 모션 타입: {self.params.motion_type}")
    
    def _calculate_constant_profile(self):
        """정속 회전 프로파일"""
        self.velocity_profile.fill(self.params.max_velocity)
        self.acceleration_profile.fill(0.0)
        
        # 각도 적분
        for i in range(1, self.n_samples):
            self.angle_profile[i] = self.angle_profile[i-1] + self.velocity_profile[i] * self.dt
    
    def _calculate_accelerating_profile(self):
        """선형 가속 프로파일"""
        acceleration = self.params.max_velocity / self.params.duration
        
        for i, t in enumerate(self.time_array):
            self.velocity_profile[i] = acceleration * t
            self.acceleration_profile[i] = acceleration
            
            if i > 0:
                self.angle_profile[i] = self.angle_profile[i-1] + self.velocity_profile[i] * self.dt
    
    def _calculate_sinusoidal_profile(self):
        """사인파 속도 변화 프로파일"""
        omega = 2 * np.pi * self.params.frequency
        
        for i, t in enumerate(self.time_array):
            # 속도: 사인파 변화 (항상 양수)
            self.velocity_profile[i] = self.params.max_velocity * (0.5 + 0.5 * np.sin(omega * t))
            # 가속도: 코사인
            self.acceleration_profile[i] = self.params.max_velocity * omega * 0.5 * np.cos(omega * t)
            
            if i > 0:
                self.angle_profile[i] = self.angle_profile[i-1] + self.velocity_profile[i] * self.dt
    
    def _calculate_trapezoidal_profile(self):
        """사다리꼴 각속도 프로파일"""
        t_accel = self.params.acceleration_time
        t_decel_start = self.params.duration - self.params.acceleration_time
        acceleration = self.params.max_velocity / t_accel
        
        for i, t in enumerate(self.time_array):
            if t <= t_accel:
                # 가속 구간
                velocity = acceleration * t
                accel = acceleration
            elif t <= t_decel_start:
                # 등속 구간
                velocity = self.params.max_velocity
                accel = 0.0
            else:
                # 감속 구간
                remaining_time = self.params.duration - t
                velocity = max(0, acceleration * remaining_time)
                accel = -acceleration
            
            self.velocity_profile[i] = velocity
            self.acceleration_profile[i] = accel
            
            if i > 0:
                self.angle_profile[i] = self.angle_profile[i-1] + velocity * self.dt
    
    def _calculate_step_profile(self):
        """스텝 속도 변화 프로파일"""
        step_index = int(self.params.step_time / self.params.dt)
        
        for i in range(self.n_samples):
            if i < step_index:
                self.velocity_profile[i] = 0.0
            else:
                self.velocity_profile[i] = self.params.max_velocity
            
            # 스텝에서 무한대 가속도 (실제로는 매우 큰 값)
            if i == step_index and step_index < self.n_samples - 1:
                self.acceleration_profile[i] = 100.0  # 큰 값으로 근사
            else:
                self.acceleration_profile[i] = 0.0
            
            if i > 0:
                self.angle_profile[i] = self.angle_profile[i-1] + self.velocity_profile[i] * self.dt
    
    def _calculate_ramp_profile(self):
        """램프 속도 증가 프로파일 (선형 증가)"""
        velocity_slope = self.params.max_velocity / self.params.duration
        
        for i, t in enumerate(self.time_array):
            self.velocity_profile[i] = min(velocity_slope * t, self.params.max_velocity)
            
            if t < self.params.duration - self.dt:
                self.acceleration_profile[i] = velocity_slope
            else:
                self.acceleration_profile[i] = 0.0
            
            if i > 0:
                self.angle_profile[i] = self.angle_profile[i-1] + self.velocity_profile[i] * self.dt
    
    def _calculate_scurve_profile(self):
        """S-커브 프로파일 (7세그먼트)"""
        t_total = self.params.duration
        t_accel = self.params.acceleration_time
        v_max = self.params.max_velocity
        
        # 7세그먼트 시간 분할 (단순화된 버전)
        t1 = t_accel / 3  # 저크 증가
        t2 = t_accel / 3  # 일정 가속
        t3 = t_accel / 3  # 저크 감소
        t4 = max(0, t_total - 2 * t_accel)  # 정속
        
        jerk_max = 6 * v_max / (t_accel * t_accel)
        accel_max = v_max / t_accel
        
        for i, t in enumerate(self.time_array):
            velocity = 0.0
            acceleration = 0.0
            
            if t <= t1:
                # 세그먼트 1: 저크 증가
                acceleration = 0.5 * jerk_max * t**2 / t1
                velocity = jerk_max * t**3 / (6 * t1)
            elif t <= t1 + t2:
                # 세그먼트 2: 일정 가속
                t_seg = t - t1
                acceleration = accel_max
                velocity = jerk_max * t1**2 / 6 + accel_max * t_seg
            elif t <= t1 + t2 + t3:
                # 세그먼트 3: 저크 감소
                t_seg = t - t1 - t2
                acceleration = accel_max * (1 - t_seg / t3)
                velocity = v_max - 0.5 * accel_max * (t3 - t_seg)**2 / t3
            elif t <= t1 + t2 + t3 + t4:
                # 세그먼트 4: 정속
                acceleration = 0
                velocity = v_max
            else:
                # 감속 구간 (대칭)
                t_remaining = t_total - t
                if t_remaining >= t3:
                    acceleration = 0
                    velocity = v_max
                elif t_remaining >= t2:
                    acceleration = -accel_max
                    velocity = v_max - accel_max * (t3 - t_remaining)
                else:
                    t_seg = t_remaining
                    acceleration = -0.5 * jerk_max * t_seg**2 / t1
                    velocity = jerk_max * t_seg**3 / (6 * t1)
            
            self.velocity_profile[i] = max(0, velocity)
            self.acceleration_profile[i] = acceleration
            
            if i > 0:
                self.angle_profile[i] = self.angle_profile[i-1] + self.velocity_profile[i] * self.dt
    
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
    
    def get_profile_info(self) -> Dict[str, str]:
        """프로파일 정보 반환
        
        Returns:
            프로파일 정보 딕셔너리
        """
        motion_names = {
            MotionType.CONSTANT: "Constant Velocity",
            MotionType.ACCELERATING: "Linear Acceleration", 
            MotionType.SINUSOIDAL: "Sinusoidal Velocity",
            MotionType.TRAPEZOIDAL: "Trapezoidal Velocity Profile",
            MotionType.STEP: "Step Velocity Change",
            MotionType.RAMP: "Ramp Velocity Increase",
            MotionType.SCURVE: "S-Curve Profile"
        }
        
        info = {
            'motion_type': motion_names[self.params.motion_type],
            'max_velocity': f"{self.params.max_velocity:.3f} rad/s ({self.params.max_velocity * 30/np.pi:.1f} RPM)",
            'total_duration': f"{self.params.duration:.1f} s",
            'total_angle': f"{self.angle_profile[-1]:.3f} rad ({np.degrees(self.angle_profile[-1]):.1f}°)",
            'noise_std': f"{self.params.noise_std:.4f} rad ({np.degrees(self.params.noise_std):.3f}°)",
            'quantization_bits': f"{self.params.quantization_bits} bits",
            'sampling_rate': f"{1/self.params.dt:.0f} Hz"
        }
        
        # 모션별 특화 정보 추가
        if self.params.motion_type == MotionType.TRAPEZOIDAL:
            info['acceleration_time'] = f"{self.params.acceleration_time:.1f} s"
            info['max_acceleration'] = f"{self.params.max_velocity/self.params.acceleration_time:.3f} rad/s² ({self.params.max_velocity/self.params.acceleration_time * 30/np.pi:.1f} RPM/s)"
        elif self.params.motion_type == MotionType.SINUSOIDAL:
            info['frequency'] = f"{self.params.frequency:.1f} Hz"
        elif self.params.motion_type == MotionType.STEP:
            info['step_time'] = f"{self.params.step_time:.1f} s"
        elif self.params.motion_type == MotionType.SCURVE:
            info['acceleration_time'] = f"{self.params.acceleration_time:.1f} s"
        
        return info


def create_robot_motion(motion_type: MotionType,
                       duration: float = 10.0, 
                       max_velocity: float = 2.0,
                       acceleration_time: float = 2.0,
                       frequency: float = 0.2,
                       step_time: float = 5.0,
                       noise_std: float = 0.01,
                       quantization_bits: int = 12) -> RobotMotionGenerator:
    """로봇 모션 시나리오 생성 헬퍼 함수
    
    Args:
        motion_type: 모션 타입
        duration: 전체 시간 (s)
        max_velocity: 최대 각속도 (rad/s)
        acceleration_time: 가속/감속 시간 (s)
        frequency: 사인파 주파수 (Hz)
        step_time: 스텝 변화 시간 (s)
        noise_std: 측정 노이즈 표준편차 (rad)
        quantization_bits: 양자화 비트 수
        
    Returns:
        RobotMotionGenerator 인스턴스
    """
    params = MotionParams(
        motion_type=motion_type,
        duration=duration,
        max_velocity=max_velocity,
        acceleration_time=acceleration_time,
        frequency=frequency,
        step_time=step_time,
        noise_std=noise_std,
        quantization_bits=quantization_bits
    )
    
    return RobotMotionGenerator(params)


# 기존 호환성을 위한 래퍼 함수
def create_robot_arm_scenario(duration: float = 10.0, 
                            max_velocity: float = 2.0,
                            acceleration_time: float = 2.0,
                            noise_std: float = 0.01,
                            quantization_bits: int = 12) -> RobotMotionGenerator:
    """사다리꼴 로봇 팔 시나리오 생성 (기존 호환성)"""
    return create_robot_motion(
        motion_type=MotionType.TRAPEZOIDAL,
        duration=duration,
        max_velocity=max_velocity,
        acceleration_time=acceleration_time,
        noise_std=noise_std,
        quantization_bits=quantization_bits
    )


def demo_all_motions():
    """모든 모션 시나리오 데모"""
    motion_types = list(MotionType)
    results = {}
    
    for motion_type in motion_types:
        print(f"\n=== {motion_type.value.upper()} 모션 시나리오 ===")
        
        # 모션별 특화 파라미터
        if motion_type == MotionType.SINUSOIDAL:
            robot = create_robot_motion(motion_type, frequency=0.3, max_velocity=1.5)
        elif motion_type == MotionType.STEP:
            robot = create_robot_motion(motion_type, step_time=4.0, max_velocity=2.0)
        elif motion_type == MotionType.SCURVE:
            robot = create_robot_motion(motion_type, acceleration_time=3.0, max_velocity=1.8)
        else:
            robot = create_robot_motion(motion_type, max_velocity=2.0)
        
        # 데이터 생성
        time, true_angles, measured_angles, true_velocities, true_accelerations = robot.generate_measurements()
        
        # 프로파일 정보 출력
        info = robot.get_profile_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        results[motion_type] = {
            'time': time,
            'true_angles': true_angles,
            'measured_angles': measured_angles,
            'true_velocities': true_velocities,
            'true_accelerations': true_accelerations,
            'info': info,
            'generator': robot
        }
    
    return results


# 기존 호환성 함수
def demo_robot_arm_motion():
    """사다리꼴 로봇 팔 모션 데모 (기존 호환성)"""
    robot = create_robot_arm_scenario(
        duration=8.0,
        max_velocity=1.5,
        acceleration_time=1.5,
        noise_std=0.02
    )
    
    time, true_angles, measured_angles, true_velocities, true_accelerations = robot.generate_measurements()
    
    info = robot.get_profile_info()
    print("=== 로봇 팔 엔코더 시나리오 ===")
    for key, value in info.items():
        print(f"{key}: {value}")
    
    print(f"\n샘플 수: {len(time)}")
    print(f"최종 각도: {true_angles[-1]:.3f} rad ({np.degrees(true_angles[-1]):.1f}°)")
    print(f"최대 측정 노이즈: {np.std(measured_angles - true_angles):.4f} rad")
    
    return time, true_angles, measured_angles, true_velocities, true_accelerations


if __name__ == "__main__":
    demo_all_motions()