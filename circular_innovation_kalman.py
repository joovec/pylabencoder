# -*- coding: utf-8 -*-
"""
순환 혁신(Circular Innovation) 방식의 칼만 필터 구현.

각도의 모듈러 특성을 올바르게 처리하기 위해 순환 혁신을 사용합니다.
언래핑의 누적 오차와 오버플로 문제를 근본적으로 해결합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from typing import Tuple, Optional
from dataclasses import dataclass

from src.pytemplate.core import ang
from src.pytemplate.utils.plot_helper import setup_korean_font


@dataclass
class KalmanConfig:
    """칼만 필터 설정을 위한 데이터 클래스."""
    dt: float = 0.01  # 샘플링 시간
    process_noise: float = 0.01  # 프로세스 노이즈
    measurement_noise: float = 0.1  # 측정 노이즈
    initial_position_variance: float = 1.0  # 초기 위치 분산
    initial_velocity_variance: float = 0.1  # 초기 속도 분산


class CircularInnovationKalman:
    """순환 혁신 방식의 칼만 필터.
    
    이 방식은 각도 차이를 모듈러 공간에서 직접 계산하여
    언래핑 없이도 정확한 추정을 수행합니다.
    """
    
    def __init__(self, config: KalmanConfig):
        self.config = config
        self.kf = self._initialize_filter()
        
    def _initialize_filter(self) -> KalmanFilter:
        """filterpy 칼만 필터 초기화."""
        kf = KalmanFilter(dim_x=2, dim_z=1)
        
        # 상태 전이 행렬 (등속 모델)
        kf.F = np.array([
            [1.0, self.config.dt],
            [0.0, 1.0]
        ])
        
        # 측정 행렬
        kf.H = np.array([[1.0, 0.0]])
        
        # 프로세스 노이즈 공분산
        q = self.config.process_noise
        kf.Q = q * np.array([
            [self.config.dt**3/3, self.config.dt**2/2],
            [self.config.dt**2/2, self.config.dt]
        ])
        
        # 측정 노이즈 공분산
        kf.R = np.array([[self.config.measurement_noise**2]])
        
        # 초기 상태 공분산
        kf.P = np.diag([
            self.config.initial_position_variance,
            self.config.initial_velocity_variance
        ])
        
        # 초기 상태
        kf.x = np.array([[0], [0]])
        
        return kf
    
    def update(self, measured_angle: float) -> Tuple[float, float]:
        """
        순환 혁신을 사용한 칼만 필터 업데이트.
        
        핵심 아이디어:
        1. 예측된 각도를 -π ~ π 범위로 정규화
        2. 측정값과 예측값의 차이를 모듈러 공간에서 계산 (diffpi)
        3. 이 차이(혁신)를 사용하여 상태 업데이트
        
        Args:
            measured_angle: 측정된 각도 (rad, -π ~ π)
            
        Returns:
            estimated_angle: 추정된 각도 (-π ~ π)
            estimated_velocity: 추정된 각속도
        """
        # 예측 단계
        self.kf.predict()
        
        # 순환 혁신 계산 (핵심 부분)
        predicted_angle = ang.wrap_pi(self.kf.x[0, 0])
        innovation = ang.diffpi(measured_angle, predicted_angle)
        
        # 칼만 게인 계산
        S = self.kf.H @ self.kf.P @ self.kf.H.T + self.kf.R
        K = self.kf.P @ self.kf.H.T / S[0, 0]
        
        # 상태 업데이트
        self.kf.x = self.kf.x + K * innovation
        
        # 공분산 업데이트
        self.kf.P = (np.eye(2) - K @ self.kf.H) @ self.kf.P
        
        # 추정값 반환 (각도는 wrap)
        estimated_angle = ang.wrap_pi(self.kf.x[0, 0])
        estimated_velocity = self.kf.x[1, 0]
        
        return estimated_angle, estimated_velocity
    
    def reset(self, initial_angle: float = 0.0, initial_velocity: float = 0.0):
        """필터를 초기 상태로 리셋."""
        self.kf.x = np.array([[initial_angle], [initial_velocity]])


class AngleDataGenerator:
    """각도 데이터 생성 클래스."""
    
    def __init__(self, dt: float = 0.01):
        self.dt = dt
    
    def generate_motion(
        self,
        duration: float,
        motion_type: str = "constant",
        base_velocity: float = 0.5,
        acceleration: float = 0.1,
        noise_std: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        다양한 운동 패턴의 각도 데이터 생성.
        
        Args:
            duration: 시뮬레이션 시간
            motion_type: "constant", "accelerating", "sinusoidal"
            base_velocity: 기본 각속도
            acceleration: 각가속도 (accelerating 모드)
            noise_std: 측정 노이즈 표준편차
        """
        num_samples = int(duration / self.dt)
        time_array = np.linspace(0, duration, num_samples)
        
        if motion_type == "constant":
            true_angles_unwrapped = base_velocity * time_array
            
        elif motion_type == "accelerating":
            true_angles_unwrapped = (base_velocity * time_array + 
                                    0.5 * acceleration * time_array**2)
            
        elif motion_type == "sinusoidal":
            freq = 0.2  # Hz
            amplitude = 2.0  # rad
            true_angles_unwrapped = amplitude * np.sin(2 * np.pi * freq * time_array)
            
        else:
            raise ValueError(f"Unknown motion type: {motion_type}")
        
        # Wrap to -π ~ π
        true_angles = np.array([ang.wrap_pi(angle) for angle in true_angles_unwrapped])
        
        # 노이즈 추가
        noise = np.random.normal(0, noise_std, num_samples)
        measured_angles_noisy = true_angles_unwrapped + noise
        measured_angles = np.array([ang.wrap_pi(angle) for angle in measured_angles_noisy])
        
        return time_array, true_angles, measured_angles


class KalmanAnalyzer:
    """칼만 필터 성능 분석 및 시각화."""
    
    def __init__(self):
        setup_korean_font()
        
    def run_simulation(
        self,
        kalman_filter: CircularInnovationKalman,
        data_generator: AngleDataGenerator,
        duration: float = 30.0,
        motion_type: str = "constant",
        base_velocity: float = 0.5,
        noise_std: float = 0.1
    ) -> dict:
        """시뮬레이션 실행."""
        
        # 데이터 생성
        time_array, true_angles, measured_angles = data_generator.generate_motion(
            duration=duration,
            motion_type=motion_type,
            base_velocity=base_velocity,
            noise_std=noise_std
        )
        
        # 칼만 필터 실행
        kalman_filter.reset(initial_angle=measured_angles[0])
        estimated_angles = []
        estimated_velocities = []
        innovations = []  # 혁신 기록
        
        for measured in measured_angles:
            # 혁신 계산 (디버깅/분석용)
            if len(estimated_angles) > 0:
                innovation = ang.diffpi(measured, estimated_angles[-1])
                innovations.append(innovation)
            
            angle, velocity = kalman_filter.update(measured)
            estimated_angles.append(angle)
            estimated_velocities.append(velocity)
        
        return {
            'time': time_array,
            'true_angles': true_angles,
            'measured_angles': measured_angles,
            'estimated_angles': np.array(estimated_angles),
            'estimated_velocities': np.array(estimated_velocities),
            'innovations': np.array(innovations) if innovations else np.array([]),
            'true_velocity': base_velocity,
            'motion_type': motion_type
        }
    
    def plot_results(self, results: dict, save_path: Optional[str] = None):
        """시뮬레이션 결과 시각화."""
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        # 1. 각도 추정
        ax = axes[0]
        ax.plot(results['time'], np.degrees(results['true_angles']), 'g-', 
                label='실제 각도', linewidth=2)
        ax.plot(results['time'], np.degrees(results['measured_angles']), 'k.', 
                label='측정값', markersize=1, alpha=0.3)
        ax.plot(results['time'], np.degrees(results['estimated_angles']), 'r-', 
                label='순환 혁신 칼만', linewidth=1.5)
        ax.set_ylabel('각도 (deg)')
        ax.set_title(f'각도 추정 (참값 vs 측정값 vs 칼만 추정): {results["motion_type"]} 운동')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-180 - 30, 180 + 30])
        
        # 2. 각속도 추정
        ax = axes[1]
        
        # 참값 계산 (시간 미분으로부터)
        if results['motion_type'] == 'constant':
            true_velocity_rpm = results['true_velocity'] * 30 / np.pi
            ax.axhline(y=true_velocity_rpm, color='g', linestyle='-', 
                       label=f'실제 각속도: {true_velocity_rpm:.1f} rpm', linewidth=2)
        elif results['motion_type'] == 'accelerating':
            # 가속 운동의 경우 시간에 따른 각속도 변화
            true_velocities = []
            dt = results['time'][1] - results['time'][0]
            for i in range(len(results['time'])):
                if i == 0:
                    true_velocities.append(0.2)  # 초기 각속도
                else:
                    # 각속도 = 초기속도 + 가속도 * 시간
                    true_velocities.append(0.2 + 0.1 * results['time'][i])
            true_velocities_rpm = np.array(true_velocities) * 30 / np.pi
            ax.plot(results['time'], true_velocities_rpm, 'g-', 
                    label='실제 각속도', linewidth=2)
        elif results['motion_type'] == 'sinusoidal':
            # 사인파 운동의 경우 코사인 미분
            freq = 0.2
            amplitude = 2.0
            true_velocities = amplitude * 2 * np.pi * freq * np.cos(2 * np.pi * freq * results['time'])
            true_velocities_rpm = true_velocities * 30 / np.pi
            ax.plot(results['time'], true_velocities_rpm, 'g-', 
                    label='실제 각속도', linewidth=2)
        
        estimated_velocities_rpm = results['estimated_velocities'] * 30 / np.pi
        ax.plot(results['time'], estimated_velocities_rpm, 'r-', 
                label='칼만 추정', linewidth=1.5)
        ax.set_ylabel('각속도 (rpm)')
        ax.set_title('각속도 추정 (참값 vs 칼만 추정)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 추정 오차
        ax = axes[2]
        errors = []
        for true, est in zip(results['true_angles'], results['estimated_angles']):
            errors.append(ang.diffpi(est, true))
        errors = np.array(errors)
        errors_deg = np.degrees(errors)
        
        ax.plot(results['time'], errors_deg, 'r-', linewidth=1.5)
        ax.fill_between(results['time'], errors_deg, alpha=0.3, color='red')
        ax.set_ylabel('오차 (deg)')
        ax.set_title(f'추정 오차 (RMSE: {np.sqrt(np.mean(errors**2)) * 180/np.pi:.2f} deg)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        # 4. 혁신 (Innovation)
        ax = axes[3]
        if len(results['innovations']) > 0:
            innovations_deg = np.degrees(results['innovations'])
            ax.plot(results['time'][1:], innovations_deg, 'b-', 
                    linewidth=1, alpha=0.7)
            ax.set_ylabel('혁신 (deg)')
            ax.set_title('순환 혁신 값 (측정값 - 예측값의 모듈러 차이)')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        ax.set_xlabel('시간 (초)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"그래프가 {save_path}에 저장되었습니다.")
        
        plt.show()
        
        return errors
    
    def print_metrics(self, errors: np.ndarray, results: dict):
        """성능 메트릭 출력."""
        print("\n" + "="*60)
        print("순환 혁신 칼만 필터 성능 분석")
        print("="*60)
        print(f"운동 타입: {results['motion_type']}")
        print(f"시뮬레이션 시간: {results['time'][-1]:.1f}초")
        print(f"\n[추정 정확도]")
        print(f"  RMSE: {np.sqrt(np.mean(errors**2)) * 180/np.pi:.3f} deg")
        print(f"  MAE:  {np.mean(np.abs(errors)) * 180/np.pi:.3f} deg")
        print(f"  최대 오차: {np.max(np.abs(errors)) * 180/np.pi:.3f} deg")
        
        if results['motion_type'] == 'constant':
            est_velocity_rpm = np.mean(results['estimated_velocities']) * 30 / np.pi
            true_velocity_rpm = results['true_velocity'] * 30 / np.pi
            error_rpm = abs(est_velocity_rpm - true_velocity_rpm)
            print(f"\n[각속도 추정]")
            print(f"  추정 평균: {est_velocity_rpm:.2f} rpm")
            print(f"  실제 값: {true_velocity_rpm:.2f} rpm")
            print(f"  오차: {error_rpm:.2f} rpm")


def main():
    """메인 실행 함수."""
    # 설정
    config = KalmanConfig(
        dt=0.01,
        process_noise=0.001,
        measurement_noise=0.1,
        initial_position_variance=1.0,
        initial_velocity_variance=0.1
    )
    
    # 객체 생성
    kalman_filter = CircularInnovationKalman(config)
    data_generator = AngleDataGenerator(dt=config.dt)
    analyzer = KalmanAnalyzer()
    
    print("순환 혁신 칼만 필터 시뮬레이션")
    print("="*60)
    
    # 1. 등속 운동
    print("\n1. 등속 운동 테스트")
    results = analyzer.run_simulation(
        kalman_filter=kalman_filter,
        data_generator=data_generator,
        duration=30.0,
        motion_type="constant",
        base_velocity=0.5,
        noise_std=0.1
    )
    errors = analyzer.plot_results(results)
    analyzer.print_metrics(errors, results)
    
    # 2. 가속 운동
    print("\n2. 가속 운동 테스트")
    results = analyzer.run_simulation(
        kalman_filter=kalman_filter,
        data_generator=data_generator,
        duration=20.0,
        motion_type="accelerating",
        base_velocity=0.2,
        noise_std=0.1
    )
    errors = analyzer.plot_results(results)
    analyzer.print_metrics(errors, results)
    
    # 3. 사인파 운동
    print("\n3. 사인파 운동 테스트")
    results = analyzer.run_simulation(
        kalman_filter=kalman_filter,
        data_generator=data_generator,
        duration=20.0,
        motion_type="sinusoidal",
        noise_std=0.05
    )
    errors = analyzer.plot_results(results)
    analyzer.print_metrics(errors, results)


if __name__ == "__main__":
    main()