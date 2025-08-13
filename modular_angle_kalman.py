# -*- coding: utf-8 -*-
"""
모듈러 각도 특성을 고려한 칼만 필터 구현.

filterpy를 사용하여 각도 추정을 수행하며, 
각도의 wrap-around 특성을 올바르게 처리합니다.
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
    initial_velocity_variance: float = 1.0  # 초기 속도 분산


class AngleDataGenerator:
    """각도 데이터 생성을 담당하는 클래스."""
    
    def __init__(self, dt: float = 0.01):
        """
        Args:
            dt: 샘플링 시간 간격
        """
        self.dt = dt
    
    def generate_constant_velocity(
        self, 
        velocity: float, 
        duration: float, 
        initial_angle: float = 0.0,
        noise_std: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        등속 원운동 데이터를 생성합니다.
        
        Args:
            velocity: 각속도 (rad/s)
            duration: 시뮬레이션 시간 (초)
            initial_angle: 초기 각도 (rad)
            noise_std: 측정 노이즈 표준편차
            
        Returns:
            time_array: 시간 배열
            true_angles: 실제 각도 배열 (wrap된 상태)
            measured_angles: 노이즈가 포함된 측정 각도 배열
        """
        num_samples = int(duration / self.dt)
        time_array = np.linspace(0, duration, num_samples)
        
        # 실제 각도 (누적)
        true_angles_unwrapped = initial_angle + velocity * time_array
        
        # -π ~ π 범위로 정규화
        true_angles = np.array([ang.wrap_pi(angle) for angle in true_angles_unwrapped])
        
        # 측정 노이즈 추가
        noise = np.random.normal(0, noise_std, num_samples)
        measured_angles_noisy = true_angles_unwrapped + noise
        measured_angles = np.array([ang.wrap_pi(angle) for angle in measured_angles_noisy])
        
        return time_array, true_angles, measured_angles


class ModularAngleKalmanFilter:
    """순환 혁신(Circular Innovation) 방식의 칼만 필터.
    
    각도의 모듈러 특성을 올바르게 처리하기 위해 
    언래핑 대신 순환 혁신을 사용합니다.
    """
    
    def __init__(self, config: KalmanConfig):
        """
        Args:
            config: 칼만 필터 설정
        """
        self.config = config
        self.kf = self._initialize_filter()
        
    def _initialize_filter(self) -> KalmanFilter:
        """filterpy 칼만 필터를 초기화합니다."""
        kf = KalmanFilter(dim_x=2, dim_z=1)
        
        # 상태 전이 행렬 (등속 모델)
        kf.F = np.array([
            [1, self.config.dt],
            [0, 1]
        ])
        
        # 측정 행렬 (각도만 측정)
        kf.H = np.array([[1, 0]])
        
        # 프로세스 노이즈 공분산
        q = self.config.process_noise
        kf.Q = np.array([
            [self.config.dt**4/4, self.config.dt**3/2],
            [self.config.dt**3/2, self.config.dt**2]
        ]) * q
        
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
        
        Args:
            measured_angle: 측정된 각도 (rad, -π ~ π)
            
        Returns:
            estimated_angle: 추정된 각도
            estimated_velocity: 추정된 각속도
        """
        # 예측 단계
        self.kf.predict()
        
        # 순환 혁신 계산 (핵심: 모듈러 공간에서 차이 계산)
        predicted_angle = ang.wrap_pi(self.kf.x[0, 0])
        innovation = ang.diffpi(measured_angle, predicted_angle)
        
        # 칼만 게인 계산
        S = self.kf.H @ self.kf.P @ self.kf.H.T + self.kf.R
        K = self.kf.P @ self.kf.H.T / S[0, 0]
        
        # 상태 업데이트 (혁신만큼 업데이트)
        self.kf.x = self.kf.x + K * innovation
        
        # 공분산 업데이트
        self.kf.P = (np.eye(2) - K @ self.kf.H) @ self.kf.P
        
        # 추정값 반환 (각도는 wrap하여 반환)
        estimated_angle = ang.wrap_pi(self.kf.x[0, 0])
        estimated_velocity = self.kf.x[1, 0]
        
        return estimated_angle, estimated_velocity
    
    def reset(self, initial_angle: float = 0.0, initial_velocity: float = 0.0):
        """필터를 초기 상태로 리셋합니다."""
        self.kf.x = np.array([[initial_angle], [initial_velocity]])


class KalmanFilterAnalyzer:
    """칼만 필터 성능 분석 및 시각화."""
    
    def __init__(self):
        """분석기 초기화."""
        setup_korean_font()
        
    def run_simulation(
        self,
        kalman_filter: ModularAngleKalmanFilter,
        data_generator: AngleDataGenerator,
        velocity: float = 0.5,
        duration: float = 10.0,
        noise_std: float = 0.1
    ) -> dict:
        """
        시뮬레이션을 실행하고 결과를 반환합니다.
        
        Args:
            kalman_filter: 칼만 필터 인스턴스
            data_generator: 데이터 생성기
            velocity: 각속도 (rad/s)
            duration: 시뮬레이션 시간 (초)
            noise_std: 측정 노이즈 표준편차
            
        Returns:
            시뮬레이션 결과 딕셔너리
        """
        # 데이터 생성
        time_array, true_angles, measured_angles = data_generator.generate_constant_velocity(
            velocity=velocity,
            duration=duration,
            noise_std=noise_std
        )
        
        # 칼만 필터 실행
        estimated_angles = []
        estimated_velocities = []
        
        kalman_filter.reset(initial_angle=measured_angles[0])
        
        for measured_angle in measured_angles:
            est_angle, est_velocity = kalman_filter.update(measured_angle)
            estimated_angles.append(est_angle)
            estimated_velocities.append(est_velocity)
        
        return {
            'time': time_array,
            'true_angles': true_angles,
            'measured_angles': measured_angles,
            'estimated_angles': np.array(estimated_angles),
            'estimated_velocities': np.array(estimated_velocities),
            'true_velocity': velocity
        }
    
    def plot_results(self, results: dict, save_path: Optional[str] = None):
        """
        시뮬레이션 결과를 시각화합니다.
        
        Args:
            results: 시뮬레이션 결과
            save_path: 그래프 저장 경로 (선택사항)
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 1. 각도 비교
        ax = axes[0]
        ax.plot(results['time'], results['true_angles'], 'g-', 
                label='실제 각도', linewidth=2)
        ax.plot(results['time'], results['measured_angles'], 'b.', 
                label='측정 각도', markersize=2, alpha=0.5)
        ax.plot(results['time'], results['estimated_angles'], 'r-', 
                label='칼만 필터 추정', linewidth=1.5)
        ax.set_ylabel('각도 (rad)')
        ax.set_title('각도 추정 비교 (등속 운동)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-np.pi - 0.5, np.pi + 0.5])
        
        # 2. 각속도 추정
        ax = axes[1]
        ax.axhline(y=results['true_velocity'], color='g', linestyle='--', 
                   label=f'실제 각속도: {results["true_velocity"]:.3f} rad/s')
        ax.plot(results['time'], results['estimated_velocities'], 'r-', 
                label='추정 각속도', linewidth=1.5)
        ax.set_ylabel('각속도 (rad/s)')
        ax.set_title('각속도 추정')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 추정 오차
        ax = axes[2]
        errors = []
        for true, est in zip(results['true_angles'], results['estimated_angles']):
            errors.append(ang.diffpi(est, true))
        errors = np.array(errors)
        
        ax.plot(results['time'], errors, 'r-', linewidth=1.5)
        ax.fill_between(results['time'], errors, alpha=0.3, color='red')
        ax.set_xlabel('시간 (초)')
        ax.set_ylabel('추정 오차 (rad)')
        ax.set_title(f'추정 오차 (RMSE: {np.sqrt(np.mean(errors**2)):.4f} rad)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"그래프가 {save_path}에 저장되었습니다.")
        
        plt.show()
        
        return errors


def main():
    """메인 실행 함수.
    
    순환 혁신 방식의 칼만 필터를 사용하여
    각도 추정을 수행합니다.
    """
    # 설정
    config = KalmanConfig(
        dt=0.01,
        process_noise=0.001,
        measurement_noise=0.1,
        initial_position_variance=1.0,
        initial_velocity_variance=0.1
    )
    
    # 객체 생성
    data_generator = AngleDataGenerator(dt=config.dt)
    kalman_filter = ModularAngleKalmanFilter(config)
    analyzer = KalmanFilterAnalyzer()
    
    # 시뮬레이션 실행
    print("칼만 필터 시뮬레이션 시작...")
    results = analyzer.run_simulation(
        kalman_filter=kalman_filter,
        data_generator=data_generator,
        velocity=0.5,  # rad/s
        duration=20.0,  # seconds
        noise_std=0.1
    )
    
    # 결과 시각화
    errors = analyzer.plot_results(results)
    
    # 성능 메트릭 출력
    print("\n=== 성능 분석 ===")
    print(f"RMSE: {np.sqrt(np.mean(errors**2)):.6f} rad")
    print(f"MAE: {np.mean(np.abs(errors)):.6f} rad")
    print(f"최대 오차: {np.max(np.abs(errors)):.6f} rad")
    print(f"추정 각속도 평균: {np.mean(results['estimated_velocities']):.6f} rad/s")
    print(f"실제 각속도: {results['true_velocity']:.6f} rad/s")


if __name__ == "__main__":
    main()