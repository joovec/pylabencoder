# -*- coding: utf-8 -*-
"""
칼만 필터 성능 분석 메인 스크립트

CV, CA, CJ 칼만 필터와 측정값의 각도 오차 비교
사다리꼴 각속도 프로파일을 사용한 로봇 팔 엔코더 시나리오
"""

import numpy as np
import sys
import os

# 로컬 모듈 import
from kalman.cv import CVKalmanFilter, CVConfig
from kalman.ca import CAKalmanFilter, CAConfig  
from kalman.cj import CJKalmanFilter, CJConfig
from motion.motion import create_robot_arm_scenario
from visual import create_visualizer
from src.pytemplate.core import ang


def run_kalman_analysis():
    """칼만 필터 성능 분석 실행"""
    
    print("=" * 60)
    print("칼만 필터 성능 분석 - 사다리꼴 각속도 프로파일")
    print("=" * 60)
    
    # === 1. 로봇 팔 모션 시나리오 생성 ===
    print("\n1. 로봇 팔 모션 시나리오 생성...")
    
    robot_scenario = create_robot_arm_scenario(
        duration=12.0,           # 12초 동안 (정속 구간 늘림)
        max_velocity=2.0,        # 최대 2 rad/s (약 19 RPM)
        acceleration_time=2.0,   # 2초 가속/감속 (정속 구간 8초)
        noise_std=0.02,         # 측정 노이즈 0.02 rad (약 1.1°)
        quantization_bits=12     # 12비트 ADC
    )
    
    # 데이터 생성
    time, true_angles, measured_angles, true_velocities, true_accelerations = robot_scenario.generate_measurements(seed=42)
    
    # 모션 정보 출력
    motion_info = robot_scenario.get_profile_info()
    print("모션 프로파일 정보:")
    for key, value in motion_info.items():
        print(f"  • {key}: {value}")
    
    # === 2. 칼만 필터 설정 및 초기화 ===
    print("\n2. 칼만 필터 초기화...")
    
    dt = time[1] - time[0]
    
    # 업계 표준값 기반 설정 (광학 엔코더 시스템)
    # 조사 결과: Q가 R보다 훨씬 큰 값 사용 (Q=10000 수준)
    # 고정밀 엔코더는 측정값을 더 신뢰 -> Q/R 비율을 크게
    process_noise = 0.1      # 프로세스 노이즈 (증가)
    measurement_noise = 0.0001  # 측정 노이즈 분산 (감소 - 고정밀 엔코더)
    
    # 필터 설정
    cv_config = CVConfig(dt=dt, process_noise=process_noise, measurement_noise=measurement_noise)
    ca_config = CAConfig(dt=dt, process_noise=process_noise, measurement_noise=measurement_noise)
    cj_config = CJConfig(dt=dt, process_noise=process_noise, measurement_noise=measurement_noise)
    
    # 필터 인스턴스 생성
    cv_filter = CVKalmanFilter(cv_config)
    ca_filter = CAKalmanFilter(ca_config)
    cj_filter = CJKalmanFilter(cj_config)
    
    print(f"  • 샘플링 시간: {dt:.3f} s ({1/dt:.0f} Hz)")
    print(f"  • 프로세스 노이즈 Q: {process_noise:.1e}")
    print(f"  • 측정 노이즈 R: {measurement_noise:.1e}")
    print(f"  • 측정 노이즈 표준편차: {np.sqrt(measurement_noise):.4f} rad ({np.degrees(np.sqrt(measurement_noise)):.3f}°)")
    print(f"  • Q/R 비율: {process_noise/measurement_noise:.1e}")
    print(f"  • 참고: 업계 표준 - 고정밀 광학 엔코더는 Q/R > 1000 사용")
    
    # === 3. 칼만 필터 실행 ===
    print("\n3. 칼만 필터 실행...")
    
    n_samples = len(time)
    
    # 결과 저장 배열
    cv_angles = np.zeros(n_samples)
    ca_angles = np.zeros(n_samples)
    cj_angles = np.zeros(n_samples)
    
    cv_velocities = np.zeros(n_samples)
    ca_velocities = np.zeros(n_samples) 
    cj_velocities = np.zeros(n_samples)
    
    ca_accelerations = np.zeros(n_samples)
    cj_accelerations = np.zeros(n_samples)
    cj_jerks = np.zeros(n_samples)
    
    # 초기화
    initial_angle = measured_angles[0]
    cv_filter.reset(initial_angle)
    ca_filter.reset(initial_angle)
    cj_filter.reset(initial_angle)
    
    # 각 시점에서 필터 업데이트
    for i in range(n_samples):
        measured_angle = measured_angles[i]
        
        # CV 필터 (2차 상태: 각도, 각속도)
        cv_angle, cv_vel = cv_filter.update(measured_angle)
        cv_angles[i] = cv_angle
        cv_velocities[i] = cv_vel
        
        # CA 필터 (3차 상태: 각도, 각속도, 각가속도)
        ca_angle, ca_vel, ca_accel = ca_filter.update(measured_angle)
        ca_angles[i] = ca_angle
        ca_velocities[i] = ca_vel
        ca_accelerations[i] = ca_accel
        
        # CJ 필터 (4차 상태: 각도, 각속도, 각가속도, 저크)
        cj_angle, cj_vel, cj_accel, cj_jerk = cj_filter.update(measured_angle)
        cj_angles[i] = cj_angle
        cj_velocities[i] = cj_vel
        cj_accelerations[i] = cj_accel
        cj_jerks[i] = cj_jerk
    
    print(f"  • 총 {n_samples}개 샘플 처리 완료")
    
    # === 4. 성능 분석 ===
    print("\n4. 성능 분석...")
    
    # 오차 계산 (순환 차이)
    cv_errors = np.array([ang.diffpi(est, true) for est, true in zip(cv_angles, true_angles)])
    ca_errors = np.array([ang.diffpi(est, true) for est, true in zip(ca_angles, true_angles)])
    cj_errors = np.array([ang.diffpi(est, true) for est, true in zip(cj_angles, true_angles)])
    measured_errors = np.array([ang.diffpi(meas, true) for meas, true in zip(measured_angles, true_angles)])
    
    # 통계 계산
    def calculate_statistics(errors, name):
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(np.abs(errors))
        max_error = np.max(np.abs(errors))
        std_error = np.std(errors)
        
        print(f"  {name}:")
        print(f"    RMSE: {rmse*180/np.pi:.3f}° ({rmse:.6f} rad)")
        print(f"    MAE:  {mae*180/np.pi:.3f}° ({mae:.6f} rad)")  
        print(f"    Max:  {max_error*180/np.pi:.3f}° ({max_error:.6f} rad)")
        print(f"    STD:  {std_error*180/np.pi:.3f}° ({std_error:.6f} rad)")
        
        return rmse, mae, max_error, std_error
    
    measured_stats = calculate_statistics(measured_errors, "측정값 오차")
    cv_stats = calculate_statistics(cv_errors, "CV 칼만 오차")
    ca_stats = calculate_statistics(ca_errors, "CA 칼만 오차") 
    cj_stats = calculate_statistics(cj_errors, "CJ 칼만 오차")
    
    # 개선율 계산
    print(f"\n  오차 개선율 (vs 측정값):")
    cv_improvement = (measured_stats[0] - cv_stats[0])/measured_stats[0] * 100
    ca_improvement = (measured_stats[0] - ca_stats[0])/measured_stats[0] * 100
    cj_improvement = (measured_stats[0] - cj_stats[0])/measured_stats[0] * 100
    
    print(f"    CV 칼만: {cv_improvement:.1f}% {'개선' if cv_improvement > 0 else '악화'}")
    print(f"    CA 칼만: {ca_improvement:.1f}% {'개선' if ca_improvement > 0 else '악화'}") 
    print(f"    CJ 칼만: {cj_improvement:.1f}% {'개선' if cj_improvement > 0 else '악화'}")
    
    # === 5. 시각화 ===
    print("\n5. 결과 시각화...")
    
    visualizer = create_visualizer()
    
    # 필터 설정 정보 딕셔너리
    filter_configs = {
        'dt': dt,
        'process_noise': process_noise,
        'measurement_noise': measurement_noise,
        'q_r_ratio': process_noise / measurement_noise
    }
    
    # 메인 비교 그래프
    visualizer.plot_angle_comparison(
        time=time,
        true_angles=true_angles,
        measured_angles=measured_angles, 
        cv_angles=cv_angles,
        ca_angles=ca_angles,
        cj_angles=cj_angles,
        motion_info=motion_info,
        filter_configs=filter_configs
    )
    
    # 오차 분포 히스토그램
    visualizer.plot_error_statistics(cv_errors, ca_errors, cj_errors)
    
    print("  • 그래프 출력 완료")
    
    # === 6. 요약 ===
    print("\n" + "=" * 60)
    print("분석 결과 요약")
    print("=" * 60)
    
    # 성능 비교 (측정값 포함)
    filter_names = ['측정값', 'CV', 'CA', 'CJ']
    rmse_values = [measured_stats[0], cv_stats[0], ca_stats[0], cj_stats[0]]
    best_filter_idx = np.argmin(rmse_values)
    
    print(f"최적 성능: {filter_names[best_filter_idx]} (RMSE: {rmse_values[best_filter_idx]*180/np.pi:.2f}°)")
    
    # 칼만 필터만의 최적 성능
    kalman_rmse = [cv_stats[0], ca_stats[0], cj_stats[0]]
    best_kalman_idx = np.argmin(kalman_rmse)
    print(f"칼만 필터 중 최적: {['CV', 'CA', 'CJ'][best_kalman_idx]} 필터")
    
    if cj_improvement > 0 and ca_improvement > 0:
        print(f"사다리꼴 각속도 프로파일에서 CA, CJ 칼만 필터가 측정 노이즈를 효과적으로 감소시킴")
    else:
        print(f"현재 설정에서 일부 칼만 필터의 추가 튜닝이 필요할 수 있음")
    
    print("순환 혁신 방식으로 각도 오버플로 문제 없이 안정적 추정")


if __name__ == "__main__":
    try:
        run_kalman_analysis()
        
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n분석 완료.")