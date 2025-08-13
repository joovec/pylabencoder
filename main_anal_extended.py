# -*- coding: utf-8 -*-
"""
확장된 칼만 필터 및 다양한 필터 성능 분석 메인 스크립트

CV, CA, CJ 칼만 필터 + IIR, EMA 필터의 성능 비교
7가지 로봇 운동 시나리오별 개별 분석
"""

import numpy as np
import sys
import os

# 로컬 모듈 import
from kalman.cv import CVKalmanFilter, CVConfig
from kalman.ca import CAKalmanFilter, CAConfig  
from kalman.cj import CJKalmanFilter, CJConfig
from kalman.iir import IIRFilter, IIRConfig
from kalman.ema import EMAFilter, EMAConfig
from kalman.fir import FIRFilter, FIRConfig
from motion.motion import MotionType, create_robot_motion, demo_all_motions
from visual import create_visualizer
from src.pytemplate.core import ang


def run_single_motion_analysis(motion_type: MotionType):
    """단일 모션에 대한 모든 필터 분석"""
    
    print(f"\n{'='*60}")
    print(f"{motion_type.value.upper()} 모션 필터 성능 분석")
    print(f"{'='*60}")
    
    # === 1. 모션 시나리오 생성 ===
    print(f"\n1. {motion_type.value} 모션 시나리오 생성...")
    
    # 모션별 특화 파라미터
    if motion_type == MotionType.SINUSOIDAL:
        robot = create_robot_motion(motion_type, duration=12.0, frequency=0.3, max_velocity=1.5, noise_std=0.02)
    elif motion_type == MotionType.STEP:
        robot = create_robot_motion(motion_type, duration=12.0, step_time=4.0, max_velocity=2.0, noise_std=0.02)
    elif motion_type == MotionType.SCURVE:
        robot = create_robot_motion(motion_type, duration=12.0, acceleration_time=3.0, max_velocity=1.8, noise_std=0.02)
    else:
        robot = create_robot_motion(motion_type, duration=12.0, max_velocity=2.0, acceleration_time=2.0, noise_std=0.02)
    
    # 데이터 생성
    time, true_angles, measured_angles, true_velocities, true_accelerations = robot.generate_measurements(seed=42)
    
    # 모션 정보 출력
    motion_info = robot.get_profile_info()
    print("모션 프로파일 정보:")
    for key, value in motion_info.items():
        print(f"  • {key}: {value}")
    
    # === 2. 필터 설정 및 초기화 ===
    print("\n2. 필터 초기화...")
    
    dt = time[1] - time[0]
    
    # 업계 표준 칼만 필터 설정
    process_noise = 0.1      
    measurement_noise = 0.0001
    
    # 칼만 필터 설정
    cv_config = CVConfig(dt=dt, process_noise=process_noise, measurement_noise=measurement_noise)
    ca_config = CAConfig(dt=dt, process_noise=process_noise, measurement_noise=measurement_noise)
    cj_config = CJConfig(dt=dt, process_noise=process_noise, measurement_noise=measurement_noise)
    
    # IIR/EMA/FIR 필터 설정
    iir_config = IIRConfig(alpha=0.1, dt=dt)  # 적당한 알파값
    ema_config = EMAConfig(alpha=0.2, adaptive=False)  # 조금 더 빠른 응답
    fir_config = FIRConfig(window_size=10, filter_type="moving_average", dt=dt)  # 이동평균 필터
    
    # 필터 인스턴스 생성
    cv_filter = CVKalmanFilter(cv_config)
    ca_filter = CAKalmanFilter(ca_config)
    cj_filter = CJKalmanFilter(cj_config)
    iir_filter = IIRFilter(iir_config)
    ema_filter = EMAFilter(ema_config)
    fir_filter = FIRFilter(fir_config)
    
    print(f"  • 샘플링 시간: {dt:.3f} s ({1/dt:.0f} Hz)")
    print(f"  • 프로세스 노이즈 Q: {process_noise:.1e}")
    print(f"  • 측정 노이즈 R: {measurement_noise:.1e}")
    print(f"  • Q/R 비율: {process_noise/measurement_noise:.1e}")
    print(f"  • IIR 알파: {iir_config.alpha:.2f}")
    print(f"  • EMA 알파: {ema_config.alpha:.2f}")
    print(f"  • FIR 윈도우: {fir_config.window_size} ({fir_config.filter_type})")
    
    # === 3. 필터 실행 ===
    print("\n3. 모든 필터 실행...")
    
    n_samples = len(time)
    
    # 결과 저장 배열
    cv_angles = np.zeros(n_samples)
    ca_angles = np.zeros(n_samples)
    cj_angles = np.zeros(n_samples)
    iir_angles = np.zeros(n_samples)
    ema_angles = np.zeros(n_samples)
    fir_angles = np.zeros(n_samples)
    
    # 초기화
    initial_angle = measured_angles[0]
    cv_filter.reset(initial_angle)
    ca_filter.reset(initial_angle)
    cj_filter.reset(initial_angle)
    iir_filter.reset(initial_angle)
    ema_filter.reset(initial_angle)
    fir_filter.reset(initial_angle)
    
    # 각 시점에서 필터 업데이트
    for i in range(n_samples):
        measured_angle = measured_angles[i]
        
        # 칼만 필터들
        cv_angle, _ = cv_filter.update(measured_angle)
        ca_angle, _, _ = ca_filter.update(measured_angle)
        cj_angle, _, _, _ = cj_filter.update(measured_angle)
        
        # 기타 필터들
        iir_angle = iir_filter.update(measured_angle)
        ema_angle = ema_filter.update(measured_angle)
        fir_angle = fir_filter.update(measured_angle)
        
        # 결과 저장
        cv_angles[i] = cv_angle
        ca_angles[i] = ca_angle
        cj_angles[i] = cj_angle
        iir_angles[i] = iir_angle
        ema_angles[i] = ema_angle
        fir_angles[i] = fir_angle
    
    print(f"  • 총 {n_samples}개 샘플 처리 완료")
    
    # === 4. 성능 분석 ===
    print("\n4. 성능 분석...")
    
    # 오차 계산 (순환 차이)
    measured_errors = np.array([ang.diffpi(meas, true) for meas, true in zip(measured_angles, true_angles)])
    cv_errors = np.array([ang.diffpi(est, true) for est, true in zip(cv_angles, true_angles)])
    ca_errors = np.array([ang.diffpi(est, true) for est, true in zip(ca_angles, true_angles)])
    cj_errors = np.array([ang.diffpi(est, true) for est, true in zip(cj_angles, true_angles)])
    iir_errors = np.array([ang.diffpi(est, true) for est, true in zip(iir_angles, true_angles)])
    ema_errors = np.array([ang.diffpi(est, true) for est, true in zip(ema_angles, true_angles)])
    fir_errors = np.array([ang.diffpi(est, true) for est, true in zip(fir_angles, true_angles)])
    
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
    iir_stats = calculate_statistics(iir_errors, "IIR 필터 오차")
    ema_stats = calculate_statistics(ema_errors, "EMA 필터 오차")
    fir_stats = calculate_statistics(fir_errors, "FIR 필터 오차")
    
    # 개선율 계산
    print(f"\n  오차 개선율 (vs 측정값):")
    for name, stats in [('CV 칼만', cv_stats), ('CA 칼만', ca_stats), ('CJ 칼만', cj_stats), 
                        ('IIR 필터', iir_stats), ('EMA 필터', ema_stats), ('FIR 필터', fir_stats)]:
        improvement = (measured_stats[0] - stats[0])/measured_stats[0] * 100
        print(f"    {name}: {improvement:.1f}% {'개선' if improvement > 0 else '악화'}")
    
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
    
    # 필터 결과 딕셔너리
    filter_results = {
        'cv': cv_angles,
        'ca': ca_angles,
        'cj': cj_angles,
        'iir': iir_angles,
        'ema': ema_angles,
        'fir': fir_angles
    }
    
    # 통합 비교 그래프
    visualizer.plot_all_filters_comparison(
        time=time,
        true_angles=true_angles,
        measured_angles=measured_angles,
        filter_results=filter_results,
        motion_info=motion_info,
        filter_configs=filter_configs
    )
    
    # 히스토그램 출력 제거됨
    
    print("  • 그래프 출력 완료")
    
    # === 6. 요약 ===
    print(f"\n{'='*60}")
    print("분석 결과 요약")
    print(f"{'='*60}")
    
    # 전체 성능 비교 (측정값 포함)
    all_names = ['측정값', 'CV', 'CA', 'CJ', 'IIR', 'EMA', 'FIR']
    all_rmse = [measured_stats[0], cv_stats[0], ca_stats[0], cj_stats[0], iir_stats[0], ema_stats[0], fir_stats[0]]
    best_idx = np.argmin(all_rmse)
    
    print(f"전체 최적 성능: {all_names[best_idx]} (RMSE: {all_rmse[best_idx]*180/np.pi:.2f}°)")
    
    # 칼만 필터 중 최적
    kalman_rmse = [cv_stats[0], ca_stats[0], cj_stats[0]]
    best_kalman_idx = np.argmin(kalman_rmse)
    print(f"칼만 필터 중 최적: {['CV', 'CA', 'CJ'][best_kalman_idx]} 필터")
    
    # 기타 필터 중 최적
    other_rmse = [iir_stats[0], ema_stats[0], fir_stats[0]]
    best_other_idx = np.argmin(other_rmse)
    print(f"기타 필터 중 최적: {['IIR', 'EMA', 'FIR'][best_other_idx]} 필터")
    
    print(f"{motion_type.value} 모션에서의 필터 성능이 분석되었습니다.")
    
    return {
        'motion_type': motion_type,
        'motion_info': motion_info,
        'time': time,
        'true_angles': true_angles,
        'measured_angles': measured_angles,
        'filter_results': filter_results,
        'filter_errors': {
            'cv': cv_errors,
            'ca': ca_errors,
            'cj': cj_errors,
            'iir': iir_errors,
            'ema': ema_errors,
            'fir': fir_errors
        },
        'measured_errors': measured_errors,
        'stats': {
            'measured': measured_stats,
            'cv': cv_stats,
            'ca': ca_stats,
            'cj': cj_stats,
            'iir': iir_stats,
            'ema': ema_stats,
            'fir': fir_stats
        }
    }


def run_all_motions_analysis():
    """모든 모션에 대한 종합 분석"""
    
    print("="*60)
    print("7가지 로봇 모션 시나리오 - 필터 성능 종합 분석")
    print("="*60)
    
    motion_types = list(MotionType)
    all_results = {}
    
    visualizer = create_visualizer()
    
    for motion_type in motion_types:
        print(f"\n처리 중: {motion_type.value}...")
        
        # 단일 모션 분석 실행
        result = run_single_motion_analysis(motion_type)
        all_results[motion_type] = result
        
        # 모션별 개별 그래프 생성 (올바른 데이터 전달)
        motion_data = {
            'time': result['time'],  # 실제 시간 배열
            'true_angles': result['true_angles'],  # 실제 각도 배열
            'measured_angles': result['measured_angles'],  # 측정 각도 배열
            'info': result['motion_info']
        }
        
        # 개별 모션 그래프는 주석 처리 (너무 많은 그래프 출력 방지)
        # visualizer.plot_motion_specific_analysis(motion_data, result['filter_results'])
    
    print(f"\n{'='*60}")
    print("종합 분석 완료")
    print(f"{'='*60}")
    print("모든 모션 시나리오에 대한 분석이 완료되었습니다.")
    print("각 모션별로 그래프와 성능 지표가 출력되었습니다.")
    
    return all_results


if __name__ == "__main__":
    try:
        # 특정 모션 하나만 테스트하려면:
        # run_single_motion_analysis(MotionType.TRAPEZOIDAL)
        
        # 모든 모션 분석:
        run_all_motions_analysis()
        
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n분석 완료.")