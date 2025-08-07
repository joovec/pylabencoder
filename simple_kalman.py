import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter




from src.pytemplate.core import ang
from src.pytemplate.core.anal import linear_error_analysis
from src.pytemplate.utils.csv_reader import load_csv_data, list_csv_files
from src.pytemplate.utils.plot_helper import setup_korean_font, plot_timeseries, plot_arrays, plot_histograms


def gendata( timearray, vel=1.0, xpos_init=0.0, noise_std=0.0):
    """가상의 센서 데이터 생성."""

    xpos= vel * timearray + xpos_init  # 속도에 따라 위치 계산
    
    # 가우시안 노이즈 추가
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, size=xpos.shape)
        xpos = xpos + noise

    return xpos


def gendata_sine( timearray, amplitude=5.0, frequency=0.5, phase=0.0, noise_std=0.0):
    """사인파 운동 데이터 생성."""
    
    xpos = amplitude * np.sin(2 * np.pi * frequency * timearray + phase)
    
    # 가우시안 노이즈 추가
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, size=xpos.shape)
        xpos = xpos + noise

    return xpos


def velocity_from_diff(positions, dt):
    """차분을 이용한 속도 계산"""
    velocities = np.zeros_like(positions)
    velocities[1:] = (positions[1:] - positions[:-1]) / dt
    velocities[0] = velocities[1]  # 첫 번째 값은 두 번째 값으로 설정
    return velocities


def calculate_mse(estimated, true_values):
    """MSE (Mean Square Error) 계산"""
    return np.mean((estimated - true_values) ** 2)


def find_optimal_rq_combination(measurements, true_positions, dt=0.001):
    """최적의 R, Q 조합 찾기"""
    
    # 테스트할 R, Q 값들
    r_values = np.logspace(-2, 1, 10)  # 0.01 ~ 10
    q_values = np.logspace(-3, 0, 10)  # 0.001 ~ 1
    
    best_mse = float('inf')
    best_r = None
    best_q = None
    results = []
    
    print("R, Q 조합 테스트 중...")
    
    for r in r_values:
        for q in q_values:
            # 칼만 필터 적용
            filtered_pos, _ = apply_kalman_filter(measurements, dt=dt, 
                                                process_noise=q, measurement_noise=r)
            
            # MSE 계산
            mse = calculate_mse(filtered_pos, true_positions)
            results.append((r, q, mse))
            
            # 최적값 업데이트
            if mse < best_mse:
                best_mse = mse
                best_r = r
                best_q = q
    
    print(f"최적 조합: R={best_r:.4f}, Q={best_q:.4f}, MSE={best_mse:.6f}")
    
    return best_r, best_q, best_mse, results


def apply_kalman_filter(measurements, dt=0.01, process_noise=0.01, measurement_noise=0.2):
    """선형 칼만 필터 적용 - 등속 운동 모델"""
    
    # 칼만 필터 초기화
    kf = KalmanFilter(dim_x=2, dim_z=1)
    
    # 상태 전이 행렬 (위치, 속도)
    kf.F = np.array([[1, dt],
                     [0, 1]])
    
    # 측정 행렬 (위치만 측정)
    kf.H = np.array([[1, 0]])
    
    # 프로세스 노이즈
    kf.Q = np.array([[dt**3/3, dt**2/2],
                     [dt**2/2, dt]]) * process_noise
    
    # 측정 노이즈
    kf.R = measurement_noise ** 2
    
    # 초기 상태
    kf.x = np.array([measurements[0], 0])
    
    # 초기 공분산
    kf.P = np.eye(2) * 10
    
    # 필터링 수행
    filtered_positions = []
    filtered_velocities = []
    
    for z in measurements:
        kf.predict()
        kf.update(z)
        filtered_positions.append(kf.x[0])
        filtered_velocities.append(kf.x[1])


    print("칼만 필터링 완료")
    print(f"측정 노이즈 공분산 (R): {kf.R}")
    print(f"프로세스 노이즈 공분산 (Q):\n{kf.Q}")
    print(f"상태 공분산 (P):\n{kf.P}")
    
    return np.array(filtered_positions), np.array(filtered_velocities)




if __name__ == "__main__":

    t=np.arange(0, 10, 0.001)  # 시간 배열
    
    # 파형 발생기 파라미터
    amplitude = 5.0
    frequency = 1
    noise_std = 0.1
    
    # 칼만 필터 파라미터
    process_noise_low = 0.01
    process_noise_med = 0.1
    process_noise_high = 1.0
    
    measurement_noise_low = 0.1
    measurement_noise_med = 0.5
    measurement_noise_high = 1.0
    
    # 신호 생성
    xpos = gendata_sine(t, amplitude=amplitude, frequency=frequency, noise_std=noise_std)
    
    # 실제(참) 위치
    true_pos = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # 최적의 R, Q 조합 찾기
    dt_val = t[1] - t[0]  # 실제 dt 값
    best_r, best_q, best_mse, all_results = find_optimal_rq_combination(xpos, true_pos, dt=dt_val)
    
    # 최적 조합으로 필터링
    optimal_pos, optimal_vel = apply_kalman_filter(xpos, dt=dt_val, 
                                                 process_noise=best_q, measurement_noise=best_r)

    # measurement noise 비교용 (process noise 고정)
    pos_meas_low, vel_meas_low = apply_kalman_filter(xpos, dt=dt_val, process_noise=0.01, measurement_noise=measurement_noise_low)
    pos_meas_med, vel_meas_med = apply_kalman_filter(xpos, dt=dt_val, process_noise=0.01, measurement_noise=measurement_noise_med)
    pos_meas_high, vel_meas_high = apply_kalman_filter(xpos, dt=dt_val, process_noise=0.01, measurement_noise=measurement_noise_high)
    
    # process noise 비교용 (measurement noise 고정)
    pos_proc_low, vel_proc_low = apply_kalman_filter(xpos, dt=dt_val, process_noise=process_noise_low, measurement_noise=0.2)
    pos_proc_med, vel_proc_med = apply_kalman_filter(xpos, dt=dt_val, process_noise=process_noise_med, measurement_noise=0.2)
    pos_proc_high, vel_proc_high = apply_kalman_filter(xpos, dt=dt_val, process_noise=process_noise_high, measurement_noise=0.2)
    
    # 차분을 이용한 속도 계산
    diff_vel = velocity_from_diff(xpos, dt=0.01)

   
    
    
    # 플롯 설정
    setup_korean_font()
    
    # 결과 플롯 - 최적 조합과 비교
    plt.figure(figsize=(16, 12))
    
    # 최적 조합 결과
    plt.subplot(2, 2, 1)
    plt.plot(t, xpos, 'b.', alpha=0.2, markersize=0.5, label='Measured')
    plt.plot(t, true_pos, 'r--', linewidth=2, label='True')
    plt.plot(t, optimal_pos, 'g-', linewidth=3, label=f'Optimal (R={best_r:.4f}, Q={best_q:.4f})')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title(f'최적 R,Q 조합 결과 (MSE={best_mse:.6f})')
    plt.legend(fontsize=8)
    plt.grid(True)
    
    # MSE 히트맵
    plt.subplot(2, 2, 2)
    r_vals = np.logspace(-2, 1, 10)
    q_vals = np.logspace(-3, 0, 10)
    mse_grid = np.zeros((len(q_vals), len(r_vals)))
    
    for i, (r, q, mse) in enumerate(all_results):
        r_idx = np.argmin(np.abs(r_vals - r))
        q_idx = np.argmin(np.abs(q_vals - q))
        mse_grid[q_idx, r_idx] = mse
    
    im = plt.imshow(mse_grid, extent=[np.log10(r_vals[0]), np.log10(r_vals[-1]), 
                                     np.log10(q_vals[0]), np.log10(q_vals[-1])], 
                    aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(im, label='MSE')
    plt.scatter(np.log10(best_r), np.log10(best_q), color='red', s=100, marker='x', linewidth=3)
    plt.xlabel('log10(R) - Measurement Noise')
    plt.ylabel('log10(Q) - Process Noise')
    plt.title('MSE 히트맵 (빨간 X = 최적점)')
    
    # 위치 비교 - 다양한 조합
    plt.subplot(2, 2, 3)
    plt.plot(t, true_pos, 'r--', linewidth=2, label='True')
    plt.plot(t, optimal_pos, 'g-', linewidth=3, label=f'Optimal')
    plt.plot(t, pos_meas_low, 'b-', linewidth=1, alpha=0.7, label=f'R={measurement_noise_low}, Q=0.01')
    plt.plot(t, pos_proc_med, 'm-', linewidth=1, alpha=0.7, label=f'R=0.2, Q={process_noise_med}')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('위치 추적 비교')
    plt.legend(fontsize=8)
    plt.grid(True)
    
    # 오차 분석
    plt.subplot(2, 2, 4)
    error_optimal = np.abs(optimal_pos - true_pos)
    error_meas_low = np.abs(pos_meas_low - true_pos)
    error_proc_med = np.abs(pos_proc_med - true_pos)
    
    plt.plot(t, error_optimal, 'g-', linewidth=2, label=f'Optimal (MSE={calculate_mse(optimal_pos, true_pos):.6f})')
    plt.plot(t, error_meas_low, 'b-', linewidth=1, label=f'R={measurement_noise_low} (MSE={calculate_mse(pos_meas_low, true_pos):.6f})')
    plt.plot(t, error_proc_med, 'm-', linewidth=1, label=f'Q={process_noise_med} (MSE={calculate_mse(pos_proc_med, true_pos):.6f})')
    plt.xlabel('Time (s)')
    plt.ylabel('Absolute Error')
    plt.title('절대 오차 비교')
    plt.legend(fontsize=8)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
