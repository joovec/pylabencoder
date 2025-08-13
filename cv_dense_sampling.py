import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.family'] = 'Malgun Gothic'

def wrap_2pi(angle):
    """입력 각도를 [0, 2pi) 범위로 래핑"""
    return np.mod(angle, 2*np.pi)

def generate_trapezoid_profile(dt, total_time, rpm_max=1000, accel_time=0.5):
    """사다리꼴 속도 프로파일 생성"""
    t = np.arange(0, total_time, dt)
    omega_max = rpm_max * 2 * np.pi / 60
    
    omega = np.zeros_like(t)
    alpha = np.zeros_like(t)
    
    accel_rate = omega_max / accel_time
    
    for i, ti in enumerate(t):
        if ti <= accel_time:
            omega[i] = accel_rate * ti
            alpha[i] = accel_rate
        elif ti <= total_time - accel_time:
            omega[i] = omega_max
            alpha[i] = 0
        else:
            t_decel = ti - (total_time - accel_time)
            omega[i] = omega_max - accel_rate * t_decel
            alpha[i] = -accel_rate
            if omega[i] < 0:
                omega[i] = 0
                alpha[i] = 0
    
    theta = np.cumsum(omega) * dt
    return t, theta, omega, alpha

def run_cv_filter(theta_meas, omega_init, dt, q_value, noise_std):
    """CV 칼만필터 실행"""
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.F = np.array([[1.0, dt], [0.0, 1.0]])
    kf.H = np.array([[1.0, 0.0]])
    kf.Q = q_value * np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]])
    kf.R = np.array([[noise_std**2]])
    kf.P = np.diag([noise_std**2, 0.1])
    kf.x = np.array([[theta_meas[0]], [omega_init]])
    
    theta_est = np.zeros_like(theta_meas)
    omega_est = np.zeros_like(theta_meas)
    
    for i in range(len(theta_meas)):
        if i > 0:
            kf.predict()
        kf.update(np.array([[theta_meas[i]]]))
        theta_est[i] = kf.x[0, 0]
        omega_est[i] = kf.x[1, 0]
    
    return theta_est, omega_est

def main():
    # 매우 촘촘한 샘플링 시간 테스트
    dt_values = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005]  # 10kHz, 5kHz, 2kHz, 1kHz, 500Hz, 200Hz
    q_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    
    total_time = 2.0
    noise_std = np.deg2rad(0.5)
    
    # 결과 저장
    results_matrix = np.zeros((len(dt_values), len(q_values)))
    
    # 각 dt에 대해 테스트
    for dt_idx, dt in enumerate(dt_values):
        print(f"\n샘플링: dt={dt*1000:.2f}ms ({1/dt:.0f}Hz)")
        
        # 프로파일 생성
        t, theta_true, omega_true, alpha_true = generate_trapezoid_profile(
            dt, total_time, rpm_max=1000, accel_time=0.4
        )
        
        # 측정값 생성
        np.random.seed(42)
        theta_meas = theta_true + np.random.normal(0, noise_std, len(theta_true))
        
        # 각 Q 값에 대해 테스트
        for q_idx, q in enumerate(q_values):
            theta_est, omega_est = run_cv_filter(theta_meas, omega_true[0], dt, q, noise_std)
            err = theta_est - theta_true
            rms = np.degrees(np.sqrt(np.mean(err**2)))
            results_matrix[dt_idx, q_idx] = rms
            
            print(f"  Q={q:.1e}: RMS={rms:.3f}°")
    
    # 히트맵 플롯
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. RMS 에러 히트맵
    im = axes[0, 0].imshow(results_matrix, aspect='auto', cmap='RdYlGn_r', 
                          vmin=0, vmax=5, interpolation='nearest')
    axes[0, 0].set_xticks(range(len(q_values)))
    axes[0, 0].set_xticklabels([f'{q:.0e}' for q in q_values], rotation=45)
    axes[0, 0].set_yticks(range(len(dt_values)))
    axes[0, 0].set_yticklabels([f'{dt*1000:.2f}ms\n({1/dt:.0f}Hz)' for dt in dt_values])
    axes[0, 0].set_xlabel('Q value')
    axes[0, 0].set_ylabel('Sampling Time')
    axes[0, 0].set_title('RMS Error (degrees)')
    plt.colorbar(im, ax=axes[0, 0])
    
    # 값 표시
    for i in range(len(dt_values)):
        for j in range(len(q_values)):
            text = axes[0, 0].text(j, i, f'{results_matrix[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=8)
    
    # 2. 최적 Q에 대한 RMS vs dt
    best_rms = np.min(results_matrix, axis=1)
    best_q_idx = np.argmin(results_matrix, axis=1)
    best_q = [q_values[idx] for idx in best_q_idx]
    
    axes[0, 1].loglog(dt_values, best_rms, 'o-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Sampling Time dt (s)')
    axes[0, 1].set_ylabel('Best RMS Error (deg)')
    axes[0, 1].set_title('최적 튜닝 후 RMS vs 샘플링 시간')
    axes[0, 1].grid(True, alpha=0.3, which='both')
    axes[0, 1].axhline(y=0.5, color='r', linestyle='--', label='Target (0.5°)')
    
    # 주파수별 표시
    for dt, rms in zip(dt_values, best_rms):
        axes[0, 1].annotate(f'{1/dt:.0f}Hz', xy=(dt, rms), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[0, 1].legend()
    
    # 3. 최적 Q vs dt
    axes[1, 0].loglog(dt_values, best_q, 'o-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Sampling Time dt (s)')
    axes[1, 0].set_ylabel('Optimal Q value')
    axes[1, 0].set_title('최적 Q vs 샘플링 시간')
    axes[1, 0].grid(True, alpha=0.3, which='both')
    
    # 4. 가장 좋은 결과 시각화
    best_overall_idx = np.unravel_index(np.argmin(results_matrix), results_matrix.shape)
    best_dt = dt_values[best_overall_idx[0]]
    best_q_overall = q_values[best_overall_idx[1]]
    
    # 최적 조합으로 다시 실행
    t, theta_true, omega_true, alpha_true = generate_trapezoid_profile(
        best_dt, total_time, rpm_max=1000, accel_time=0.4
    )
    np.random.seed(42)
    theta_meas = theta_true + np.random.normal(0, noise_std, len(theta_true))
    theta_est, omega_est = run_cv_filter(theta_meas, omega_true[0], best_dt, best_q_overall, noise_std)
    
    err = theta_est - theta_true
    
    # 에러 플롯
    axes[1, 1].plot(t, np.degrees(err), 'b-', linewidth=1)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Error (deg)')
    axes[1, 1].set_title(f'Best: dt={best_dt*1000:.2f}ms ({1/best_dt:.0f}Hz), Q={best_q_overall:.1e}, RMS={np.min(results_matrix):.3f}°')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([-2, 2])
    
    # 구간별 배경색
    axes[1, 1].axvspan(0, 0.4, alpha=0.2, color='red', label='가속')
    axes[1, 1].axvspan(0.4, 1.6, alpha=0.2, color='green', label='등속')
    axes[1, 1].axvspan(1.6, 2.0, alpha=0.2, color='blue', label='감속')
    axes[1, 1].legend()
    
    plt.suptitle('CV 칼만필터: 촘촘한 샘플링으로 가속도 프로파일 처리', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 결과 테이블 출력
    print("\n" + "="*70)
    print("CV 칼만필터 성능 비교 (RMS 에러, 단위: degree)")
    print("="*70)
    print(f"{'dt (ms)':<12} {'Freq (Hz)':<12} {'최적 Q':<12} {'RMS (deg)':<12}")
    print("-"*48)
    
    for i, dt in enumerate(dt_values):
        freq = 1/dt
        print(f"{dt*1000:<12.2f} {freq:<12.0f} {best_q[i]:<12.1e} {best_rms[i]:<12.3f}")
    
    print("\n" + "="*70)
    print(f"최고 성능: dt={best_dt*1000:.2f}ms ({1/best_dt:.0f}Hz), Q={best_q_overall:.1e}")
    print(f"RMS 에러: {np.min(results_matrix):.3f}°")
    print("="*70)
    
    # 구간별 성능 분석
    t, theta_true, omega_true, alpha_true = generate_trapezoid_profile(
        best_dt, total_time, rpm_max=1000, accel_time=0.4
    )
    
    accel_idx = (t <= 0.4)
    const_idx = (t > 0.4) & (t <= 1.6)
    decel_idx = (t > 1.6)
    
    print(f"\n구간별 성능 (최적 설정):")
    print(f"  가속 구간: {np.degrees(np.sqrt(np.mean(err[accel_idx]**2))):.3f}°")
    print(f"  등속 구간: {np.degrees(np.sqrt(np.mean(err[const_idx]**2))):.3f}°")
    print(f"  감속 구간: {np.degrees(np.sqrt(np.mean(err[decel_idx]**2))):.3f}°")
    
    plt.show()

if __name__ == "__main__":
    main()