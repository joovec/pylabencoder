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
    kf.P = np.diag([noise_std**2, 1.0])
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
    # 샘플링 시간과 Q 값 조합 테스트
    dt_values = [0.001, 0.002, 0.005, 0.01]  # 1kHz, 500Hz, 200Hz, 100Hz
    q_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    
    total_time = 2.0
    noise_std = np.deg2rad(0.5)
    
    # 결과 저장
    results = {}
    
    fig, axes = plt.subplots(len(dt_values), 2, figsize=(14, 12))
    
    for dt_idx, dt in enumerate(dt_values):
        # 프로파일 생성
        t, theta_true, omega_true, alpha_true = generate_trapezoid_profile(
            dt, total_time, rpm_max=1000, accel_time=0.4
        )
        
        # 측정값 생성
        np.random.seed(42)
        theta_meas = theta_true + np.random.normal(0, noise_std, len(theta_true))
        
        rms_errors = []
        
        # 각 Q 값에 대해 테스트
        for q in q_values:
            theta_est, omega_est = run_cv_filter(theta_meas, omega_true[0], dt, q, noise_std)
            err = theta_est - theta_true
            rms = np.degrees(np.sqrt(np.mean(err**2)))
            rms_errors.append(rms)
            
            # 최적 Q 값 저장
            if not results.get(dt) or rms < results[dt]['rms']:
                results[dt] = {'q': q, 'rms': rms, 'theta_est': theta_est, 'omega_est': omega_est}
        
        # Q에 따른 RMS 플롯
        axes[dt_idx, 0].semilogx(q_values, rms_errors, 'o-', linewidth=2, markersize=8)
        axes[dt_idx, 0].set_xlabel('Q value')
        axes[dt_idx, 0].set_ylabel('RMS Error (deg)')
        axes[dt_idx, 0].set_title(f'dt={dt*1000:.0f}ms ({1/dt:.0f}Hz)')
        axes[dt_idx, 0].grid(True, alpha=0.3, which='both')
        axes[dt_idx, 0].axhline(y=0.5, color='r', linestyle='--', label='Target (0.5°)')
        
        # 최적 Q에서의 각도 추정
        best = results[dt]
        axes[dt_idx, 1].plot(t, omega_true * 60/(2*np.pi), 'k-', label='True', linewidth=2)
        axes[dt_idx, 1].plot(t, best['omega_est'] * 60/(2*np.pi), 'b-', 
                            label=f"Est (Q={best['q']:.0e})", alpha=0.8)
        axes[dt_idx, 1].set_xlabel('Time (s)')
        axes[dt_idx, 1].set_ylabel('RPM')
        axes[dt_idx, 1].set_title(f"Best: RMS={best['rms']:.3f}°")
        axes[dt_idx, 1].legend()
        axes[dt_idx, 1].grid(True, alpha=0.3)
    
    plt.suptitle('CV 칼만필터: 샘플링 시간과 Q 값 튜닝', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 최적 조합 출력
    print("\n=== CV 칼만필터 최적 튜닝 결과 ===")
    print(f"{'dt (ms)':<10} {'Freq (Hz)':<10} {'Q value':<12} {'RMS (deg)':<10}")
    print("-" * 45)
    
    for dt in dt_values:
        best = results[dt]
        print(f"{dt*1000:<10.1f} {1/dt:<10.0f} {best['q']:<12.2e} {best['rms']:<10.3f}")
    
    # 추가 분석: 최적 Q와 dt의 관계
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    optimal_q = [results[dt]['q'] for dt in dt_values]
    plt.loglog(dt_values, optimal_q, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Sampling Time dt (s)')
    plt.ylabel('Optimal Q value')
    plt.title('최적 Q vs 샘플링 시간')
    plt.grid(True, alpha=0.3, which='both')
    
    # Q를 dt의 함수로 피팅
    # Q = a * dt^b 형태로 가정
    log_dt = np.log(dt_values)
    log_q = np.log(optimal_q)
    coeffs = np.polyfit(log_dt, log_q, 1)
    b = coeffs[0]
    a = np.exp(coeffs[1])
    
    dt_fit = np.logspace(np.log10(min(dt_values)), np.log10(max(dt_values)), 100)
    q_fit = a * dt_fit**b
    plt.plot(dt_fit, q_fit, 'r--', label=f'Q = {a:.2e} * dt^{b:.2f}')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    optimal_rms = [results[dt]['rms'] for dt in dt_values]
    plt.semilogx(dt_values, optimal_rms, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Sampling Time dt (s)')
    plt.ylabel('RMS Error (deg)')
    plt.title('최적 튜닝 후 RMS 에러')
    plt.grid(True, alpha=0.3, which='both')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Target')
    plt.legend()
    
    plt.suptitle('Q 값과 샘플링 시간의 관계', fontsize=12)
    plt.tight_layout()
    
    print(f"\n추천 Q 값 공식: Q = {a:.2e} * dt^{b:.2f}")
    print("(가속도가 있는 사다리꼴 프로파일 기준)")
    
    # 실제 적용 예시
    print("\n=== 실제 적용 예시 ===")
    test_dt = 0.001  # 1kHz
    recommended_q = a * test_dt**b
    print(f"dt = {test_dt*1000}ms ({1/test_dt:.0f}Hz) -> 추천 Q = {recommended_q:.2e}")
    
    # 추천 Q로 테스트
    t, theta_true, omega_true, alpha_true = generate_trapezoid_profile(
        test_dt, total_time, rpm_max=1000, accel_time=0.4
    )
    np.random.seed(42)
    theta_meas = theta_true + np.random.normal(0, noise_std, len(theta_true))
    theta_est, omega_est = run_cv_filter(theta_meas, omega_true[0], test_dt, recommended_q, noise_std)
    
    err = theta_est - theta_true
    rms = np.degrees(np.sqrt(np.mean(err**2)))
    print(f"결과 RMS: {rms:.3f}°")
    
    plt.show()

if __name__ == "__main__":
    main()