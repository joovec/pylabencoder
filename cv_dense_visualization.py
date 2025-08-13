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
    total_time = 2.0
    noise_std = np.deg2rad(0.5)
    
    # 세 가지 대표적인 샘플링 주파수 비교
    configs = [
        {'dt': 0.0001, 'q': 0.1, 'label': '10 kHz', 'color': 'green'},
        {'dt': 0.001, 'q': 0.1, 'label': '1 kHz', 'color': 'blue'},
        {'dt': 0.005, 'q': 0.1, 'label': '200 Hz', 'color': 'red'}
    ]
    
    # Figure 1: 성능 비교
    fig1, axes1 = plt.subplots(3, 3, figsize=(15, 10))
    
    for idx, config in enumerate(configs):
        dt = config['dt']
        q = config['q']
        
        # 데이터 생성
        t, theta_true, omega_true, alpha_true = generate_trapezoid_profile(
            dt, total_time, rpm_max=1000, accel_time=0.4
        )
        
        np.random.seed(42)
        theta_meas = theta_true + np.random.normal(0, noise_std, len(theta_true))
        
        # 필터 실행
        theta_est, omega_est = run_cv_filter(theta_meas, omega_true[0], dt, q, noise_std)
        
        err = theta_est - theta_true
        rms = np.degrees(np.sqrt(np.mean(err**2)))
        
        # 속도 프로파일
        axes1[0, idx].plot(t, omega_true * 60/(2*np.pi), 'k-', label='True', linewidth=2)
        axes1[0, idx].plot(t, omega_est * 60/(2*np.pi), config['color'], 
                          label=f'Est (RMS={rms:.2f}°)', alpha=0.8, linewidth=1.5)
        axes1[0, idx].set_title(f'{config["label"]} (dt={dt*1000:.1f}ms)')
        axes1[0, idx].set_ylabel('RPM')
        axes1[0, idx].legend(fontsize=9)
        axes1[0, idx].grid(True, alpha=0.3)
        
        # 각도
        axes1[1, idx].plot(t[::max(1, len(t)//1000)], np.degrees(theta_true[::max(1, len(t)//1000)]), 
                          'k-', label='True', linewidth=2)
        axes1[1, idx].plot(t[::max(1, len(t)//1000)], np.degrees(theta_meas[::max(1, len(t)//1000)]), 
                          'gray', marker='.', markersize=1, linestyle='', alpha=0.3, label='Meas')
        axes1[1, idx].plot(t[::max(1, len(t)//1000)], np.degrees(theta_est[::max(1, len(t)//1000)]), 
                          config['color'], label='Estimate', alpha=0.8, linewidth=1.5)
        axes1[1, idx].set_ylabel('Angle (deg)')
        axes1[1, idx].legend(fontsize=9)
        axes1[1, idx].grid(True, alpha=0.3)
        
        # 에러
        axes1[2, idx].plot(t, np.degrees(err), config['color'], linewidth=1)
        axes1[2, idx].axhspan(-0.5, 0.5, alpha=0.2, color='green', label='Target ±0.5°')
        axes1[2, idx].set_xlabel('Time (s)')
        axes1[2, idx].set_ylabel('Error (deg)')
        axes1[2, idx].set_ylim([-5, 5])
        axes1[2, idx].grid(True, alpha=0.3)
        
        # 구간별 배경
        for ax in [axes1[0, idx], axes1[1, idx], axes1[2, idx]]:
            ax.axvspan(0, 0.4, alpha=0.1, color='red')
            ax.axvspan(1.6, 2.0, alpha=0.1, color='blue')
    
    axes1[0, 0].set_title(f'10 kHz (dt=0.1ms)\n최고 성능', fontweight='bold')
    plt.suptitle('CV 칼만필터: 샘플링 주파수별 성능 비교', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Figure 2: Q 값의 영향
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    
    dt_test = 0.001  # 1kHz
    q_values = [1e-4, 1e-3, 1e-2, 1e-1]
    colors = ['purple', 'blue', 'orange', 'green']
    
    t, theta_true, omega_true, alpha_true = generate_trapezoid_profile(
        dt_test, total_time, rpm_max=1000, accel_time=0.4
    )
    np.random.seed(42)
    theta_meas = theta_true + np.random.normal(0, noise_std, len(theta_true))
    
    for q, color in zip(q_values, colors):
        theta_est, omega_est = run_cv_filter(theta_meas, omega_true[0], dt_test, q, noise_std)
        err = theta_est - theta_true
        rms = np.degrees(np.sqrt(np.mean(err**2)))
        
        # 속도
        axes2[0, 0].plot(t, omega_est * 60/(2*np.pi), color=color, 
                        label=f'Q={q:.0e} (RMS={rms:.1f}°)', alpha=0.7, linewidth=1.5)
        
        # 에러
        axes2[0, 1].plot(t, np.degrees(err), color=color, 
                        label=f'Q={q:.0e}', alpha=0.7, linewidth=1)
    
    axes2[0, 0].plot(t, omega_true * 60/(2*np.pi), 'k-', label='True', linewidth=2)
    axes2[0, 0].set_xlabel('Time (s)')
    axes2[0, 0].set_ylabel('RPM')
    axes2[0, 0].set_title('속도 추정 (1kHz 샘플링)')
    axes2[0, 0].legend(fontsize=8)
    axes2[0, 0].grid(True, alpha=0.3)
    
    axes2[0, 1].set_xlabel('Time (s)')
    axes2[0, 1].set_ylabel('Error (deg)')
    axes2[0, 1].set_title('각도 에러')
    axes2[0, 1].legend(fontsize=8)
    axes2[0, 1].grid(True, alpha=0.3)
    axes2[0, 1].set_ylim([-10, 10])
    
    # Q vs RMS 관계
    q_range = np.logspace(-6, 0, 50)
    rms_results = []
    
    for q in q_range:
        theta_est, _ = run_cv_filter(theta_meas, omega_true[0], dt_test, q, noise_std)
        err = theta_est - theta_true
        rms = np.degrees(np.sqrt(np.mean(err**2)))
        rms_results.append(rms)
    
    axes2[1, 0].loglog(q_range, rms_results, 'b-', linewidth=2)
    axes2[1, 0].scatter(q_values, [np.degrees(np.sqrt(np.mean((run_cv_filter(theta_meas, omega_true[0], 
                       dt_test, q, noise_std)[0] - theta_true)**2))) for q in q_values], 
                       c=colors, s=100, zorder=5)
    axes2[1, 0].set_xlabel('Q value')
    axes2[1, 0].set_ylabel('RMS Error (deg)')
    axes2[1, 0].set_title('Q 튜닝 곡선 (1kHz)')
    axes2[1, 0].grid(True, alpha=0.3, which='both')
    axes2[1, 0].axhline(y=0.5, color='r', linestyle='--', label='Target')
    axes2[1, 0].legend()
    
    # 샘플링 주파수 vs 최적 성능
    dt_range = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]
    best_rms = []
    
    for dt in dt_range:
        t_temp, theta_true_temp, omega_true_temp, _ = generate_trapezoid_profile(
            dt, total_time, rpm_max=1000, accel_time=0.4
        )
        np.random.seed(42)
        theta_meas_temp = theta_true_temp + np.random.normal(0, noise_std, len(theta_true_temp))
        theta_est_temp, _ = run_cv_filter(theta_meas_temp, omega_true_temp[0], dt, 0.1, noise_std)
        err_temp = theta_est_temp - theta_true_temp
        rms_temp = np.degrees(np.sqrt(np.mean(err_temp**2)))
        best_rms.append(rms_temp)
    
    freq_range = [1/dt for dt in dt_range]
    
    axes2[1, 1].semilogx(freq_range, best_rms, 'go-', linewidth=2, markersize=8)
    axes2[1, 1].set_xlabel('Sampling Frequency (Hz)')
    axes2[1, 1].set_ylabel('RMS Error (deg)')
    axes2[1, 1].set_title('최적 성능 vs 샘플링 주파수 (Q=0.1)')
    axes2[1, 1].grid(True, alpha=0.3, which='both')
    axes2[1, 1].axhline(y=0.5, color='r', linestyle='--', label='Target')
    axes2[1, 1].axhline(y=1.0, color='orange', linestyle='--', label='Acceptable')
    axes2[1, 1].legend()
    
    # 주요 포인트 표시
    for freq, rms in zip([10000, 1000, 200], [best_rms[0], best_rms[3], best_rms[5]]):
        axes2[1, 1].annotate(f'{freq}Hz\n{rms:.1f}°', xy=(freq, rms), 
                            xytext=(10, 10), textcoords='offset points', fontsize=9)
    
    plt.suptitle('CV 칼만필터: Q 값 영향 및 주파수 분석', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.show()

if __name__ == "__main__":
    main()