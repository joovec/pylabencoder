import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.family'] = 'Malgun Gothic'

def wrap_2pi(angle):
    """입력 각도를 [0, 2pi) 범위로 래핑"""
    return np.mod(angle, 2*np.pi)

def diffpi(angle1, angle2):
    """두 각도의 최단 차이 (-pi, pi]"""
    diff = angle1 - angle2
    return np.arctan2(np.sin(diff), np.cos(diff))

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
    return t, theta, omega, alpha  # theta는 언랩 상태로 반환

def main():
    # 파라미터
    dt = 0.001
    total_time = 2.0
    noise_std = np.deg2rad(0.5)
    
    # 사다리꼴 프로파일 생성 (언랩 각도)
    t, theta_true_unwrap, omega_true, alpha_true = generate_trapezoid_profile(
        dt, total_time, rpm_max=1000, accel_time=0.4
    )
    
    # 측정값 생성 (노이즈 추가, 언랩 상태)
    np.random.seed(42)
    theta_meas_unwrap = theta_true_unwrap + np.random.normal(0, noise_std, len(theta_true_unwrap))
    
    # CV 칼만필터 설정
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.F = np.array([[1.0, dt], [0.0, 1.0]])
    kf.H = np.array([[1.0, 0.0]])
    kf.Q = 1e-3 * np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]])  # 적절한 프로세스 노이즈
    kf.R = np.array([[noise_std**2]])
    kf.P = np.diag([noise_std**2, 1.0])  # 초기 불확실성
    kf.x = np.array([[theta_meas_unwrap[0]], [omega_true[0]]])
    
    # 칼만필터 실행 (언랩 상태에서)
    theta_est_unwrap = np.zeros_like(theta_meas_unwrap)
    omega_est = np.zeros_like(theta_meas_unwrap)
    
    for i in range(len(t)):
        if i > 0:
            kf.predict()
        
        # 언랩 상태에서 직접 업데이트
        kf.update(np.array([[theta_meas_unwrap[i]]]))
        
        theta_est_unwrap[i] = kf.x[0, 0]
        omega_est[i] = kf.x[1, 0]
    
    # Wrap된 각도로 변환 (시각화용)
    theta_true_wrap = wrap_2pi(theta_true_unwrap)
    theta_meas_wrap = wrap_2pi(theta_meas_unwrap)
    theta_est_wrap = wrap_2pi(theta_est_unwrap)
    
    # 에러 계산 (언랩 상태에서)
    err_meas = theta_meas_unwrap - theta_true_unwrap
    err_est = theta_est_unwrap - theta_true_unwrap
    
    rms_meas = np.degrees(np.sqrt(np.mean(err_meas**2)))
    rms_est = np.degrees(np.sqrt(np.mean(err_est**2)))
    
    # 플롯
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # 속도
    axes[0].plot(t, omega_true * 60/(2*np.pi), 'k-', label='True RPM', linewidth=2)
    axes[0].plot(t, omega_est * 60/(2*np.pi), 'b-', label='CV Kalman', alpha=0.8)
    axes[0].set_ylabel('RPM')
    axes[0].set_title('사다리꼴 속도 프로파일 - CV 칼만필터')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 가속도
    axes[1].plot(t, alpha_true * 60/(2*np.pi), 'k-', label='True Accel', linewidth=2)
    axes[1].set_ylabel('RPM/s')
    axes[1].set_title('가속도')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 각도 (언랩)
    axes[2].plot(t, np.degrees(theta_true_unwrap), 'k-', label='True', linewidth=2)
    axes[2].plot(t, np.degrees(theta_meas_unwrap), 'r.', label='Measured', markersize=0.5, alpha=0.3)
    axes[2].plot(t, np.degrees(theta_est_unwrap), 'b-', label='CV Kalman', alpha=0.8)
    axes[2].set_ylabel('Angle (deg)')
    axes[2].set_title('누적 각도 (언랩)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 에러
    axes[3].plot(t, np.degrees(err_meas), 'r-', label=f'Meas (RMS={rms_meas:.3f}°)', linewidth=0.5, alpha=0.5)
    axes[3].plot(t, np.degrees(err_est), 'b-', label=f'CV (RMS={rms_est:.3f}°)', linewidth=1)
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('Error (deg)')
    axes[3].set_title('각도 에러')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    axes[3].set_ylim([-2, 2])
    
    plt.tight_layout()
    
    print(f"\n=== 사다리꼴 프로파일 결과 ===")
    print(f"측정값 RMS: {rms_meas:.3f}°")
    print(f"CV 칼만 RMS: {rms_est:.3f}°")
    print(f"개선율: {rms_meas/rms_est:.2f}x")
    
    # 구간별 성능
    accel_idx = (t <= 0.4)
    const_idx = (t > 0.4) & (t <= 1.6)
    decel_idx = (t > 1.6)
    
    print(f"\n구간별 성능:")
    print(f"가속 구간: {np.degrees(np.sqrt(np.mean(err_est[accel_idx]**2))):.3f}°")
    print(f"등속 구간: {np.degrees(np.sqrt(np.mean(err_est[const_idx]**2))):.3f}°")
    print(f"감속 구간: {np.degrees(np.sqrt(np.mean(err_est[decel_idx]**2))):.3f}°")
    
    plt.show()

if __name__ == "__main__":
    main()