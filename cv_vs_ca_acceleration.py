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

def main():
    # 파라미터
    dt = 0.001
    total_time = 2.0
    noise_std = np.deg2rad(0.5)
    
    # 사다리꼴 프로파일 생성 (언랩 각도)
    t, theta_true, omega_true, alpha_true = generate_trapezoid_profile(
        dt, total_time, rpm_max=1000, accel_time=0.4
    )
    
    # 측정값 생성
    np.random.seed(42)
    theta_meas = theta_true + np.random.normal(0, noise_std, len(theta_true))
    
    # CV 칼만필터 (Constant Velocity)
    kf_cv = KalmanFilter(dim_x=2, dim_z=1)
    kf_cv.F = np.array([[1.0, dt], [0.0, 1.0]])
    kf_cv.H = np.array([[1.0, 0.0]])
    kf_cv.Q = 1e-3 * np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]])
    kf_cv.R = np.array([[noise_std**2]])
    kf_cv.P = np.diag([noise_std**2, 1.0])
    kf_cv.x = np.array([[theta_meas[0]], [omega_true[0]]])
    
    # CA 칼만필터 (Constant Acceleration)
    kf_ca = KalmanFilter(dim_x=3, dim_z=1)
    kf_ca.F = np.array([[1.0, dt, 0.5*dt*dt],
                        [0.0, 1.0, dt],
                        [0.0, 0.0, 1.0]])
    kf_ca.H = np.array([[1.0, 0.0, 0.0]])
    kf_ca.Q = 1e-1 * np.array([[dt**5/20, dt**4/8, dt**3/6],
                               [dt**4/8, dt**3/3, dt**2/2],
                               [dt**3/6, dt**2/2, dt]])
    kf_ca.R = np.array([[noise_std**2]])
    kf_ca.P = np.diag([noise_std**2, 1.0, 10.0])
    kf_ca.x = np.array([[theta_meas[0]], [omega_true[0]], [0.0]])
    
    # 필터 실행
    theta_cv = np.zeros_like(theta_meas)
    omega_cv = np.zeros_like(theta_meas)
    
    theta_ca = np.zeros_like(theta_meas)
    omega_ca = np.zeros_like(theta_meas)
    alpha_ca = np.zeros_like(theta_meas)
    
    for i in range(len(t)):
        # CV 필터
        if i > 0:
            kf_cv.predict()
        kf_cv.update(np.array([[theta_meas[i]]]))
        theta_cv[i] = kf_cv.x[0, 0]
        omega_cv[i] = kf_cv.x[1, 0]
        
        # CA 필터
        if i > 0:
            kf_ca.predict()
        kf_ca.update(np.array([[theta_meas[i]]]))
        theta_ca[i] = kf_ca.x[0, 0]
        omega_ca[i] = kf_ca.x[1, 0]
        alpha_ca[i] = kf_ca.x[2, 0]
    
    # 에러 계산
    err_meas = theta_meas - theta_true
    err_cv = theta_cv - theta_true
    err_ca = theta_ca - theta_true
    
    rms_meas = np.degrees(np.sqrt(np.mean(err_meas**2)))
    rms_cv = np.degrees(np.sqrt(np.mean(err_cv**2)))
    rms_ca = np.degrees(np.sqrt(np.mean(err_ca**2)))
    
    # 플롯
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # 속도
    axes[0].plot(t, omega_true * 60/(2*np.pi), 'k-', label='True', linewidth=2)
    axes[0].plot(t, omega_cv * 60/(2*np.pi), 'b-', label='CV', alpha=0.7)
    axes[0].plot(t, omega_ca * 60/(2*np.pi), 'g-', label='CA', alpha=0.7)
    axes[0].set_ylabel('RPM')
    axes[0].set_title('사다리꼴 프로파일 - CV vs CA 칼만필터')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 가속도
    axes[1].plot(t, alpha_true * 60/(2*np.pi), 'k-', label='True', linewidth=2)
    axes[1].plot(t, alpha_ca * 60/(2*np.pi), 'g-', label='CA Est', alpha=0.7)
    axes[1].set_ylabel('RPM/s')
    axes[1].set_title('가속도')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 각도
    axes[2].plot(t, np.degrees(theta_true), 'k-', label='True', linewidth=2)
    axes[2].plot(t, np.degrees(theta_meas), 'r.', label='Measured', markersize=0.5, alpha=0.3)
    axes[2].plot(t, np.degrees(theta_cv), 'b-', label='CV', alpha=0.7)
    axes[2].plot(t, np.degrees(theta_ca), 'g-', label='CA', alpha=0.7)
    axes[2].set_ylabel('Angle (deg)')
    axes[2].set_title('누적 각도')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 에러
    axes[3].plot(t, np.degrees(err_meas), 'r-', label=f'Meas ({rms_meas:.3f}°)', linewidth=0.3, alpha=0.5)
    axes[3].plot(t, np.degrees(err_cv), 'b-', label=f'CV ({rms_cv:.3f}°)', linewidth=1)
    axes[3].plot(t, np.degrees(err_ca), 'g-', label=f'CA ({rms_ca:.3f}°)', linewidth=1)
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('Error (deg)')
    axes[3].set_title('각도 에러')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    axes[3].set_ylim([-2, 2])
    
    plt.tight_layout()
    
    print(f"\n=== 가속도 프로파일 결과 ===")
    print(f"측정값 RMS: {rms_meas:.3f}°")
    print(f"CV 칼만 RMS: {rms_cv:.3f}° (개선율: {rms_meas/rms_cv:.2f}x)")
    print(f"CA 칼만 RMS: {rms_ca:.3f}° (개선율: {rms_meas/rms_ca:.2f}x)")
    
    # 구간별 성능
    accel_idx = (t <= 0.4)
    const_idx = (t > 0.4) & (t <= 1.6)
    decel_idx = (t > 1.6)
    
    print(f"\nCV 구간별 성능:")
    print(f"  가속: {np.degrees(np.sqrt(np.mean(err_cv[accel_idx]**2))):.3f}°")
    print(f"  등속: {np.degrees(np.sqrt(np.mean(err_cv[const_idx]**2))):.3f}°")
    print(f"  감속: {np.degrees(np.sqrt(np.mean(err_cv[decel_idx]**2))):.3f}°")
    
    print(f"\nCA 구간별 성능:")
    print(f"  가속: {np.degrees(np.sqrt(np.mean(err_ca[accel_idx]**2))):.3f}°")
    print(f"  등속: {np.degrees(np.sqrt(np.mean(err_ca[const_idx]**2))):.3f}°")
    print(f"  감속: {np.degrees(np.sqrt(np.mean(err_ca[decel_idx]**2))):.3f}°")
    
    plt.show()

if __name__ == "__main__":
    main()