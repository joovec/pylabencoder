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
    dt = 0.001  # 1kHz
    total_time = 2.0
    noise_std = np.deg2rad(0.5)
    
    # 데이터 생성
    t, theta_true, omega_true, alpha_true = generate_trapezoid_profile(
        dt, total_time, rpm_max=1000, accel_time=0.4
    )
    
    np.random.seed(42)
    theta_meas = theta_true + np.random.normal(0, noise_std, len(theta_true))
    
    # ========== STEP 1: CV 칼만필터 (각도 측정 -> 각속도 추정) ==========
    kf_cv = KalmanFilter(dim_x=2, dim_z=1)
    kf_cv.F = np.array([[1.0, dt], [0.0, 1.0]])
    kf_cv.H = np.array([[1.0, 0.0]])  # 각도만 측정
    kf_cv.Q = 1e-1 * np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]])  # CV도 Q 증가
    kf_cv.R = np.array([[noise_std**2]])
    kf_cv.P = np.diag([noise_std**2, 1.0])
    kf_cv.x = np.array([[theta_meas[0]], [omega_true[0]]])
    
    theta_cv = np.zeros_like(theta_meas)
    omega_cv = np.zeros_like(theta_meas)
    
    print("Step 1: CV 필터 실행 (각도 측정 -> 각속도 추정)")
    for i in range(len(t)):
        if i > 0:
            kf_cv.predict()
        kf_cv.update(np.array([[theta_meas[i]]]))
        theta_cv[i] = kf_cv.x[0, 0]
        omega_cv[i] = kf_cv.x[1, 0]  # 이 각속도를 CA의 측정값으로 사용
    
    # ========== STEP 2: CA 칼만필터 (각도+각속도 측정 -> 가속도 추정) ==========
    kf_ca = KalmanFilter(dim_x=3, dim_z=2)  # 2개 측정 (각도, 각속도)
    kf_ca.F = np.array([[1.0, dt, 0.5*dt*dt],
                        [0.0, 1.0, dt],
                        [0.0, 0.0, 1.0]])
    kf_ca.H = np.array([[1.0, 0.0, 0.0],   # 각도 측정
                        [0.0, 1.0, 0.0]])   # 각속도 측정 (CV에서 추정한 값)
    
    # 프로세스 노이즈
    kf_ca.Q = 1e-1 * np.array([[dt**5/20, dt**4/8, dt**3/6],
                               [dt**4/8, dt**3/3, dt**2/2],
                               [dt**3/6, dt**2/2, dt]])
    
    # 측정 노이즈 (각도는 원래 노이즈, 각속도는 CV 추정 불확실성)
    # 등속 구간에서는 CV 각속도가 정확하므로 작은 노이즈
    # 가속 구간에서는 부정확하므로 큰 노이즈
    omega_cv_noise = 0.01  # CV 각속도 추정의 불확실성 (작게 설정하여 더 신뢰)
    kf_ca.R = np.array([[noise_std**2, 0],
                        [0, omega_cv_noise]])
    
    kf_ca.P = np.diag([noise_std**2, 1.0, 10.0])
    kf_ca.x = np.array([[theta_meas[0]], [omega_cv[0]], [0.0]])
    
    theta_ca = np.zeros_like(theta_meas)
    omega_ca = np.zeros_like(theta_meas)
    alpha_ca = np.zeros_like(theta_meas)
    
    print("Step 2: CA 필터 실행 (각도 + CV 각속도 측정 -> 가속도 추정)")
    for i in range(len(t)):
        if i > 0:
            kf_ca.predict()
        
        # 2개의 측정값: 원본 각도 + CV에서 추정한 각속도
        z = np.array([[theta_meas[i]], [omega_cv[i]]])
        kf_ca.update(z)
        
        theta_ca[i] = kf_ca.x[0, 0]
        omega_ca[i] = kf_ca.x[1, 0]
        alpha_ca[i] = kf_ca.x[2, 0]
    
    # ========== 표준 CA 필터 (비교용: 각도만 측정) ==========
    kf_ca_std = KalmanFilter(dim_x=3, dim_z=1)
    kf_ca_std.F = np.array([[1.0, dt, 0.5*dt*dt],
                           [0.0, 1.0, dt],
                           [0.0, 0.0, 1.0]])
    kf_ca_std.H = np.array([[1.0, 0.0, 0.0]])  # 각도만 측정
    kf_ca_std.Q = 1e-2 * np.array([[dt**5/20, dt**4/8, dt**3/6],
                                   [dt**4/8, dt**3/3, dt**2/2],
                                   [dt**3/6, dt**2/2, dt]])
    kf_ca_std.R = np.array([[noise_std**2]])
    kf_ca_std.P = np.diag([noise_std**2, 1.0, 10.0])
    kf_ca_std.x = np.array([[theta_meas[0]], [omega_true[0]], [0.0]])
    
    theta_ca_std = np.zeros_like(theta_meas)
    omega_ca_std = np.zeros_like(theta_meas)
    alpha_ca_std = np.zeros_like(theta_meas)
    
    for i in range(len(t)):
        if i > 0:
            kf_ca_std.predict()
        kf_ca_std.update(np.array([[theta_meas[i]]]))
        theta_ca_std[i] = kf_ca_std.x[0, 0]
        omega_ca_std[i] = kf_ca_std.x[1, 0]
        alpha_ca_std[i] = kf_ca_std.x[2, 0]
    
    # 에러 계산
    err_meas = theta_meas - theta_true
    err_cv = theta_cv - theta_true
    err_ca = theta_ca - theta_true
    err_ca_std = theta_ca_std - theta_true
    
    rms_meas = np.degrees(np.sqrt(np.mean(err_meas**2)))
    rms_cv = np.degrees(np.sqrt(np.mean(err_cv**2)))
    rms_ca = np.degrees(np.sqrt(np.mean(err_ca**2)))
    rms_ca_std = np.degrees(np.sqrt(np.mean(err_ca_std**2)))
    
    # 플롯
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    
    # 왼쪽: Cascade 방식
    axes[0, 0].plot(t, omega_true * 60/(2*np.pi), 'k-', label='True', linewidth=2)
    axes[0, 0].plot(t, omega_cv * 60/(2*np.pi), 'b--', label='CV 추정', alpha=0.7)
    axes[0, 0].plot(t, omega_ca * 60/(2*np.pi), 'g-', label='CA (Cascade)', alpha=0.8)
    axes[0, 0].set_ylabel('RPM')
    axes[0, 0].set_title('Cascade: CV 각속도 → CA 측정값')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[1, 0].plot(t, alpha_true * 60/(2*np.pi), 'k-', label='True', linewidth=2)
    axes[1, 0].plot(t, alpha_ca * 60/(2*np.pi), 'g-', label='CA (Cascade)', alpha=0.8)
    axes[1, 0].set_ylabel('RPM/s')
    axes[1, 0].set_title('가속도 추정')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[2, 0].plot(t, np.degrees(theta_true), 'k-', label='True', linewidth=2)
    axes[2, 0].plot(t, np.degrees(theta_meas), 'gray', marker='.', markersize=0.5, 
                   linestyle='', alpha=0.3, label='Meas')
    axes[2, 0].plot(t, np.degrees(theta_cv), 'b--', label=f'CV ({rms_cv:.2f}°)', alpha=0.7)
    axes[2, 0].plot(t, np.degrees(theta_ca), 'g-', label=f'CA Cascade ({rms_ca:.2f}°)', alpha=0.8)
    axes[2, 0].set_ylabel('Angle (deg)')
    axes[2, 0].set_title('각도 추정')
    axes[2, 0].legend(fontsize=8)
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[3, 0].plot(t, np.degrees(err_cv), 'b--', label='CV', alpha=0.7, linewidth=1)
    axes[3, 0].plot(t, np.degrees(err_ca), 'g-', label='CA (Cascade)', alpha=0.8, linewidth=1.5)
    axes[3, 0].set_xlabel('Time (s)')
    axes[3, 0].set_ylabel('Error (deg)')
    axes[3, 0].set_title('각도 에러')
    axes[3, 0].legend()
    axes[3, 0].grid(True, alpha=0.3)
    axes[3, 0].set_ylim([-5, 5])
    
    # 오른쪽: 표준 CA vs Cascade CA
    axes[0, 1].plot(t, omega_true * 60/(2*np.pi), 'k-', label='True', linewidth=2)
    axes[0, 1].plot(t, omega_ca_std * 60/(2*np.pi), 'r--', label='CA 표준', alpha=0.7)
    axes[0, 1].plot(t, omega_ca * 60/(2*np.pi), 'g-', label='CA Cascade', alpha=0.8)
    axes[0, 1].set_ylabel('RPM')
    axes[0, 1].set_title('표준 CA vs Cascade CA')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 1].plot(t, alpha_true * 60/(2*np.pi), 'k-', label='True', linewidth=2)
    axes[1, 1].plot(t, alpha_ca_std * 60/(2*np.pi), 'r--', label='CA 표준', alpha=0.7)
    axes[1, 1].plot(t, alpha_ca * 60/(2*np.pi), 'g-', label='CA Cascade', alpha=0.8)
    axes[1, 1].set_ylabel('RPM/s')
    axes[1, 1].set_title('가속도 비교')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[2, 1].plot(t, np.degrees(theta_true), 'k-', label='True', linewidth=2)
    axes[2, 1].plot(t, np.degrees(theta_ca_std), 'r--', 
                   label=f'CA 표준 ({rms_ca_std:.2f}°)', alpha=0.7)
    axes[2, 1].plot(t, np.degrees(theta_ca), 'g-', 
                   label=f'CA Cascade ({rms_ca:.2f}°)', alpha=0.8)
    axes[2, 1].set_ylabel('Angle (deg)')
    axes[2, 1].set_title('각도 비교')
    axes[2, 1].legend(fontsize=8)
    axes[2, 1].grid(True, alpha=0.3)
    
    axes[3, 1].plot(t, np.degrees(err_ca_std), 'r--', label='CA 표준', alpha=0.7, linewidth=1)
    axes[3, 1].plot(t, np.degrees(err_ca), 'g-', label='CA Cascade', alpha=0.8, linewidth=1.5)
    axes[3, 1].set_xlabel('Time (s)')
    axes[3, 1].set_ylabel('Error (deg)')
    axes[3, 1].set_title('에러 비교')
    axes[3, 1].legend()
    axes[3, 1].grid(True, alpha=0.3)
    axes[3, 1].set_ylim([-5, 5])
    
    # 구간 표시
    for ax in axes.flat:
        ax.axvspan(0, 0.4, alpha=0.1, color='red')    # 가속
        ax.axvspan(1.6, 2.0, alpha=0.1, color='blue')  # 감속
    
    plt.suptitle('CV-CA Cascade: CV 각속도를 CA 측정값으로 활용', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 성능 비교 출력
    print("\n" + "="*60)
    print("성능 비교 (RMS 각도 에러)")
    print("="*60)
    print(f"측정값:        {rms_meas:.3f}°")
    print(f"CV:            {rms_cv:.3f}°")
    print(f"CA 표준:       {rms_ca_std:.3f}°")
    print(f"CA Cascade:    {rms_ca:.3f}°")
    print("="*60)
    
    # 구간별 성능
    accel_idx = (t <= 0.4)
    const_idx = (t > 0.4) & (t <= 1.6)
    decel_idx = (t > 1.6)
    
    print("\nCA Cascade 구간별 성능:")
    print(f"  가속 구간: {np.degrees(np.sqrt(np.mean(err_ca[accel_idx]**2))):.3f}°")
    print(f"  등속 구간: {np.degrees(np.sqrt(np.mean(err_ca[const_idx]**2))):.3f}°")
    print(f"  감속 구간: {np.degrees(np.sqrt(np.mean(err_ca[decel_idx]**2))):.3f}°")
    
    print("\nCA 표준 구간별 성능:")
    print(f"  가속 구간: {np.degrees(np.sqrt(np.mean(err_ca_std[accel_idx]**2))):.3f}°")
    print(f"  등속 구간: {np.degrees(np.sqrt(np.mean(err_ca_std[const_idx]**2))):.3f}°")
    print(f"  감속 구간: {np.degrees(np.sqrt(np.mean(err_ca_std[decel_idx]**2))):.3f}°")
    
    plt.show()

if __name__ == "__main__":
    main()