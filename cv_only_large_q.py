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
    
    theta_unwrap = np.cumsum(omega) * dt
    return t, theta_unwrap, omega, alpha

def run_circular_cv_filter(theta_meas_wrapped, omega_init, dt, q_value, noise_std):
    """순환 CV 칼만필터 실행"""
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.F = np.array([[1.0, dt], [0.0, 1.0]])
    kf.H = np.array([[1.0, 0.0]])
    kf.Q = q_value * np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]])
    kf.R = np.array([[noise_std**2]])
    kf.P = np.diag([noise_std**2, 0.1])
    kf.x = np.array([[theta_meas_wrapped[0]], [omega_init]])
    
    theta_est = np.zeros_like(theta_meas_wrapped)
    omega_est = np.zeros_like(theta_meas_wrapped)
    
    for i in range(len(theta_meas_wrapped)):
        if i > 0:
            kf.predict()
        
        # 순환 혁신 계산
        z = theta_meas_wrapped[i]
        h_x = wrap_2pi(kf.x[0, 0])
        innovation = diffpi(z, h_x)
        
        # 칼만 업데이트
        S = kf.H @ kf.P @ kf.H.T + kf.R
        K = kf.P @ kf.H.T / S[0, 0]
        kf.x = kf.x + K * innovation
        kf.P = (np.eye(2) - K @ kf.H) @ kf.P
        
        theta_est[i] = kf.x[0, 0]
        omega_est[i] = kf.x[1, 0]
    
    return theta_est, omega_est

def main():
    # 파라미터
    dt = 0.0005  # 2kHz (더 촘촘하게)
    total_time = 2.0
    noise_std = np.deg2rad(0.5)
    q_large = 100.0  # 큰 Q값 사용
    
    print("=== CV 모델 + 큰 Q값으로 가속도 대응 ===\n")
    print(f"샘플링: {dt*1000:.1f}ms ({1/dt:.0f}Hz)")
    print(f"Q값: {q_large}")
    
    # 사다리꼴 프로파일 생성
    t, theta_true_unwrap, omega_true, alpha_true = generate_trapezoid_profile(
        dt, total_time, rpm_max=1000, accel_time=0.4
    )
    
    # 현실적인 측정값: wrap된 각도 + 노이즈
    np.random.seed(42)
    theta_true_wrapped = wrap_2pi(theta_true_unwrap)
    theta_meas_wrapped = wrap_2pi(theta_true_wrapped + np.random.normal(0, noise_std, len(t)))
    
    # CV 칼만필터 실행
    theta_cv, omega_cv = run_circular_cv_filter(theta_meas_wrapped, omega_true[0], dt, q_large, noise_std)
    
    # 에러 계산
    err_meas = diffpi(theta_meas_wrapped, theta_true_wrapped)
    err_cv = theta_cv - theta_true_unwrap
    
    rms_meas = np.degrees(np.sqrt(np.mean(err_meas**2)))
    rms_cv = np.degrees(np.sqrt(np.mean(err_cv**2)))
    
    print(f"측정 RMS: {rms_meas:.3f}°")
    print(f"CV RMS: {rms_cv:.3f}° (개선율: {rms_meas/rms_cv:.1f}x)")
    
    # 구간 정의
    accel_time = 0.4
    accel_idx = (t <= accel_time)
    const_idx = (t > accel_time) & (t <= total_time - accel_time)  
    decel_idx = (t > total_time - accel_time)
    
    # 구간별 성능
    print(f"\n=== 구간별 성능 ===")
    print(f"가속 구간 (0-{accel_time}s): {np.degrees(np.sqrt(np.mean(err_cv[accel_idx]**2))):.3f}°")
    print(f"등속 구간 ({accel_time}-{total_time-accel_time}s): {np.degrees(np.sqrt(np.mean(err_cv[const_idx]**2))):.3f}°")
    print(f"감속 구간 ({total_time-accel_time}-{total_time}s): {np.degrees(np.sqrt(np.mean(err_cv[decel_idx]**2))):.3f}°")
    
    # 시각화 - 구간별 상세 분석
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    segments = [
        ("가속", accel_idx, 'red'),
        ("등속", const_idx, 'green'), 
        ("감속", decel_idx, 'blue')
    ]
    
    for seg_idx, (seg_name, seg_mask, color) in enumerate(segments):
        t_seg = t[seg_mask]
        
        # 각도 비교 (wrapped)
        axes[seg_idx, 0].plot(t_seg, np.degrees(theta_true_wrapped[seg_mask]), 'k-', 
                              label='참값', linewidth=2)
        axes[seg_idx, 0].plot(t_seg[::5], np.degrees(theta_meas_wrapped[seg_mask][::5]), 
                              'gray', marker='.', markersize=3, linestyle='', alpha=0.5, label='측정')
        axes[seg_idx, 0].plot(t_seg, np.degrees(wrap_2pi(theta_cv[seg_mask])), color, 
                              label='칼만예측', linewidth=1.5, alpha=0.8)
        axes[seg_idx, 0].set_ylabel('각도 (deg)')
        axes[seg_idx, 0].set_title(f'{seg_name} 구간 - Wrapped 각도')
        axes[seg_idx, 0].legend()
        axes[seg_idx, 0].grid(True, alpha=0.3)
        
        # 속도 비교
        axes[seg_idx, 1].plot(t_seg, omega_true[seg_mask] * 60/(2*np.pi), 'k-', 
                              label='참값', linewidth=2)
        axes[seg_idx, 1].plot(t_seg, omega_cv[seg_mask] * 60/(2*np.pi), color, 
                              label='칼만예측', linewidth=1.5, alpha=0.8)
        axes[seg_idx, 1].set_ylabel('속도 (RPM)')
        axes[seg_idx, 1].set_title(f'{seg_name} 구간 - 속도')
        axes[seg_idx, 1].legend()
        axes[seg_idx, 1].grid(True, alpha=0.3)
        
        # 에러 분석
        err_seg = err_cv[seg_mask]
        axes[seg_idx, 2].plot(t_seg, np.degrees(err_seg), color, linewidth=1.5)
        axes[seg_idx, 2].axhline(0, color='k', linestyle='--', alpha=0.5)
        axes[seg_idx, 2].set_ylabel('각도 에러 (deg)')
        axes[seg_idx, 2].set_title(f'{seg_name} 구간 - 에러 (RMS: {np.degrees(np.sqrt(np.mean(err_seg**2))):.3f}°)')
        axes[seg_idx, 2].grid(True, alpha=0.3)
        axes[seg_idx, 2].set_ylim([-2, 2])
        
        if seg_idx == 2:  # 마지막 행
            axes[seg_idx, 0].set_xlabel('시간 (s)')
            axes[seg_idx, 1].set_xlabel('시간 (s)')
            axes[seg_idx, 2].set_xlabel('시간 (s)')
    
    plt.suptitle(f'CV 칼만필터 구간별 분석 (Q={q_large}, {1/dt:.0f}Hz)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 전체 시간 시각화
    fig2, axes2 = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # 전체 각도
    axes2[0].plot(t, np.degrees(theta_true_wrapped), 'k-', label='참값', linewidth=2)
    axes2[0].plot(t[::20], np.degrees(theta_meas_wrapped[::20]), 'gray', 
                  marker='.', markersize=1, linestyle='', alpha=0.3, label='측정')
    axes2[0].plot(t, np.degrees(wrap_2pi(theta_cv)), 'b-', label='칼만예측', alpha=0.8)
    axes2[0].set_ylabel('각도 (deg)')
    axes2[0].set_title(f'전체 각도 (RMS: {rms_cv:.3f}°)')
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)
    
    # 전체 속도
    axes2[1].plot(t, omega_true * 60/(2*np.pi), 'k-', label='참값', linewidth=2)
    axes2[1].plot(t, omega_cv * 60/(2*np.pi), 'b-', label='칼만예측', alpha=0.8)
    axes2[1].set_ylabel('속도 (RPM)')
    axes2[1].set_title('속도 추정')
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3)
    
    # 전체 에러
    axes2[2].plot(t, np.degrees(err_meas), 'gray', label=f'측정에러 ({rms_meas:.2f}°)', 
                  linewidth=0.5, alpha=0.5)
    axes2[2].plot(t, np.degrees(err_cv), 'b-', label=f'칼만에러 ({rms_cv:.2f}°)', linewidth=1)
    axes2[2].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes2[2].set_xlabel('시간 (s)')
    axes2[2].set_ylabel('각도 에러 (deg)')
    axes2[2].set_title('전체 에러')
    axes2[2].legend()
    axes2[2].grid(True, alpha=0.3)
    axes2[2].set_ylim([-3, 3])
    
    # 구간 배경 표시
    for ax in axes2:
        ax.axvspan(0, accel_time, alpha=0.1, color='red', label='가속')
        ax.axvspan(accel_time, total_time-accel_time, alpha=0.1, color='green', label='등속')
        ax.axvspan(total_time-accel_time, total_time, alpha=0.1, color='blue', label='감속')
    
    plt.suptitle(f'CV 칼만필터 전체 성능 (Q={q_large})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 성능 요약
    print(f"\n=== 성능 요약 ===")
    print(f"큰 Q값 ({q_large})으로 CV 모델이 가속도에 잘 대응!")
    print(f"전체 RMS: {rms_cv:.3f}° (측정값 대비 {rms_meas/rms_cv:.1f}x 개선)")
    print(f"등속 구간에서 가장 우수한 성능")
    
    plt.show()

if __name__ == "__main__":
    main()