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

def run_circular_ca_filter(theta_meas_wrapped, omega_init, dt, q_value, noise_std):
    """순환 CA 칼만필터 실행"""
    kf = KalmanFilter(dim_x=3, dim_z=1)
    kf.F = np.array([[1.0, dt, 0.5*dt**2],
                     [0.0, 1.0, dt],
                     [0.0, 0.0, 1.0]])
    kf.H = np.array([[1.0, 0.0, 0.0]])
    kf.Q = q_value * np.array([[dt**5/20, dt**4/8, dt**3/6],
                               [dt**4/8, dt**3/3, dt**2/2],
                               [dt**3/6, dt**2/2, dt]])
    kf.R = np.array([[noise_std**2]])
    kf.P = np.diag([noise_std**2, 0.1, 1.0])
    kf.x = np.array([[theta_meas_wrapped[0]], [omega_init], [0.0]])
    
    theta_est = np.zeros_like(theta_meas_wrapped)
    omega_est = np.zeros_like(theta_meas_wrapped)
    alpha_est = np.zeros_like(theta_meas_wrapped)
    
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
        kf.P = (np.eye(3) - K @ kf.H) @ kf.P
        
        theta_est[i] = kf.x[0, 0]
        omega_est[i] = kf.x[1, 0]
        alpha_est[i] = kf.x[2, 0]
    
    return theta_est, omega_est, alpha_est

def main():
    # 파라미터
    dt = 0.0005  # 2kHz
    total_time = 2.0
    noise_std = np.deg2rad(0.5)
    
    # 큰 Q값들
    q_cv_large = 500.0   # CV용 큰 Q값
    q_ca_large = 1000.0  # CA용 더 큰 Q값
    
    print("=== CV vs CA 모델: 큰 Q값 비교 ===\n")
    print(f"샘플링: {dt*1000:.1f}ms ({1/dt:.0f}Hz)")
    print(f"CV Q값: {q_cv_large}")
    print(f"CA Q값: {q_ca_large}")
    
    # 사다리꼴 프로파일 생성
    t, theta_true_unwrap, omega_true, alpha_true = generate_trapezoid_profile(
        dt, total_time, rpm_max=1000, accel_time=0.4
    )
    
    # 현실적인 측정값: wrap된 각도 + 노이즈
    np.random.seed(42)
    theta_true_wrapped = wrap_2pi(theta_true_unwrap)
    theta_meas_wrapped = wrap_2pi(theta_true_wrapped + np.random.normal(0, noise_std, len(t)))
    
    # CV 칼만필터 실행
    print("\nCV 필터 실행 중...")
    theta_cv, omega_cv = run_circular_cv_filter(theta_meas_wrapped, omega_true[0], dt, q_cv_large, noise_std)
    
    # CA 칼만필터 실행
    print("CA 필터 실행 중...")
    theta_ca, omega_ca, alpha_ca = run_circular_ca_filter(theta_meas_wrapped, omega_true[0], dt, q_ca_large, noise_std)
    
    # 에러 계산
    err_meas = diffpi(theta_meas_wrapped, theta_true_wrapped)
    err_cv = theta_cv - theta_true_unwrap
    err_ca = theta_ca - theta_true_unwrap
    
    rms_meas = np.degrees(np.sqrt(np.mean(err_meas**2)))
    rms_cv = np.degrees(np.sqrt(np.mean(err_cv**2)))
    rms_ca = np.degrees(np.sqrt(np.mean(err_ca**2)))
    
    print(f"\n=== 전체 성능 비교 ===")
    print(f"측정 RMS:    {rms_meas:.3f}°")
    print(f"CV RMS:      {rms_cv:.3f}° (개선율: {rms_meas/rms_cv:.1f}x)")
    print(f"CA RMS:      {rms_ca:.3f}° (개선율: {rms_meas/rms_ca:.1f}x)")
    
    if rms_cv < rms_ca:
        print(f"🏆 CV 모델 승리! ({rms_ca/rms_cv:.1f}x 더 우수)")
    else:
        print(f"🏆 CA 모델 승리! ({rms_cv/rms_ca:.1f}x 더 우수)")
    
    # 구간별 성능
    accel_time = 0.4
    accel_idx = (t <= accel_time)
    const_idx = (t > accel_time) & (t <= total_time - accel_time)
    decel_idx = (t > total_time - accel_time)
    
    print(f"\n=== 구간별 성능 비교 ===")
    segments = [
        ("가속", accel_idx),
        ("등속", const_idx),
        ("감속", decel_idx)
    ]
    
    for seg_name, seg_idx in segments:
        cv_rms = np.degrees(np.sqrt(np.mean(err_cv[seg_idx]**2)))
        ca_rms = np.degrees(np.sqrt(np.mean(err_ca[seg_idx]**2)))
        winner = "CV" if cv_rms < ca_rms else "CA"
        ratio = ca_rms/cv_rms if cv_rms < ca_rms else cv_rms/ca_rms
        print(f"{seg_name} 구간: CV={cv_rms:.3f}°, CA={ca_rms:.3f}° → {winner} 승리 ({ratio:.1f}x)")
    
    # 시각화 1: 전체 비교
    fig1, axes1 = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # 각도 비교
    axes1[0].plot(t, np.degrees(theta_true_wrapped), 'k-', label='참값', linewidth=2)
    axes1[0].plot(t[::40], np.degrees(theta_meas_wrapped[::40]), 'gray', 
                  marker='.', markersize=1, linestyle='', alpha=0.3, label='측정')
    axes1[0].plot(t, np.degrees(wrap_2pi(theta_cv)), 'b-', label=f'CV (Q={q_cv_large})', alpha=0.8)
    axes1[0].plot(t, np.degrees(wrap_2pi(theta_ca)), 'g-', label=f'CA (Q={q_ca_large})', alpha=0.8)
    axes1[0].set_ylabel('각도 (deg)')
    axes1[0].set_title('Wrapped 각도 비교')
    axes1[0].legend()
    axes1[0].grid(True, alpha=0.3)
    
    # 속도 비교
    axes1[1].plot(t, omega_true * 60/(2*np.pi), 'k-', label='참값', linewidth=2)
    axes1[1].plot(t, omega_cv * 60/(2*np.pi), 'b-', label='CV', alpha=0.8)
    axes1[1].plot(t, omega_ca * 60/(2*np.pi), 'g-', label='CA', alpha=0.8)
    axes1[1].set_ylabel('속도 (RPM)')
    axes1[1].set_title('속도 추정')
    axes1[1].legend()
    axes1[1].grid(True, alpha=0.3)
    
    # 가속도 비교 (CA만)
    axes1[2].plot(t, alpha_true * 60/(2*np.pi), 'k-', label='참값', linewidth=2)
    axes1[2].plot(t, alpha_ca * 60/(2*np.pi), 'g-', label='CA 추정', alpha=0.8)
    axes1[2].set_ylabel('가속도 (RPM/s)')
    axes1[2].set_title('가속도 추정 (CA만 가능)')
    axes1[2].legend()
    axes1[2].grid(True, alpha=0.3)
    
    # 에러 비교
    axes1[3].plot(t, np.degrees(err_meas), 'gray', label=f'측정에러 ({rms_meas:.2f}°)', 
                  linewidth=0.5, alpha=0.5)
    axes1[3].plot(t, np.degrees(err_cv), 'b-', label=f'CV에러 ({rms_cv:.2f}°)', linewidth=1)
    axes1[3].plot(t, np.degrees(err_ca), 'g-', label=f'CA에러 ({rms_ca:.2f}°)', linewidth=1)
    axes1[3].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes1[3].set_xlabel('시간 (s)')
    axes1[3].set_ylabel('각도 에러 (deg)')
    axes1[3].set_title('에러 비교')
    axes1[3].legend()
    axes1[3].grid(True, alpha=0.3)
    axes1[3].set_ylim([-2, 2])
    
    # 구간 배경
    for ax in axes1:
        ax.axvspan(0, accel_time, alpha=0.1, color='red')
        ax.axvspan(accel_time, total_time-accel_time, alpha=0.1, color='green')
        ax.axvspan(total_time-accel_time, total_time, alpha=0.1, color='blue')
    
    plt.suptitle('CV vs CA: 큰 Q값 성능 비교', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 시각화 2: 구간별 상세 분석
    fig2, axes2 = plt.subplots(3, 4, figsize=(16, 10))
    
    segment_data = [
        ("가속", accel_idx, 'red'),
        ("등속", const_idx, 'green'), 
        ("감속", decel_idx, 'blue')
    ]
    
    for seg_idx, (seg_name, seg_mask, color) in enumerate(segment_data):
        t_seg = t[seg_mask]
        
        # 각도
        axes2[seg_idx, 0].plot(t_seg, np.degrees(theta_true_wrapped[seg_mask]), 'k-', 
                               label='참값', linewidth=2)
        axes2[seg_idx, 0].plot(t_seg, np.degrees(wrap_2pi(theta_cv[seg_mask])), 'b-', 
                               label='CV', alpha=0.8)
        axes2[seg_idx, 0].plot(t_seg, np.degrees(wrap_2pi(theta_ca[seg_mask])), 'g-', 
                               label='CA', alpha=0.8)
        axes2[seg_idx, 0].set_ylabel('각도 (deg)')
        axes2[seg_idx, 0].set_title(f'{seg_name} - 각도')
        axes2[seg_idx, 0].legend(fontsize=8)
        axes2[seg_idx, 0].grid(True, alpha=0.3)
        
        # 속도
        axes2[seg_idx, 1].plot(t_seg, omega_true[seg_mask] * 60/(2*np.pi), 'k-', 
                               label='참값', linewidth=2)
        axes2[seg_idx, 1].plot(t_seg, omega_cv[seg_mask] * 60/(2*np.pi), 'b-', 
                               label='CV', alpha=0.8)
        axes2[seg_idx, 1].plot(t_seg, omega_ca[seg_mask] * 60/(2*np.pi), 'g-', 
                               label='CA', alpha=0.8)
        axes2[seg_idx, 1].set_ylabel('속도 (RPM)')
        axes2[seg_idx, 1].set_title(f'{seg_name} - 속도')
        axes2[seg_idx, 1].legend(fontsize=8)
        axes2[seg_idx, 1].grid(True, alpha=0.3)
        
        # 에러 비교
        err_cv_seg = err_cv[seg_mask]
        err_ca_seg = err_ca[seg_mask]
        cv_rms_seg = np.degrees(np.sqrt(np.mean(err_cv_seg**2)))
        ca_rms_seg = np.degrees(np.sqrt(np.mean(err_ca_seg**2)))
        
        axes2[seg_idx, 2].plot(t_seg, np.degrees(err_cv_seg), 'b-', 
                               label=f'CV ({cv_rms_seg:.3f}°)', linewidth=1)
        axes2[seg_idx, 2].plot(t_seg, np.degrees(err_ca_seg), 'g-', 
                               label=f'CA ({ca_rms_seg:.3f}°)', linewidth=1)
        axes2[seg_idx, 2].axhline(0, color='k', linestyle='--', alpha=0.5)
        axes2[seg_idx, 2].set_ylabel('에러 (deg)')
        axes2[seg_idx, 2].set_title(f'{seg_name} - 에러')
        axes2[seg_idx, 2].legend(fontsize=8)
        axes2[seg_idx, 2].grid(True, alpha=0.3)
        axes2[seg_idx, 2].set_ylim([-1, 1])
        
        # RMS 비교 (막대그래프)
        models = ['CV', 'CA']
        rms_values = [cv_rms_seg, ca_rms_seg]
        colors_bar = ['blue', 'green']
        bars = axes2[seg_idx, 3].bar(models, rms_values, color=colors_bar, alpha=0.7)
        axes2[seg_idx, 3].set_ylabel('RMS (deg)')
        axes2[seg_idx, 3].set_title(f'{seg_name} - RMS 비교')
        axes2[seg_idx, 3].grid(True, alpha=0.3, axis='y')
        
        # 값 표시
        for bar, val in zip(bars, rms_values):
            axes2[seg_idx, 3].text(bar.get_x() + bar.get_width()/2, val + 0.01,
                                   f'{val:.3f}°', ha='center', fontsize=9)
        
        if seg_idx == 2:  # 마지막 행
            for col in range(3):
                axes2[seg_idx, col].set_xlabel('시간 (s)')
    
    plt.suptitle('구간별 CV vs CA 상세 비교', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 최종 요약
    print(f"\n=== 최종 결론 ===")
    print(f"큰 Q값으로 CV와 CA 모두 가속도에 잘 대응!")
    if rms_cv < rms_ca:
        print(f"🏆 전체적으로 CV가 더 우수: {rms_cv:.3f}° vs {rms_ca:.3f}°")
    else:
        print(f"🏆 전체적으로 CA가 더 우수: {rms_ca:.3f}° vs {rms_cv:.3f}°")
    print(f"추가 장점 - CA: 가속도까지 추정 가능")
    
    plt.show()

if __name__ == "__main__":
    main()