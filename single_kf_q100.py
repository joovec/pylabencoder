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

def run_single_circular_kf(theta_meas_wrapped, omega_init, dt, q_value, noise_std):
    """단일 순환 칼만필터 (CV 모델)"""
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
    # 파라미터 (적응형 필터와 동일)
    dt = 0.001  # 1kHz
    total_time = 3.0
    noise_std = np.deg2rad(0.5)
    q_uniform = 100.0  # 균일 Q값
    
    print("=== 단일 KF vs 적응형 계층 KF 비교 ===\n")
    print(f"단일 KF: Q={q_uniform} (균일)")
    print(f"적응형 KF: Q=50(정속), Q=100(가속)\n")
    
    # 동일한 프로파일 생성 (적응형과 동일)
    t, theta_true_unwrap, omega_true, alpha_true = generate_trapezoid_profile(
        dt, total_time, rpm_max=1500, accel_time=0.4
    )
    
    # 추가 가속도 구간 (1.5~2.0초) - 적응형과 동일
    accel_start2 = int(1.5/dt)
    accel_end2 = int(2.0/dt)
    for i in range(accel_start2, min(accel_end2, len(alpha_true))):
        alpha_true[i] = 30.0  # 추가 가속
        omega_true[i] = omega_true[accel_start2-1] + 30.0 * (i-accel_start2+1) * dt
    
    # 각도 재계산
    theta_true_unwrap = np.cumsum(omega_true) * dt
    
    # 측정값 생성 (적응형과 동일)
    np.random.seed(42)
    theta_true_wrapped = wrap_2pi(theta_true_unwrap)
    theta_meas_wrapped = wrap_2pi(theta_true_wrapped + 
                                 np.random.normal(0, noise_std, len(t)))
    
    # 단일 칼만필터 실행
    print("단일 KF (Q=100) 실행 중...")
    theta_single, omega_single = run_single_circular_kf(
        theta_meas_wrapped, omega_true[0], dt, q_uniform, noise_std
    )
    
    # 비교를 위해 적응형도 재실행 (동일 조건)
    from adaptive_cascaded_kalman import AdaptiveCascadedKalman
    
    print("적응형 계층 KF 재실행 중...")
    akf = AdaptiveCascadedKalman(dt, noise_std)
    akf.initialize(theta_meas_wrapped[0], omega_true[0])
    
    theta_adaptive = np.zeros_like(t)
    omega_adaptive = np.zeros_like(t)
    alpha_adaptive = np.zeros_like(t)
    q1_history = np.zeros_like(t)
    
    for i in range(len(t)):
        theta_adaptive[i], omega_adaptive[i], alpha_adaptive[i], q1_current, mode = \
            akf.predict_and_update(theta_meas_wrapped[i])
        q1_history[i] = q1_current
    
    # 에러 계산
    err_meas = diffpi(theta_meas_wrapped, theta_true_wrapped)
    err_single = theta_single - theta_true_unwrap
    err_adaptive = theta_adaptive - theta_true_unwrap
    
    rms_meas = np.degrees(np.sqrt(np.mean(err_meas**2)))
    rms_single = np.degrees(np.sqrt(np.mean(err_single**2)))
    rms_adaptive = np.degrees(np.sqrt(np.mean(err_adaptive**2)))
    
    print(f"\n=== 성능 비교 ===")
    print(f"측정값 RMS:    {rms_meas:.3f}°")
    print(f"단일 KF RMS:   {rms_single:.3f}° (개선율: {rms_meas/rms_single:.1f}x)")
    print(f"적응형 RMS:    {rms_adaptive:.3f}° (개선율: {rms_meas/rms_adaptive:.1f}x)")
    print(f"\n성능 차이:     {abs(rms_single-rms_adaptive):.3f}° 차이")
    if rms_single < rms_adaptive:
        print(f"🏆 단일 KF 승리! ({rms_adaptive/rms_single:.1f}x 더 우수)")
    else:
        print(f"🏆 적응형 승리! ({rms_single/rms_adaptive:.1f}x 더 우수)")
    
    # 구간별 분석
    print(f"\n=== 구간별 성능 비교 ===")
    segments = [
        ("가속1 (0-0.4s)", (t <= 0.4)),
        ("정속1 (0.4-1.5s)", (t > 0.4) & (t <= 1.5)),
        ("가속2 (1.5-2.0s)", (t > 1.5) & (t <= 2.0)),
        ("정속2 (2.0-2.6s)", (t > 2.0) & (t <= 2.6)),
        ("감속 (2.6-3.0s)", (t > 2.6))
    ]
    
    for seg_name, seg_mask in segments:
        single_rms = np.degrees(np.sqrt(np.mean(err_single[seg_mask]**2)))
        adaptive_rms = np.degrees(np.sqrt(np.mean(err_adaptive[seg_mask]**2)))
        winner = "단일" if single_rms < adaptive_rms else "적응형"
        ratio = adaptive_rms/single_rms if single_rms < adaptive_rms else single_rms/adaptive_rms
        print(f"{seg_name}: 단일={single_rms:.3f}°, 적응형={adaptive_rms:.3f}° → {winner} ({ratio:.1f}x)")
    
    # 시각화
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
    
    # 1. 각도 비교
    axes[0].plot(t, np.degrees(theta_true_wrapped), 'k-', label='참값', linewidth=2)
    axes[0].plot(t[::50], np.degrees(theta_meas_wrapped[::50]), 'gray', 
                marker='.', markersize=1, linestyle='', alpha=0.3, label='측정')
    axes[0].plot(t, np.degrees(wrap_2pi(theta_single)), 'r-', label='단일 KF', alpha=0.8)
    axes[0].plot(t, np.degrees(wrap_2pi(theta_adaptive)), 'b-', label='적응형 KF', alpha=0.8)
    axes[0].set_ylabel('각도 (deg)')
    axes[0].set_title('단일 KF vs 적응형 계층 KF: 각도 추정')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. 각속도 비교
    axes[1].plot(t, omega_true * 60/(2*np.pi), 'k-', label='참값', linewidth=2)
    axes[1].plot(t, omega_single * 60/(2*np.pi), 'r-', label='단일 KF', alpha=0.8)
    axes[1].plot(t, omega_adaptive * 60/(2*np.pi), 'b-', label='적응형 KF', alpha=0.8)
    axes[1].set_ylabel('각속도 (RPM)')
    axes[1].set_title('각속도 추정')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. 각가속도 (참값만, 단일은 추정 불가)
    axes[2].plot(t, alpha_true * 60/(2*np.pi), 'k-', label='참값', linewidth=2)
    axes[2].plot(t, alpha_adaptive * 60/(2*np.pi), 'b-', label='적응형 추정', alpha=0.8)
    axes[2].text(0.5, 0.9, '단일 KF: 가속도 추정 불가', transform=axes[2].transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    axes[2].set_ylabel('각가속도 (RPM/s)')
    axes[2].set_title('각가속도: 적응형만 추정 가능')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 4. Q값 비교
    axes[3].plot(t, [q_uniform]*len(t), 'r-', label=f'단일 KF (Q={q_uniform})', linewidth=2)
    axes[3].plot(t, q1_history, 'b-', label='적응형 KF (Q 가변)', linewidth=2)
    axes[3].axhline(50, color='b', linestyle='--', alpha=0.5, label='정속 Q=50')
    axes[3].axhline(100, color='b', linestyle='--', alpha=0.5, label='가속 Q=100')
    axes[3].set_ylabel('Q1 값')
    axes[3].set_title('Q값 비교: 단일(고정) vs 적응형(가변)')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    # 5. 에러 비교
    axes[4].plot(t, np.degrees(err_meas), 'gray', label=f'측정 ({rms_meas:.2f}°)', 
                linewidth=0.5, alpha=0.5)
    axes[4].plot(t, np.degrees(err_single), 'r-', label=f'단일 KF ({rms_single:.2f}°)', linewidth=1)
    axes[4].plot(t, np.degrees(err_adaptive), 'b-', label=f'적응형 KF ({rms_adaptive:.2f}°)', linewidth=1)
    axes[4].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[4].set_xlabel('시간 (s)')
    axes[4].set_ylabel('각도 에러 (deg)')
    axes[4].set_title('에러 비교')
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)
    axes[4].set_ylim([-2, 2])
    
    # 구간 표시
    colors = ['red', 'green', 'orange', 'green', 'blue']
    times = [(0, 0.4), (0.4, 1.5), (1.5, 2.0), (2.0, 2.6), (2.6, 3.0)]
    labels = ['가속1', '정속1', '가속2', '정속2', '감속']
    
    for ax in axes:
        for (start, end), color, label in zip(times, colors, labels):
            ax.axvspan(start, end, alpha=0.1, color=color)
    
    plt.suptitle('단일 KF(Q=100) vs 적응형 계층 KF 성능 비교', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 최종 결론
    print(f"\n=== 최종 결론 ===")
    if rms_single < rms_adaptive:
        print(f"✨ 단일 KF(Q=100)가 더 우수!")
        print(f"   - 구현 간단, 계산 빠름")
        print(f"   - Q=100이 모든 구간에 적합")
    else:
        print(f"✨ 적응형 계층 KF가 더 우수!")
        print(f"   - 상황별 최적 Q 적용")
        print(f"   - 가속도 정보 추가 제공")
    
    print(f"\n각각의 장점:")
    print(f"단일 KF:   간단, 빠름, 안정적")
    print(f"적응형 KF: 지능적, 정보 풍부, 상황 인식")

if __name__ == "__main__":
    main()