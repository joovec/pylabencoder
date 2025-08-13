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

def run_linear_kalman(theta_meas_unwrap, omega_init, dt, q_value, noise_std):
    """일반적인 선형 칼만필터 (언랩 각도 입력)"""
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.F = np.array([[1.0, dt], [0.0, 1.0]])
    kf.H = np.array([[1.0, 0.0]])
    kf.Q = q_value * np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]])
    kf.R = np.array([[noise_std**2]])
    kf.P = np.diag([noise_std**2, 0.1])
    kf.x = np.array([[theta_meas_unwrap[0]], [omega_init]])
    
    theta_est = np.zeros_like(theta_meas_unwrap)
    omega_est = np.zeros_like(theta_meas_unwrap)
    
    for i in range(len(theta_meas_unwrap)):
        if i > 0:
            kf.predict()
        
        # 표준 선형 업데이트 (언랩 각도 직접 사용)
        kf.update(np.array([[theta_meas_unwrap[i]]]))
        
        theta_est[i] = kf.x[0, 0]
        omega_est[i] = kf.x[1, 0]
    
    return theta_est, omega_est

def run_circular_kalman(theta_meas_wrapped, omega_init, dt, q_value, noise_std):
    """순환 칼만필터 (래핑된 각도 입력)"""
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
    dt = 0.001  # 1kHz
    total_time = 3.0
    noise_std = np.deg2rad(0.5)
    q_value = 100.0  # 동일한 Q값
    
    print("=== 선형 칼만 vs 순환 칼만 비교 ===\n")
    print(f"동일한 각도 프로파일, 동일한 Q={q_value}")
    print(f"선형 칼만: 언랩 각도 입력 (비현실적)")
    print(f"순환 칼만: 래핑 각도 입력 (현실적)\n")
    
    # 각도 프로파일 생성
    t, theta_true_unwrap, omega_true, alpha_true = generate_trapezoid_profile(
        dt, total_time, rpm_max=1500, accel_time=0.4
    )
    
    # 추가 가속도 구간 (1.5~2.0초)
    accel_start2 = int(1.5/dt)
    accel_end2 = int(2.0/dt)
    for i in range(accel_start2, min(accel_end2, len(alpha_true))):
        alpha_true[i] = 30.0  
        omega_true[i] = omega_true[accel_start2-1] + 30.0 * (i-accel_start2+1) * dt
    
    # 각도 재계산
    theta_true_unwrap = np.cumsum(omega_true) * dt
    
    # 측정값 생성
    np.random.seed(42)
    theta_true_wrapped = wrap_2pi(theta_true_unwrap)
    
    # 선형 칼만용: 언랩된 측정값 (비현실적)
    theta_meas_unwrap = theta_true_unwrap + np.random.normal(0, noise_std, len(t))
    
    # 순환 칼만용: 래핑된 측정값 (현실적)
    np.random.seed(42)  # 동일한 노이즈 사용
    theta_meas_wrapped = wrap_2pi(theta_true_wrapped + 
                                 np.random.normal(0, noise_std, len(t)))
    
    print("필터링 실행 중...")
    
    # 선형 칼만필터 실행
    print("  선형 칼만 (언랩 입력)...")
    theta_linear, omega_linear = run_linear_kalman(
        theta_meas_unwrap, omega_true[0], dt, q_value, noise_std
    )
    
    # 순환 칼만필터 실행
    print("  순환 칼만 (래핑 입력)...")
    theta_circular, omega_circular = run_circular_kalman(
        theta_meas_wrapped, omega_true[0], dt, q_value, noise_std
    )
    
    # 에러 계산
    # 선형 칼만: 언랩 기준 에러
    err_meas_linear = theta_meas_unwrap - theta_true_unwrap
    err_linear = theta_linear - theta_true_unwrap
    
    # 순환 칼만: 래핑 기준 에러 + 언랩 기준 에러
    err_meas_circular = diffpi(theta_meas_wrapped, theta_true_wrapped)
    err_circular = theta_circular - theta_true_unwrap
    
    # RMS 계산
    rms_meas_linear = np.degrees(np.sqrt(np.mean(err_meas_linear**2)))
    rms_linear = np.degrees(np.sqrt(np.mean(err_linear**2)))
    
    rms_meas_circular = np.degrees(np.sqrt(np.mean(err_meas_circular**2)))
    rms_circular = np.degrees(np.sqrt(np.mean(err_circular**2)))
    
    print(f"\n=== 성능 비교 ===")
    print(f"선형 칼만 (언랩 입력):")
    print(f"  측정값 RMS:  {rms_meas_linear:.3f}°")
    print(f"  칼만 RMS:    {rms_linear:.3f}° (개선율: {rms_meas_linear/rms_linear:.1f}x)")
    
    print(f"\n순환 칼만 (래핑 입력):")
    print(f"  측정값 RMS:  {rms_meas_circular:.3f}°")
    print(f"  칼만 RMS:    {rms_circular:.3f}° (개선율: {rms_meas_circular/rms_circular:.1f}x)")
    
    print(f"\n직접 비교:")
    print(f"  성능 차이:   {abs(rms_linear-rms_circular):.3f}°")
    if rms_linear < rms_circular:
        print(f"  🏆 선형 칼만 승리! ({rms_circular/rms_linear:.1f}x 더 우수)")
    else:
        print(f"  🏆 순환 칼만 승리! ({rms_linear/rms_circular:.1f}x 더 우수)")
    
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
        linear_rms = np.degrees(np.sqrt(np.mean(err_linear[seg_mask]**2)))
        circular_rms = np.degrees(np.sqrt(np.mean(err_circular[seg_mask]**2)))
        winner = "선형" if linear_rms < circular_rms else "순환"
        ratio = circular_rms/linear_rms if linear_rms < circular_rms else linear_rms/circular_rms
        print(f"{seg_name}: 선형={linear_rms:.3f}°, 순환={circular_rms:.3f}° → {winner} ({ratio:.1f}x)")
    
    # 시각화
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
    
    # 1. 언랩 각도 비교
    axes[0].plot(t, np.degrees(theta_true_unwrap), 'k-', label='참값 (언랩)', linewidth=2)
    axes[0].plot(t[::50], np.degrees(theta_meas_unwrap[::50]), 'gray', 
                marker='x', markersize=2, linestyle='', alpha=0.5, label='측정 (언랩)')
    axes[0].plot(t, np.degrees(theta_linear), 'r-', label='선형 칼만', alpha=0.8)
    axes[0].plot(t, np.degrees(theta_circular), 'b-', label='순환 칼만', alpha=0.8)
    axes[0].set_ylabel('각도 (deg)')
    axes[0].set_title('언랩 각도: 선형 칼만 vs 순환 칼만')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. 래핑된 각도 비교
    axes[1].plot(t[::100], np.degrees(theta_true_wrapped[::100]), 'k-', label='참값 (래핑)', linewidth=2)
    axes[1].plot(t[::50], np.degrees(theta_meas_wrapped[::50]), 'gray', 
                marker='.', markersize=2, linestyle='', alpha=0.5, label='측정 (래핑)')
    axes[1].plot(t[::100], np.degrees(wrap_2pi(theta_linear[::100])), 'r-', 
                label='선형→래핑', alpha=0.8)
    axes[1].plot(t[::100], np.degrees(wrap_2pi(theta_circular[::100])), 'b-', 
                label='순환 칼만', alpha=0.8)
    axes[1].set_ylabel('각도 (deg)')
    axes[1].set_title('래핑된 각도: 실제 센서 출력 형태')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. 각속도 비교
    axes[2].plot(t, omega_true * 60/(2*np.pi), 'k-', label='참값', linewidth=2)
    axes[2].plot(t, omega_linear * 60/(2*np.pi), 'r-', label='선형 칼만', alpha=0.8)
    axes[2].plot(t, omega_circular * 60/(2*np.pi), 'b-', label='순환 칼만', alpha=0.8)
    axes[2].set_ylabel('각속도 (RPM)')
    axes[2].set_title('각속도 추정')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 4. 각가속도 (참값만)
    axes[3].plot(t, alpha_true * 60/(2*np.pi), 'k-', label='참값', linewidth=2)
    axes[3].set_ylabel('각가속도 (RPM/s)')
    axes[3].set_title('각가속도 프로파일 (CV 모델은 추정 불가)')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    # 5. 에러 비교
    axes[4].plot(t, np.degrees(err_meas_linear), 'lightgray', 
                label=f'선형 측정 ({rms_meas_linear:.2f}°)', linewidth=0.8, alpha=0.7)
    axes[4].plot(t, np.degrees(err_meas_circular), 'gray', 
                label=f'순환 측정 ({rms_meas_circular:.2f}°)', linewidth=0.8, alpha=0.7)
    axes[4].plot(t, np.degrees(err_linear), 'r-', 
                label=f'선형 칼만 ({rms_linear:.2f}°)', linewidth=1)
    axes[4].plot(t, np.degrees(err_circular), 'b-', 
                label=f'순환 칼만 ({rms_circular:.2f}°)', linewidth=1)
    axes[4].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[4].set_xlabel('시간 (s)')
    axes[4].set_ylabel('각도 에러 (deg)')
    axes[4].set_title('에러 비교: 선형 vs 순환')
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)
    axes[4].set_ylim([-3, 3])
    
    # 구간 표시
    colors = ['red', 'green', 'orange', 'green', 'blue']
    times = [(0, 0.4), (0.4, 1.5), (1.5, 2.0), (2.0, 2.6), (2.6, 3.0)]
    labels = ['가속1', '정속1', '가속2', '정속2', '감속']
    
    for ax in axes:
        for (start, end), color, label in zip(times, colors, labels):
            ax.axvspan(start, end, alpha=0.1, color=color)
    
    plt.suptitle('선형 칼만(언랩 입력) vs 순환 칼만(래핑 입력) 비교', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 불연속점 분석
    discontinuities = []
    for i in range(1, len(theta_meas_wrapped)):
        diff = abs(theta_meas_wrapped[i] - theta_meas_wrapped[i-1])
        if diff > np.pi:
            discontinuities.append((i, t[i]))
    
    print(f"\n=== 측정값 분석 ===")
    print(f"래핑된 측정값 불연속점: {len(discontinuities)}개")
    print(f"언랩된 측정값: 연속적 (비현실적)")
    
    # 최종 결론
    print(f"\n=== 최종 결론 ===")
    print(f"📊 성능 비교:")
    print(f"   선형 칼만: {rms_linear:.3f}° (언랩 입력)")
    print(f"   순환 칼만: {rms_circular:.3f}° (래핑 입력)")
    
    print(f"\n🔍 핵심 차이:")
    print(f"   선형 칼만: 이상적 조건 (언랩 측정값)")
    print(f"   순환 칼만: 현실적 조건 (래핑 측정값)")
    
    if rms_linear < rms_circular:
        diff = rms_circular - rms_linear
        print(f"\n✨ 선형이 {diff:.3f}° 더 우수하지만,")
        print(f"   실제로는 언랩 측정값 획득 불가능!")
        print(f"   → 순환 칼만이 유일한 현실적 솔루션")
    else:
        print(f"\n🏆 순환 칼만이 우수하면서 현실적!")
    
    print(f"\n💡 실용적 선택: 순환 칼만")
    print(f"   - 실제 센서 출력 처리 가능")
    print(f"   - 각도 불연속 올바른 처리")
    print(f"   - 물리적 의미 보존")

if __name__ == "__main__":
    main()