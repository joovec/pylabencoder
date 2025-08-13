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

def main():
    print("=== 각도 칼만필터 문제점 분석 ===\n")
    
    # 문제점 1: 언랩 방식의 오류
    print("1. 기존 방식 (언랩 상태에서 직접 처리)")
    print("   - 측정값도 언랩으로 생성: theta_meas_unwrap = theta_true_unwrap + noise")
    print("   - 칼만필터도 언랩 상태로 동작")
    print("   - 실제 센서는 wrap된 각도만 제공!")
    print("   - 이는 현실적이지 않은 가정\n")
    
    # 문제점 2: 모듈러 특성 무시
    print("2. 각도의 모듈러(Circular) 특성 무시")
    print("   - 직선 운동: 위치는 무한대까지 증가 가능")
    print("   - 회전 운동: 각도는 0~2π에서 순환")
    print("   - 359° → 1° 변화 시 Δθ = -358° (잘못) vs 2° (올바름)")
    print("   - 칼만필터의 선형 가정이 각도 불연속에서 깨짐\n")
    
    # 문제점 3: 측정 혁신(Innovation) 계산 오류
    print("3. 측정 혁신 계산 문제")
    print("   기존: innovation = z - H*x  (선형)")
    print("   각도: innovation = diffpi(z, H*x)  (순환)")
    print("   각속도/가속도는 선형으로 처리 가능\n")
    
    # 올바른 방법 제시
    print("=== 올바른 각도 칼만필터 구현 방법 ===\n")
    
    print("1. 측정값은 항상 wrap된 각도 사용")
    print("   - 실제 센서가 제공하는 형태")
    print("   - theta_meas = wrap_2pi(theta_true + noise)")
    
    print("\n2. 상태벡터는 언랩 유지 (내부 연산용)")
    print("   - [θ_unwrap, ω, α]")
    print("   - 각속도, 가속도는 선형 처리")
    
    print("\n3. 혁신 계산 시 각도만 순환 처리")
    print("   - innovation = diffpi(z_wrapped, wrap_2pi(H*x))")
    print("   - H*x는 언랩 상태에서 wrap하여 비교")
    
    print("\n4. 상태 업데이트 후 각도만 연속성 유지")
    print("   - 각도 상태만 언랩 상태로 유지")
    print("   - 각속도, 가속도는 일반 처리\n")
    
    # 비교 실험
    print("=== 비교 실험: 잘못된 방법 vs 올바른 방법 ===")
    
    dt = 0.001
    total_time = 2.0
    t = np.arange(0, total_time, dt)
    
    # 진짜 각도 (언랩)
    rpm = 1000
    omega_true = rpm * 2 * np.pi / 60
    theta_true_unwrap = omega_true * t
    theta_true_wrap = wrap_2pi(theta_true_unwrap)
    
    # 노이즈 추가
    noise_std = np.deg2rad(0.5)
    np.random.seed(42)
    
    # 방법 1: 잘못된 방법 (언랩 측정값)
    print("\n방법 1: 언랩 측정값 사용 (비현실적)")
    theta_meas_unwrap_wrong = theta_true_unwrap + np.random.normal(0, noise_std, len(t))
    
    kf1 = KalmanFilter(dim_x=2, dim_z=1)
    kf1.F = np.array([[1.0, dt], [0.0, 1.0]])
    kf1.H = np.array([[1.0, 0.0]])
    kf1.Q = 1e-3 * np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]])
    kf1.R = np.array([[noise_std**2]])
    kf1.P = np.diag([noise_std**2, 1.0])
    kf1.x = np.array([[theta_meas_unwrap_wrong[0]], [omega_true]])
    
    theta_est1 = np.zeros_like(t)
    omega_est1 = np.zeros_like(t)
    
    for i in range(len(t)):
        if i > 0:
            kf1.predict()
        kf1.update(np.array([[theta_meas_unwrap_wrong[i]]]))
        theta_est1[i] = kf1.x[0, 0]
        omega_est1[i] = kf1.x[1, 0]
    
    err1 = theta_est1 - theta_true_unwrap
    rms1 = np.degrees(np.sqrt(np.mean(err1**2)))
    print(f"RMS 에러: {rms1:.3f}° (인위적으로 좋음)")
    
    # 방법 2: 올바른 방법 (wrap된 측정값)
    print("\n방법 2: Wrap된 측정값 + 순환 혁신")
    np.random.seed(42)  # 동일한 노이즈
    theta_meas_wrap = wrap_2pi(theta_true_wrap + np.random.normal(0, noise_std, len(t)))
    
    kf2 = KalmanFilter(dim_x=2, dim_z=1)
    kf2.F = np.array([[1.0, dt], [0.0, 1.0]])
    kf2.H = np.array([[1.0, 0.0]])
    kf2.Q = 1e-1 * np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]])  # 더 큰 Q 필요
    kf2.R = np.array([[noise_std**2]])
    kf2.P = np.diag([noise_std**2, 1.0])
    kf2.x = np.array([[theta_meas_wrap[0]], [omega_true]])
    
    theta_est2 = np.zeros_like(t)
    omega_est2 = np.zeros_like(t)
    
    theta_unwrap_state = theta_meas_wrap[0]  # 언랩 상태 추적
    
    for i in range(len(t)):
        if i > 0:
            kf2.predict()
            
        # 올바른 혁신 계산
        z_wrap = theta_meas_wrap[i]
        pred_wrap = wrap_2pi(kf2.x[0, 0])
        innovation = diffpi(z_wrap, pred_wrap)
        
        # 언랩 상태 업데이트
        theta_unwrap_state = kf2.x[0, 0] + innovation
        kf2.update(np.array([[theta_unwrap_state]]))
        
        theta_est2[i] = kf2.x[0, 0]
        omega_est2[i] = kf2.x[1, 0]
    
    err2 = theta_est2 - theta_true_unwrap
    rms2 = np.degrees(np.sqrt(np.mean(err2**2)))
    print(f"RMS 에러: {rms2:.3f}° (현실적)")
    
    # 방법 3: 더 나은 순환 칼만필터
    print("\n방법 3: 개선된 순환 칼만필터")
    np.random.seed(42)
    
    kf3 = KalmanFilter(dim_x=2, dim_z=1)
    kf3.F = np.array([[1.0, dt], [0.0, 1.0]])
    kf3.H = np.array([[1.0, 0.0]])
    kf3.Q = 5e-2 * np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]])
    kf3.R = np.array([[noise_std**2]])
    kf3.P = np.diag([noise_std**2, 0.1])
    kf3.x = np.array([[theta_meas_wrap[0]], [omega_true]])
    
    theta_est3 = np.zeros_like(t)
    omega_est3 = np.zeros_like(t)
    
    for i in range(len(t)):
        if i > 0:
            kf3.predict()
        
        # 순환 혁신 직접 적용
        z = theta_meas_wrap[i]
        h_x = wrap_2pi(kf3.x[0, 0])
        y = diffpi(z, h_x)
        
        # 표준 칼만 업데이트 (혁신만 순환 처리)
        S = kf3.H @ kf3.P @ kf3.H.T + kf3.R
        K = kf3.P @ kf3.H.T / S[0, 0]
        
        kf3.x = kf3.x + K * y
        kf3.P = (np.eye(2) - K @ kf3.H) @ kf3.P
        
        theta_est3[i] = kf3.x[0, 0]
        omega_est3[i] = kf3.x[1, 0]
    
    err3 = theta_est3 - theta_true_unwrap
    rms3 = np.degrees(np.sqrt(np.mean(err3**2)))
    print(f"RMS 에러: {rms3:.3f}° (최적)")
    
    # 시각화
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # 각도 추정
    axes[0, 0].plot(t, theta_true_unwrap, 'k-', label='True (unwrap)', linewidth=2)
    axes[0, 0].plot(t, theta_est1, 'r--', label='방법1 (언랩측정)', alpha=0.7)
    axes[0, 0].plot(t, theta_est2, 'b-', label='방법2 (순환혁신)', alpha=0.7)
    axes[0, 0].plot(t, theta_est3, 'g-', label='방법3 (개선)', alpha=0.7)
    axes[0, 0].set_ylabel('Unwrapped Angle (rad)')
    axes[0, 0].set_title('언랩 각도 추정')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Wrap된 각도
    axes[0, 1].plot(t[::100], wrap_2pi(theta_true_unwrap[::100]), 'k-', label='True', linewidth=2)
    axes[0, 1].plot(t[::100], theta_meas_wrap[::100], 'gray', marker='.', markersize=2, 
                   linestyle='', alpha=0.5, label='Measured')
    axes[0, 1].plot(t[::100], wrap_2pi(theta_est2[::100]), 'b-', label='방법2', alpha=0.7)
    axes[0, 1].plot(t[::100], wrap_2pi(theta_est3[::100]), 'g-', label='방법3', alpha=0.7)
    axes[0, 1].set_ylabel('Wrapped Angle (rad)')
    axes[0, 1].set_title('Wrap된 각도 (실제 측정)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 속도 추정
    axes[1, 0].plot(t, [omega_true * 60/(2*np.pi)]*len(t), 'k--', label='True RPM', linewidth=2)
    axes[1, 0].plot(t, omega_est1 * 60/(2*np.pi), 'r--', label='방법1', alpha=0.7)
    axes[1, 0].plot(t, omega_est2 * 60/(2*np.pi), 'b-', label='방법2', alpha=0.7)
    axes[1, 0].plot(t, omega_est3 * 60/(2*np.pi), 'g-', label='방법3', alpha=0.7)
    axes[1, 0].set_ylabel('RPM')
    axes[1, 0].set_title('각속도 추정')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 에러 비교
    axes[1, 1].plot(t, np.degrees(err1), 'r--', label=f'방법1 ({rms1:.2f}°)', alpha=0.7)
    axes[1, 1].plot(t, np.degrees(err2), 'b-', label=f'방법2 ({rms2:.2f}°)', alpha=0.7)
    axes[1, 1].plot(t, np.degrees(err3), 'g-', label=f'방법3 ({rms3:.2f}°)', alpha=0.7)
    axes[1, 1].set_ylabel('Error (deg)')
    axes[1, 1].set_title('각도 에러')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([-5, 5])
    
    # 혁신 비교 (마지막 200 포인트)
    axes[2, 0].plot(t[-200:], np.degrees(diffpi(theta_meas_wrap[-200:], 
                   wrap_2pi(theta_true_unwrap[-200:]))), 
                   'gray', label='측정-참값', alpha=0.5)
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Innovation (deg)')
    axes[2, 0].set_title('측정 혁신 (마지막 0.2초)')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # 불연속점 분석
    discontinuities = []
    for i in range(1, len(theta_meas_wrap)):
        diff = abs(theta_meas_wrap[i] - theta_meas_wrap[i-1])
        if diff > np.pi:  # 불연속 검출
            discontinuities.append((i, t[i]))
    
    axes[2, 1].plot(t[::10], theta_meas_wrap[::10], 'b.', markersize=2, alpha=0.7, label='Measured')
    for disc_i, disc_t in discontinuities:
        axes[2, 1].axvline(disc_t, color='r', linestyle='--', alpha=0.5)
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Wrapped Angle (rad)')
    axes[2, 1].set_title(f'불연속점 분석 ({len(discontinuities)}개)')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.suptitle('각도 칼만필터 문제점 분석 및 해결방안', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== 결론 ===")
    print(f"방법 1 (언랩 측정): {rms1:.3f}° - 비현실적이지만 좋은 성능")
    print(f"방법 2 (순환 혁신): {rms2:.3f}° - 현실적이지만 성능 저하")
    print(f"방법 3 (개선):      {rms3:.3f}° - 현실적이면서 향상된 성능")
    print(f"\n핵심: 각도의 순환 특성을 혁신 계산에만 적용하고,")
    print(f"      상태벡터는 언랩으로 유지하여 연속성 확보!")

if __name__ == "__main__":
    main()