import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import ExtendedKalmanFilter
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

class SinCosEKF:
    """sin/cos 분해를 이용한 EKF (CV 모델)"""
    
    def __init__(self, dt, q_value, noise_std):
        self.dt = dt
        self.ekf = ExtendedKalmanFilter(dim_x=3, dim_z=2)
        
        # 상태: [sin(θ), cos(θ), ω]
        # F 행렬 (비선형이므로 매 스텝 계산)
        self.ekf.F = np.eye(3)
        
        # H 행렬 (측정 = [sin(θ), cos(θ)])
        self.ekf.H = np.array([[1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0]])
        
        # Q 행렬 (sin, cos, omega)
        self.ekf.Q = np.diag([q_value * dt**3/3, q_value * dt**3/3, q_value * dt])
        
        # R 행렬 (sin, cos 측정 노이즈)
        self.ekf.R = np.eye(2) * noise_std**2
        
        # P 초기값
        self.ekf.P = np.diag([noise_std**2, noise_std**2, 0.1])
    
    def initialize(self, theta_init, omega_init):
        """초기화"""
        self.ekf.x = np.array([np.sin(theta_init), np.cos(theta_init), omega_init])
    
    def f(self, x, dt):
        """상태 전이 함수 (비선형)"""
        sin_th, cos_th, omega = x
        
        # 각도 변화
        d_theta = omega * dt
        
        # 회전 행렬 적용
        new_sin = sin_th * np.cos(d_theta) + cos_th * np.sin(d_theta)
        new_cos = cos_th * np.cos(d_theta) - sin_th * np.sin(d_theta)
        new_omega = omega
        
        return np.array([new_sin, new_cos, new_omega])
    
    def F_jacobian(self, x, dt):
        """F 행렬의 야코비안"""
        sin_th, cos_th, omega = x
        d_theta = omega * dt
        
        cos_dth = np.cos(d_theta)
        sin_dth = np.sin(d_theta)
        
        # ∂f/∂x
        F = np.array([
            [cos_dth, sin_dth, dt * (cos_th * cos_dth - sin_th * sin_dth)],
            [-sin_dth, cos_dth, dt * (-sin_th * cos_dth - cos_th * sin_dth)],
            [0.0, 0.0, 1.0]
        ])
        
        return F
    
    def h(self, x):
        """측정 함수"""
        return np.array([x[0], x[1]])  # [sin(θ), cos(θ)]
    
    def H_jacobian(self, x):
        """H 행렬의 야코비안"""
        return self.ekf.H
    
    def predict_and_update(self, z_sin, z_cos):
        """예측 및 업데이트"""
        # 야코비안 계산
        F = self.F_jacobian(self.ekf.x, self.dt)
        
        # 예측
        self.ekf.x = self.f(self.ekf.x, self.dt)
        self.ekf.P = F @ self.ekf.P @ F.T + self.ekf.Q
        
        # sin²+cos²=1 제약 조건 (정규화)
        norm = np.sqrt(self.ekf.x[0]**2 + self.ekf.x[1]**2)
        if norm > 0:
            self.ekf.x[0] /= norm
            self.ekf.x[1] /= norm
        
        # 업데이트
        z = np.array([z_sin, z_cos])
        y = z - self.h(self.ekf.x)  # 잔차
        
        S = self.ekf.H @ self.ekf.P @ self.ekf.H.T + self.ekf.R
        K = self.ekf.P @ self.ekf.H.T @ np.linalg.inv(S)
        
        self.ekf.x = self.ekf.x + K @ y
        self.ekf.P = (np.eye(3) - K @ self.ekf.H) @ self.ekf.P
        
        # 다시 정규화
        norm = np.sqrt(self.ekf.x[0]**2 + self.ekf.x[1]**2)
        if norm > 0:
            self.ekf.x[0] /= norm
            self.ekf.x[1] /= norm
        
        # 각도 복원
        theta_est = np.arctan2(self.ekf.x[0], self.ekf.x[1])
        omega_est = self.ekf.x[2]
        
        return theta_est, omega_est

def run_circular_kalman(theta_meas_wrapped, omega_init, dt, q_value, noise_std):
    """순환 칼만필터 (비교용)"""
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
    q_value = 100.0
    
    print("=== Sin/Cos EKF vs 순환 칼만 비교 ===\n")
    print(f"EKF: 상태=[sin(θ), cos(θ), ω], 측정=[sin(θ), cos(θ)]")
    print(f"순환: 상태=[θ, ω], 측정=[θ] (순환 혁신)")
    print(f"동일한 Q={q_value}, 노이즈={np.degrees(noise_std):.1f}°\n")
    
    # 각도 프로파일 생성
    t, theta_true_unwrap, omega_true, alpha_true = generate_trapezoid_profile(
        dt, total_time, rpm_max=1500, accel_time=0.4
    )
    
    # 추가 가속도 구간
    accel_start2 = int(1.5/dt)
    accel_end2 = int(2.0/dt)
    for i in range(accel_start2, min(accel_end2, len(alpha_true))):
        alpha_true[i] = 30.0
        omega_true[i] = omega_true[accel_start2-1] + 30.0 * (i-accel_start2+1) * dt
    
    theta_true_unwrap = np.cumsum(omega_true) * dt
    theta_true_wrapped = wrap_2pi(theta_true_unwrap)
    
    # 측정값 생성
    np.random.seed(42)
    
    # 순환 칼만용: 각도 측정값
    theta_meas_wrapped = wrap_2pi(theta_true_wrapped + 
                                 np.random.normal(0, noise_std, len(t)))
    
    # EKF용: sin/cos 측정값
    np.random.seed(42)  # 동일한 노이즈
    theta_noisy = theta_true_wrapped + np.random.normal(0, noise_std, len(t))
    sin_meas = np.sin(theta_noisy)
    cos_meas = np.cos(theta_noisy)
    
    print("필터링 실행 중...")
    
    # Sin/Cos EKF 실행
    print("  Sin/Cos EKF...")
    ekf = SinCosEKF(dt, q_value, noise_std)
    ekf.initialize(theta_true_wrapped[0], omega_true[0])
    
    theta_ekf = np.zeros_like(t)
    omega_ekf = np.zeros_like(t)
    
    for i in range(len(t)):
        theta_ekf[i], omega_ekf[i] = ekf.predict_and_update(sin_meas[i], cos_meas[i])
    
    # 순환 칼만 실행
    print("  순환 칼만...")
    theta_circular, omega_circular = run_circular_kalman(
        theta_meas_wrapped, omega_true[0], dt, q_value, noise_std
    )
    
    # 에러 계산
    err_meas = diffpi(theta_meas_wrapped, theta_true_wrapped)
    err_ekf = theta_ekf - theta_true_unwrap
    err_circular = theta_circular - theta_true_unwrap
    
    # 순환 에러도 계산 (래핑된 각도 기준)
    err_ekf_circular = diffpi(wrap_2pi(theta_ekf), theta_true_wrapped)
    err_circular_circular = diffpi(wrap_2pi(theta_circular), theta_true_wrapped)
    
    # RMS 계산
    rms_meas = np.degrees(np.sqrt(np.mean(err_meas**2)))
    rms_ekf = np.degrees(np.sqrt(np.mean(err_ekf**2)))
    rms_circular = np.degrees(np.sqrt(np.mean(err_circular**2)))
    
    rms_ekf_circ = np.degrees(np.sqrt(np.mean(err_ekf_circular**2)))
    rms_circular_circ = np.degrees(np.sqrt(np.mean(err_circular_circular**2)))
    
    print(f"\n=== 성능 비교 (언랩 기준) ===")
    print(f"측정값 RMS:    {rms_meas:.3f}°")
    print(f"Sin/Cos EKF:   {rms_ekf:.3f}° (개선율: {rms_meas/rms_ekf:.1f}x)")
    print(f"순환 칼만:     {rms_circular:.3f}° (개선율: {rms_meas/rms_circular:.1f}x)")
    
    print(f"\n=== 성능 비교 (순환 기준) ===")
    print(f"측정값 RMS:    {rms_meas:.3f}°")
    print(f"Sin/Cos EKF:   {rms_ekf_circ:.3f}°")
    print(f"순환 칼만:     {rms_circular_circ:.3f}°")
    
    print(f"\n직접 비교:")
    if rms_ekf < rms_circular:
        print(f"🏆 Sin/Cos EKF 승리! ({rms_circular/rms_ekf:.1f}x 더 우수)")
    else:
        print(f"🏆 순환 칼만 승리! ({rms_ekf/rms_circular:.1f}x 더 우수)")
    
    # 구간별 분석
    print(f"\n=== 구간별 성능 비교 (언랩 기준) ===")
    segments = [
        ("가속1 (0-0.4s)", (t <= 0.4)),
        ("정속1 (0.4-1.5s)", (t > 0.4) & (t <= 1.5)),
        ("가속2 (1.5-2.0s)", (t > 1.5) & (t <= 2.0)),
        ("정속2 (2.0-2.6s)", (t > 2.0) & (t <= 2.6)),
        ("감속 (2.6-3.0s)", (t > 2.6))
    ]
    
    for seg_name, seg_mask in segments:
        ekf_rms = np.degrees(np.sqrt(np.mean(err_ekf[seg_mask]**2)))
        circular_rms = np.degrees(np.sqrt(np.mean(err_circular[seg_mask]**2)))
        winner = "EKF" if ekf_rms < circular_rms else "순환"
        ratio = circular_rms/ekf_rms if ekf_rms < circular_rms else ekf_rms/circular_rms
        print(f"{seg_name}: EKF={ekf_rms:.3f}°, 순환={circular_rms:.3f}° → {winner} ({ratio:.1f}x)")
    
    # 시각화
    fig, axes = plt.subplots(6, 1, figsize=(14, 18), sharex=True)
    
    # 1. 언랩 각도
    axes[0].plot(t, np.degrees(theta_true_unwrap), 'k-', label='참값', linewidth=2)
    axes[0].plot(t, np.degrees(theta_ekf), 'r-', label='Sin/Cos EKF', alpha=0.8)
    axes[0].plot(t, np.degrees(theta_circular), 'b-', label='순환 칼만', alpha=0.8)
    axes[0].set_ylabel('언랩 각도 (deg)')
    axes[0].set_title('언랩 각도: Sin/Cos EKF vs 순환 칼만')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. 래핑된 각도
    axes[1].plot(t[::100], np.degrees(theta_true_wrapped[::100]), 'k-', label='참값', linewidth=2)
    axes[1].plot(t[::50], np.degrees(theta_meas_wrapped[::50]), 'gray', 
                marker='.', markersize=1, linestyle='', alpha=0.3, label='측정')
    axes[1].plot(t[::100], np.degrees(wrap_2pi(theta_ekf[::100])), 'r-', label='EKF', alpha=0.8)
    axes[1].plot(t[::100], np.degrees(wrap_2pi(theta_circular[::100])), 'b-', label='순환', alpha=0.8)
    axes[1].set_ylabel('래핑 각도 (deg)')
    axes[1].set_title('래핑된 각도')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Sin/Cos 측정값
    axes[2].plot(t[::100], sin_meas[::100], 'r-', label='sin(θ) 측정', alpha=0.7)
    axes[2].plot(t[::100], cos_meas[::100], 'b-', label='cos(θ) 측정', alpha=0.7)
    axes[2].plot(t[::100], np.sin(theta_true_wrapped[::100]), 'k--', label='sin(θ) 참값', alpha=0.5)
    axes[2].plot(t[::100], np.cos(theta_true_wrapped[::100]), 'k:', label='cos(θ) 참값', alpha=0.5)
    axes[2].set_ylabel('Sin/Cos')
    axes[2].set_title('EKF 입력: Sin/Cos 측정값')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([-1.2, 1.2])
    
    # 4. 각속도
    axes[3].plot(t, omega_true * 60/(2*np.pi), 'k-', label='참값', linewidth=2)
    axes[3].plot(t, omega_ekf * 60/(2*np.pi), 'r-', label='Sin/Cos EKF', alpha=0.8)
    axes[3].plot(t, omega_circular * 60/(2*np.pi), 'b-', label='순환 칼만', alpha=0.8)
    axes[3].set_ylabel('각속도 (RPM)')
    axes[3].set_title('각속도 추정')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    # 5. 단위원 제약 확인 (EKF)
    sin_ekf = np.sin(theta_ekf)
    cos_ekf = np.cos(theta_ekf)
    unit_circle_error = sin_ekf**2 + cos_ekf**2 - 1
    axes[4].plot(t, unit_circle_error, 'r-', label='sin²+cos²-1')
    axes[4].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[4].set_ylabel('단위원 에러')
    axes[4].set_title('EKF 단위원 제약 조건: sin²(θ)+cos²(θ)=1')
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)
    axes[4].set_ylim([-0.01, 0.01])
    
    # 6. 에러 비교
    axes[5].plot(t, np.degrees(err_meas), 'gray', label=f'측정 ({rms_meas:.2f}°)', 
                linewidth=0.5, alpha=0.5)
    axes[5].plot(t, np.degrees(err_ekf), 'r-', label=f'EKF ({rms_ekf:.2f}°)', linewidth=1)
    axes[5].plot(t, np.degrees(err_circular), 'b-', label=f'순환 ({rms_circular:.2f}°)', linewidth=1)
    axes[5].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[5].set_xlabel('시간 (s)')
    axes[5].set_ylabel('각도 에러 (deg)')
    axes[5].set_title('에러 비교')
    axes[5].legend()
    axes[5].grid(True, alpha=0.3)
    axes[5].set_ylim([-3, 3])
    
    # 구간 표시
    colors = ['red', 'green', 'orange', 'green', 'blue']
    times = [(0, 0.4), (0.4, 1.5), (1.5, 2.0), (2.0, 2.6), (2.6, 3.0)]
    
    for ax in axes:
        for (start, end), color in zip(times, colors):
            ax.axvspan(start, end, alpha=0.1, color=color)
    
    plt.suptitle('Sin/Cos EKF vs 순환 칼만 성능 비교', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 최종 결론
    print(f"\n=== 최종 결론 ===")
    print(f"📊 성능:")
    print(f"   Sin/Cos EKF: {rms_ekf:.3f}° (언랩 기준)")
    print(f"   순환 칼만:   {rms_circular:.3f}° (언랩 기준)")
    
    print(f"\n🔍 특징:")
    print(f"   EKF: 단위원 제약, 비선형 처리, sin/cos 측정")
    print(f"   순환: 순환 혁신, 선형 처리, 각도 측정")
    
    print(f"\n💡 실용성:")
    if rms_ekf < rms_circular:
        print(f"   EKF가 더 우수하지만 복잡함")
    else:
        print(f"   순환 칼만이 간단하면서 우수함")
    
    print(f"\n✨ 권장:")
    print(f"   일반적: 순환 칼만 (간단, 효율적)")
    print(f"   특수한 경우: EKF (고정밀 요구시)")

if __name__ == "__main__":
    main()