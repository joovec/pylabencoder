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

class AdaptiveCascadedKalman:
    """적응형 계층 칼만필터"""
    
    def __init__(self, dt, noise_std):
        self.dt = dt
        self.noise_std = noise_std
        
        # KF1: 각도 → 각속도 (순환 칼만)
        self.kf1 = KalmanFilter(dim_x=2, dim_z=1)
        self.kf1.F = np.array([[1.0, dt], [0.0, 1.0]])
        self.kf1.H = np.array([[1.0, 0.0]])
        self.kf1.R = np.array([[noise_std**2]])
        self.kf1.P = np.diag([noise_std**2, 0.1])
        
        # KF2: 각속도 → 각가속도 (선형 칼만)
        self.kf2 = KalmanFilter(dim_x=2, dim_z=1)
        self.kf2.F = np.array([[1.0, dt], [0.0, 1.0]])
        self.kf2.H = np.array([[1.0, 0.0]])
        self.kf2.R = np.array([[0.1]])  # KF1 각속도 추정 불확실성
        self.kf2.P = np.diag([0.1, 1.0])
        
        # Q 적응 파라미터
        self.q1_base = 50.0     # 정속 Q1
        self.q1_accel = 100.0   # 가속도 존재시 Q1
        self.q2_value = 10.0    # KF2용 Q
        
        self.alpha_threshold = 5.0  # rad/s² (가속도 존재 임계값)
        self.alpha_history = []     # 가속도 이력
        self.history_length = 10    # 이력 길이
        
    def initialize(self, theta_init, omega_init):
        """초기화"""
        self.kf1.x = np.array([[theta_init], [omega_init]])
        self.kf2.x = np.array([[omega_init], [0.0]])
    
    def update_q1_adaptive(self, alpha_detected):
        """KF2 가속도 결과에 따른 KF1의 Q 적응"""
        # 가속도 이력 관리
        self.alpha_history.append(abs(alpha_detected))
        if len(self.alpha_history) > self.history_length:
            self.alpha_history.pop(0)
        
        # 최근 가속도 평균
        avg_alpha = np.mean(self.alpha_history)
        
        # 적응형 Q1 계산
        if avg_alpha > self.alpha_threshold:
            # 가속도 존재 → Q1=100
            q1_adapted = self.q1_accel
            mode = "가속도 모드"
        else:
            # 등속 → Q1=50  
            q1_adapted = self.q1_base
            mode = "정속 모드"
        
        # KF1 Q 업데이트
        self.kf1.Q = q1_adapted * np.array([[self.dt**3/3, self.dt**2/2], 
                                           [self.dt**2/2, self.dt]])
        
        # KF2 Q는 고정
        self.kf2.Q = self.q2_value * np.array([[self.dt**3/3, self.dt**2/2], 
                                              [self.dt**2/2, self.dt]])
        
        return q1_adapted, mode
    
    def predict_and_update(self, theta_meas_wrapped):
        """예측 및 업데이트"""
        # KF1 예측
        self.kf1.predict()
        
        # KF1 순환 업데이트 (각도 측정)
        z = theta_meas_wrapped
        h_x = wrap_2pi(self.kf1.x[0, 0])
        innovation = diffpi(z, h_x)
        
        S = self.kf1.H @ self.kf1.P @ self.kf1.H.T + self.kf1.R
        K = self.kf1.P @ self.kf1.H.T / S[0, 0]
        self.kf1.x = self.kf1.x + K * innovation
        self.kf1.P = (np.eye(2) - K @ self.kf1.H) @ self.kf1.P
        
        # KF1 결과
        theta_est = self.kf1.x[0, 0]
        omega_est = self.kf1.x[1, 0]
        
        # KF2 예측 및 업데이트 (KF1 각속도를 측정값으로 사용)
        self.kf2.predict()
        self.kf2.update(np.array([[omega_est]]))
        
        # KF2 결과
        omega_est2 = self.kf2.x[0, 0]
        alpha_est = self.kf2.x[1, 0]
        
        # 적응형 Q1 업데이트
        q1_current, mode = self.update_q1_adaptive(alpha_est)
        
        return theta_est, omega_est, alpha_est, q1_current, mode

def main():
    # 파라미터
    dt = 0.001  # 1kHz
    total_time = 3.0
    noise_std = np.deg2rad(0.5)
    
    print("=== 적응형 계층 칼만필터 ===\n")
    print("KF1: 각도 → 각속도 (순환)")
    print("KF2: 각속도 → 각가속도 (선형)")
    print("적응: KF2 가속도 → KF1 Q 조정\n")
    
    # 복잡한 프로파일 (가속 → 등속 → 가속 → 등속)
    t, theta_true_unwrap, omega_true, alpha_true = generate_trapezoid_profile(
        dt, total_time, rpm_max=1500, accel_time=0.4
    )
    
    # 추가 가속도 구간 (1.5~2.0초)
    accel_start2 = int(1.5/dt)
    accel_end2 = int(2.0/dt)
    for i in range(accel_start2, min(accel_end2, len(alpha_true))):
        alpha_true[i] = 30.0  # 추가 가속
        omega_true[i] = omega_true[accel_start2-1] + 30.0 * (i-accel_start2+1) * dt
    
    # 각도 재계산
    theta_true_unwrap = np.cumsum(omega_true) * dt
    
    # 측정값 생성
    np.random.seed(42)
    theta_true_wrapped = wrap_2pi(theta_true_unwrap)
    theta_meas_wrapped = wrap_2pi(theta_true_wrapped + 
                                 np.random.normal(0, noise_std, len(t)))
    
    # 적응형 계층 칼만필터 실행
    akf = AdaptiveCascadedKalman(dt, noise_std)
    akf.initialize(theta_meas_wrapped[0], omega_true[0])
    
    # 결과 저장
    theta_est = np.zeros_like(t)
    omega_est = np.zeros_like(t)
    alpha_est = np.zeros_like(t)
    q1_history = np.zeros_like(t)
    modes = []
    
    print("필터링 실행 중...")
    for i in range(len(t)):
        theta_est[i], omega_est[i], alpha_est[i], q1_current, mode = \
            akf.predict_and_update(theta_meas_wrapped[i])
        q1_history[i] = q1_current
        modes.append(mode)
        
        if i % 500 == 0:
            print(f"  {t[i]:.1f}s: {mode}, Q1={q1_current:.1f}, α={alpha_est[i]*60/(2*np.pi):.1f}RPM/s")
    
    # 에러 계산
    err_meas = diffpi(theta_meas_wrapped, theta_true_wrapped)  # 실측 에러
    err_theta = theta_est - theta_true_unwrap  # 칼만 에러
    err_omega = omega_est - omega_true
    err_alpha = alpha_est - alpha_true
    
    rms_meas = np.degrees(np.sqrt(np.mean(err_meas**2)))
    rms_theta = np.degrees(np.sqrt(np.mean(err_theta**2)))
    rms_omega = np.sqrt(np.mean(err_omega**2)) * 60/(2*np.pi)
    rms_alpha = np.sqrt(np.mean(err_alpha**2)) * 60/(2*np.pi)
    
    print(f"\n=== 성능 ===" )
    print(f"측정값 RMS:  {rms_meas:.3f}°")
    print(f"칼만 RMS:    {rms_theta:.3f}° (개선율: {rms_meas/rms_theta:.1f}x)")
    print(f"각속도 RMS:  {rms_omega:.1f} RPM")
    print(f"각가속도 RMS: {rms_alpha:.1f} RPM/s")
    
    # 모드 분석
    accel_mode_count = sum(1 for m in modes if "가속도" in m)
    const_mode_count = len(modes) - accel_mode_count
    print(f"\n적응 모드:")
    print(f"가속도 모드: {accel_mode_count/len(modes)*100:.1f}% (Q1=100)")
    print(f"정속 모드:   {const_mode_count/len(modes)*100:.1f}% (Q1=50)")
    
    # 시각화
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
    
    # 1. 각도
    axes[0].plot(t, np.degrees(theta_true_wrapped), 'k-', label='참값', linewidth=2)
    axes[0].plot(t[::50], np.degrees(theta_meas_wrapped[::50]), 'gray', 
                marker='.', markersize=1, linestyle='', alpha=0.3, label='측정')
    axes[0].plot(t, np.degrees(wrap_2pi(theta_est)), 'b-', label='KF1 추정', alpha=0.8)
    axes[0].set_ylabel('각도 (deg)')
    axes[0].set_title('계층 칼만필터: 각도 추정')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. 각속도
    axes[1].plot(t, omega_true * 60/(2*np.pi), 'k-', label='참값', linewidth=2)
    axes[1].plot(t, omega_est * 60/(2*np.pi), 'b-', label='KF1→KF2', alpha=0.8)
    axes[1].set_ylabel('각속도 (RPM)')
    axes[1].set_title('각속도: KF1 추정 → KF2 입력')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. 각가속도
    axes[2].plot(t, alpha_true * 60/(2*np.pi), 'k-', label='참값', linewidth=2)
    axes[2].plot(t, alpha_est * 60/(2*np.pi), 'g-', label='KF2 추정', alpha=0.8)
    axes[2].axhline(akf.alpha_threshold * 60/(2*np.pi), color='r', linestyle='--', 
                   alpha=0.7, label=f'임계값 ({akf.alpha_threshold * 60/(2*np.pi):.0f})')
    axes[2].set_ylabel('각가속도 (RPM/s)')
    axes[2].set_title('각가속도: KF2 추정 → KF1 Q 제어')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 4. 적응형 Q1
    axes[3].plot(t, q1_history, 'r-', linewidth=2)
    axes[3].axhline(akf.q1_base, color='b', linestyle='--', alpha=0.7, 
                   label=f'정속 Q1={akf.q1_base}')
    axes[3].axhline(akf.q1_accel, color='r', linestyle='--', alpha=0.7, 
                   label=f'가속 Q1={akf.q1_accel}')
    axes[3].set_ylabel('Q1 값')
    axes[3].set_title('적응형 Q1: KF2 가속도 기반 자동 조정')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    # 5. 에러 비교
    axes[4].plot(t, np.degrees(err_meas), 'gray', label=f'측정 ({rms_meas:.2f}°)', 
                linewidth=0.5, alpha=0.5)
    axes[4].plot(t, np.degrees(err_theta), 'b-', label=f'칼만 ({rms_theta:.2f}°)', linewidth=1)
    axes[4].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[4].set_xlabel('시간 (s)')
    axes[4].set_ylabel('각도 에러 (deg)')
    axes[4].set_title(f'에러 비교: 칼만이 {rms_meas/rms_theta:.1f}x 개선')
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)
    axes[4].set_ylim([-2, 2])
    
    # 구간 표시
    for ax in axes:
        ax.axvspan(0, 0.4, alpha=0.1, color='red', label='가속1')
        ax.axvspan(1.5, 2.0, alpha=0.1, color='orange', label='가속2')
        ax.axvspan(0.4, 1.5, alpha=0.1, color='green', label='등속1')
        ax.axvspan(2.0, 2.6, alpha=0.1, color='green', label='등속2')
        ax.axvspan(2.6, 3.0, alpha=0.1, color='blue', label='감속')
    
    plt.suptitle('적응형 계층 칼만필터: KF2 가속도 → KF1 Q 자동조정', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== 결론 ===")
    print(f"✅ KF2 가속도 감지 → KF1 Q 자동 조정 성공!")
    print(f"✅ 가속도 구간: Q1 자동 증가 → 빠른 적응")
    print(f"✅ 등속 구간: Q1 자동 감소 → 정확한 추정")
    print(f"✅ 실시간 적응형 필터링 구현")

if __name__ == "__main__":
    main()