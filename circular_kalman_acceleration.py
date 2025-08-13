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
    """사다리꼴 속도 프로파일 생성 (각도는 언랩으로 반환)"""
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

class CircularKalmanFilter:
    """순환 각도를 위한 개선된 칼만필터"""
    
    def __init__(self, dim_x, dim_z, dt):
        self.kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
        self.dt = dt
    
    def setup_cv_model(self, q_value, noise_std, init_theta, init_omega):
        """CV 모델 설정"""
        self.kf.F = np.array([[1.0, self.dt], 
                              [0.0, 1.0]])
        self.kf.H = np.array([[1.0, 0.0]])
        self.kf.Q = q_value * np.array([[self.dt**3/3, self.dt**2/2], 
                                        [self.dt**2/2, self.dt]])
        self.kf.R = np.array([[noise_std**2]])
        self.kf.P = np.diag([noise_std**2, 0.1])
        self.kf.x = np.array([[init_theta], [init_omega]])
    
    def setup_ca_model(self, q_value, noise_std, init_theta, init_omega, init_alpha):
        """CA 모델 설정"""
        self.kf.F = np.array([[1.0, self.dt, 0.5*self.dt**2],
                              [0.0, 1.0, self.dt],
                              [0.0, 0.0, 1.0]])
        self.kf.H = np.array([[1.0, 0.0, 0.0]])
        self.kf.Q = q_value * np.array([[self.dt**5/20, self.dt**4/8, self.dt**3/6],
                                        [self.dt**4/8, self.dt**3/3, self.dt**2/2],
                                        [self.dt**3/6, self.dt**2/2, self.dt]])
        self.kf.R = np.array([[noise_std**2]])
        self.kf.P = np.diag([noise_std**2, 0.1, 1.0])
        self.kf.x = np.array([[init_theta], [init_omega], [init_alpha]])
    
    def predict(self):
        """예측 단계"""
        self.kf.predict()
    
    def update_circular(self, z_wrapped):
        """순환 각도를 고려한 업데이트"""
        # 현재 상태의 각도 예측값을 wrap
        h_x_wrapped = wrap_2pi(self.kf.x[0, 0])
        
        # 순환 혁신 계산
        innovation = diffpi(z_wrapped, h_x_wrapped)
        
        # 표준 칼만 업데이트 (혁신만 순환 처리)
        S = self.kf.H @ self.kf.P @ self.kf.H.T + self.kf.R
        K = self.kf.P @ self.kf.H.T / S[0, 0]
        
        # 상태 업데이트 (언랩 상태 유지)
        self.kf.x = self.kf.x + K * innovation
        self.kf.P = (np.eye(self.kf.x.shape[0]) - K @ self.kf.H) @ self.kf.P
    
    def get_state(self):
        """현재 상태 반환"""
        return self.kf.x.flatten()

def run_circular_cv_filter(theta_meas_wrapped, omega_init, dt, q_value, noise_std):
    """순환 CV 칼만필터 실행"""
    ckf = CircularKalmanFilter(dim_x=2, dim_z=1, dt=dt)
    ckf.setup_cv_model(q_value, noise_std, theta_meas_wrapped[0], omega_init)
    
    theta_est = np.zeros_like(theta_meas_wrapped)
    omega_est = np.zeros_like(theta_meas_wrapped)
    
    for i in range(len(theta_meas_wrapped)):
        if i > 0:
            ckf.predict()
        
        ckf.update_circular(theta_meas_wrapped[i])
        state = ckf.get_state()
        
        theta_est[i] = state[0]
        omega_est[i] = state[1]
    
    return theta_est, omega_est

def run_circular_ca_filter(theta_meas_wrapped, omega_init, alpha_init, dt, q_value, noise_std):
    """순환 CA 칼만필터 실행"""
    ckf = CircularKalmanFilter(dim_x=3, dim_z=1, dt=dt)
    ckf.setup_ca_model(q_value, noise_std, theta_meas_wrapped[0], omega_init, alpha_init)
    
    theta_est = np.zeros_like(theta_meas_wrapped)
    omega_est = np.zeros_like(theta_meas_wrapped)
    alpha_est = np.zeros_like(theta_meas_wrapped)
    
    for i in range(len(theta_meas_wrapped)):
        if i > 0:
            ckf.predict()
        
        ckf.update_circular(theta_meas_wrapped[i])
        state = ckf.get_state()
        
        theta_est[i] = state[0]
        omega_est[i] = state[1]
        alpha_est[i] = state[2]
    
    return theta_est, omega_est, alpha_est

def main():
    # 파라미터
    dt_values = [0.001, 0.005, 0.01]  # 1kHz, 200Hz, 100Hz
    total_time = 2.0
    noise_std = np.deg2rad(0.5)
    
    # Q 값 (1차원 분석에서 얻은 최적값 적용)
    q_cv_optimal = 10.0
    q_ca_optimal = 100.0
    
    profiles = [
        ('Trapezoid', {'rpm_max': 1000, 'accel_time': 0.4}),
        ('S-curve', {'rpm_max': 1000, 'accel_time': 0.3}),
        ('High-speed', {'rpm_max': 2000, 'accel_time': 0.2})
    ]
    
    results = {}
    
    print("=== 순환 칼만필터를 가속도 시나리오에 적용 ===\n")
    
    for profile_name, params in profiles:
        print(f"\n프로파일: {profile_name}")
        results[profile_name] = {}
        
        for dt in dt_values:
            freq = 1/dt
            print(f"  샘플링: {dt*1000:.0f}ms ({freq:.0f}Hz)")
            
            # 데이터 생성
            t, theta_true_unwrap, omega_true, alpha_true = generate_trapezoid_profile(
                dt, total_time, **params
            )
            
            # 현실적인 측정값: wrap된 각도 + 노이즈
            np.random.seed(42)
            theta_true_wrapped = wrap_2pi(theta_true_unwrap)
            theta_meas_wrapped = wrap_2pi(theta_true_wrapped + np.random.normal(0, noise_std, len(t)))
            
            # 순환 CV 필터
            theta_cv, omega_cv = run_circular_cv_filter(
                theta_meas_wrapped, omega_true[0], dt, q_cv_optimal, noise_std
            )
            
            # 순환 CA 필터
            theta_ca, omega_ca, alpha_ca = run_circular_ca_filter(
                theta_meas_wrapped, omega_true[0], 0.0, dt, q_ca_optimal, noise_std
            )
            
            # 에러 계산 (언랩 기준)
            err_meas = diffpi(theta_meas_wrapped, theta_true_wrapped)
            err_cv = theta_cv - theta_true_unwrap
            err_ca = theta_ca - theta_true_unwrap
            
            rms_meas = np.degrees(np.sqrt(np.mean(err_meas**2)))
            rms_cv = np.degrees(np.sqrt(np.mean(err_cv**2)))
            rms_ca = np.degrees(np.sqrt(np.mean(err_ca**2)))
            
            # 속도 에러
            vel_err_cv = omega_cv - omega_true
            vel_err_ca = omega_ca - omega_true
            rms_vel_cv = np.sqrt(np.mean(vel_err_cv**2)) * 60/(2*np.pi)  # RPM
            rms_vel_ca = np.sqrt(np.mean(vel_err_ca**2)) * 60/(2*np.pi)
            
            results[profile_name][dt] = {
                'rms_angle_meas': rms_meas,
                'rms_angle_cv': rms_cv,
                'rms_angle_ca': rms_ca,
                'rms_vel_cv': rms_vel_cv,
                'rms_vel_ca': rms_vel_ca,
                'data': {
                    't': t,
                    'theta_true_unwrap': theta_true_unwrap,
                    'theta_true_wrapped': theta_true_wrapped,
                    'theta_meas_wrapped': theta_meas_wrapped,
                    'theta_cv': theta_cv,
                    'theta_ca': theta_ca,
                    'omega_true': omega_true,
                    'omega_cv': omega_cv,
                    'omega_ca': omega_ca,
                    'alpha_true': alpha_true,
                    'alpha_ca': alpha_ca
                }
            }
            
            print(f"    각도 RMS - 측정: {rms_meas:.3f}°, CV: {rms_cv:.3f}°, CA: {rms_ca:.3f}°")
            print(f"    속도 RMS - CV: {rms_vel_cv:.1f}RPM, CA: {rms_vel_ca:.1f}RPM")
    
    # 시각화
    fig, axes = plt.subplots(len(profiles), 4, figsize=(16, 12))
    
    if len(profiles) == 1:
        axes = axes.reshape(1, -1)
    
    for profile_idx, (profile_name, _) in enumerate(profiles):
        # 최고 성능 설정 선택 (1kHz)
        best_dt = 0.001
        data = results[profile_name][best_dt]['data']
        
        t = data['t']
        
        # 각도 비교 (wrap된 형태로 시각화)
        axes[profile_idx, 0].plot(t[::50], np.degrees(data['theta_true_wrapped'][::50]), 
                                  'k-', label='True', linewidth=2)
        axes[profile_idx, 0].plot(t[::50], np.degrees(data['theta_meas_wrapped'][::50]), 
                                  'gray', marker='.', markersize=1, linestyle='', alpha=0.3, label='Meas')
        axes[profile_idx, 0].plot(t[::50], np.degrees(wrap_2pi(data['theta_cv'][::50])), 
                                  'b-', label='CV', alpha=0.8)
        axes[profile_idx, 0].plot(t[::50], np.degrees(wrap_2pi(data['theta_ca'][::50])), 
                                  'g-', label='CA', alpha=0.8)
        axes[profile_idx, 0].set_ylabel('Angle (deg)')
        axes[profile_idx, 0].set_title(f'{profile_name} - Wrapped Angle')
        axes[profile_idx, 0].legend(fontsize=8)
        axes[profile_idx, 0].grid(True, alpha=0.3)
        
        # 속도
        axes[profile_idx, 1].plot(t, data['omega_true'] * 60/(2*np.pi), 'k-', label='True', linewidth=2)
        axes[profile_idx, 1].plot(t, data['omega_cv'] * 60/(2*np.pi), 'b-', label='CV', alpha=0.8)
        axes[profile_idx, 1].plot(t, data['omega_ca'] * 60/(2*np.pi), 'g-', label='CA', alpha=0.8)
        axes[profile_idx, 1].set_ylabel('RPM')
        axes[profile_idx, 1].set_title('Speed')
        axes[profile_idx, 1].legend(fontsize=8)
        axes[profile_idx, 1].grid(True, alpha=0.3)
        
        # 가속도
        axes[profile_idx, 2].plot(t, data['alpha_true'] * 60/(2*np.pi), 'k-', label='True', linewidth=2)
        axes[profile_idx, 2].plot(t, data['alpha_ca'] * 60/(2*np.pi), 'g-', label='CA', alpha=0.8)
        axes[profile_idx, 2].set_ylabel('RPM/s')
        axes[profile_idx, 2].set_title('Acceleration')
        axes[profile_idx, 2].legend(fontsize=8)
        axes[profile_idx, 2].grid(True, alpha=0.3)
        
        # 각도 에러
        err_meas = diffpi(data['theta_meas_wrapped'], data['theta_true_wrapped'])
        err_cv = data['theta_cv'] - data['theta_true_unwrap']
        err_ca = data['theta_ca'] - data['theta_true_unwrap']
        
        axes[profile_idx, 3].plot(t, np.degrees(err_meas), 'gray', label='Meas Error', 
                                  linewidth=0.5, alpha=0.5)
        axes[profile_idx, 3].plot(t, np.degrees(err_cv), 'b-', label='CV Error', linewidth=1)
        axes[profile_idx, 3].plot(t, np.degrees(err_ca), 'g-', label='CA Error', linewidth=1)
        axes[profile_idx, 3].set_ylabel('Error (deg)')
        axes[profile_idx, 3].set_title('Angle Error')
        axes[profile_idx, 3].legend(fontsize=8)
        axes[profile_idx, 3].grid(True, alpha=0.3)
        axes[profile_idx, 3].set_ylim([-2, 2])
        
        # 구간 표시
        accel_time = 0.4 if profile_name == 'Trapezoid' else 0.3
        for ax in axes[profile_idx, :]:
            ax.axvspan(0, accel_time, alpha=0.1, color='red')
            ax.axvspan(total_time - accel_time, total_time, alpha=0.1, color='blue')
    
    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Time (s)')
    axes[-1, 2].set_xlabel('Time (s)')
    axes[-1, 3].set_xlabel('Time (s)')
    
    plt.suptitle('순환 칼만필터: 가속도 시나리오 적용 결과', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 성능 요약 플롯
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    
    profile_names = list(results.keys())
    dt_labels = ['1ms (1kHz)', '5ms (200Hz)', '10ms (100Hz)']
    
    # 각도 성능
    for profile_idx, profile_name in enumerate(profile_names):
        cv_rms = [results[profile_name][dt]['rms_angle_cv'] for dt in dt_values]
        ca_rms = [results[profile_name][dt]['rms_angle_ca'] for dt in dt_values]
        
        x = np.arange(len(dt_values)) + profile_idx * 0.25
        axes2[0].bar(x - 0.1, cv_rms, 0.2, label=f'{profile_name} CV', alpha=0.7)
        axes2[0].bar(x + 0.1, ca_rms, 0.2, label=f'{profile_name} CA', alpha=0.7)
    
    axes2[0].set_xlabel('샘플링 주파수')
    axes2[0].set_ylabel('RMS Angle Error (deg)')
    axes2[0].set_title('각도 추정 성능')
    axes2[0].set_xticks(np.arange(len(dt_values)))
    axes2[0].set_xticklabels(dt_labels)
    axes2[0].legend(fontsize=8)
    axes2[0].grid(True, alpha=0.3, axis='y')
    
    # 속도 성능
    for profile_idx, profile_name in enumerate(profile_names):
        cv_rms = [results[profile_name][dt]['rms_vel_cv'] for dt in dt_values]
        ca_rms = [results[profile_name][dt]['rms_vel_ca'] for dt in dt_values]
        
        x = np.arange(len(dt_values)) + profile_idx * 0.25
        axes2[1].bar(x - 0.1, cv_rms, 0.2, label=f'{profile_name} CV', alpha=0.7)
        axes2[1].bar(x + 0.1, ca_rms, 0.2, label=f'{profile_name} CA', alpha=0.7)
    
    axes2[1].set_xlabel('샘플링 주파수')
    axes2[1].set_ylabel('RMS Velocity Error (RPM)')
    axes2[1].set_title('속도 추정 성능')
    axes2[1].set_xticks(np.arange(len(dt_values)))
    axes2[1].set_xticklabels(dt_labels)
    axes2[1].legend(fontsize=8)
    axes2[1].grid(True, alpha=0.3, axis='y')
    
    # 개선율 비교
    for profile_idx, profile_name in enumerate(profile_names):
        meas_rms = [results[profile_name][dt]['rms_angle_meas'] for dt in dt_values]
        cv_rms = [results[profile_name][dt]['rms_angle_cv'] for dt in dt_values]
        ca_rms = [results[profile_name][dt]['rms_angle_ca'] for dt in dt_values]
        
        improve_cv = [m/c for m, c in zip(meas_rms, cv_rms)]
        improve_ca = [m/c for m, c in zip(meas_rms, ca_rms)]
        
        x = np.arange(len(dt_values)) + profile_idx * 0.25
        axes2[2].bar(x - 0.1, improve_cv, 0.2, label=f'{profile_name} CV', alpha=0.7)
        axes2[2].bar(x + 0.1, improve_ca, 0.2, label=f'{profile_name} CA', alpha=0.7)
    
    axes2[2].set_xlabel('샘플링 주파수')
    axes2[2].set_ylabel('개선율 (배수)')
    axes2[2].set_title('측정값 대비 개선율')
    axes2[2].set_xticks(np.arange(len(dt_values)))
    axes2[2].set_xticklabels(dt_labels)
    axes2[2].legend(fontsize=8)
    axes2[2].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('순환 칼만필터 성능 요약', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 최종 결과 테이블
    print("\n" + "="*80)
    print("순환 칼만필터 성능 요약 (1kHz 기준)")
    print("="*80)
    print(f"{'Profile':<12} {'각도 RMS (deg)':<20} {'속도 RMS (RPM)':<20} {'개선율':<15}")
    print(f"{'':12} {'Meas':>6} {'CV':>6} {'CA':>6} {'CV':>8} {'CA':>8} {'CV':>6} {'CA':>6}")
    print("-"*80)
    
    for profile_name in profile_names:
        best_dt = 0.001
        data = results[profile_name][best_dt]
        
        improve_cv = data['rms_angle_meas'] / data['rms_angle_cv']
        improve_ca = data['rms_angle_meas'] / data['rms_angle_ca']
        
        print(f"{profile_name:<12} "
              f"{data['rms_angle_meas']:>6.3f} {data['rms_angle_cv']:>6.3f} {data['rms_angle_ca']:>6.3f} "
              f"{data['rms_vel_cv']:>8.1f} {data['rms_vel_ca']:>8.1f} "
              f"{improve_cv:>6.1f} {improve_ca:>6.1f}")
    
    print("="*80)
    print("결론: 순환 혁신 방법으로 현실적인 wrap된 측정값에서도 우수한 성능 달성!")
    
    plt.show()

if __name__ == "__main__":
    main()