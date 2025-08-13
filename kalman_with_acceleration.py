import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.family'] = 'Malgun Gothic'

# ============== 각도 처리 함수 ==============
def diffpi(angle1, angle2):
    """두 각도의 차이를 구하는 함수 (불연속 구간 고려)"""
    diff = angle1 - angle2
    return np.mod(diff + np.pi, 2*np.pi) - np.pi

def wrap_2pi(angle):
    """입력 각도를 [0, 2pi) 범위로 래핑"""
    return np.mod(angle, 2*np.pi)

# ============== 가속도 프로파일 데이터 생성 ==============
def generate_true_angle_acc(dt, total_time, profile_type='trapezoid', **params):
    """실제 로봇 관절의 가속도 운동을 고려한 각도 생성"""
    t = np.arange(0, total_time, dt)
    
    if profile_type == 'trapezoid':
        rpm_start = params.get('rpm_start', 0)
        rpm_end = params.get('rpm_end', 1000)
        accel_time = params.get('accel_time', 0.5)
        
        omega_start = rpm_start * 2 * np.pi / 60
        omega_end = rpm_end * 2 * np.pi / 60
        
        omega = np.zeros_like(t)
        alpha = np.zeros_like(t)
        
        accel_rate = (omega_end - omega_start) / accel_time
        decel_time = accel_time
        const_time = total_time - accel_time - decel_time
        
        for i, ti in enumerate(t):
            if ti <= accel_time:
                omega[i] = omega_start + accel_rate * ti
                alpha[i] = accel_rate
            elif ti <= accel_time + const_time:
                omega[i] = omega_end
                alpha[i] = 0
            else:
                t_decel = ti - accel_time - const_time
                omega[i] = omega_end - accel_rate * t_decel
                alpha[i] = -accel_rate
                if omega[i] < 0:
                    omega[i] = 0
                    alpha[i] = 0
        
    elif profile_type == 's_curve':
        rpm_max = params.get('rpm_max', 1000)
        jerk_time = params.get('jerk_time', 0.2)
        
        omega_max = rpm_max * 2 * np.pi / 60
        alpha_max = omega_max / (2 * jerk_time + 0.5)
        jerk = alpha_max / jerk_time
        
        omega = np.zeros_like(t)
        alpha = np.zeros_like(t)
        
        for i, ti in enumerate(t):
            if ti <= jerk_time:
                alpha[i] = jerk * ti
                omega[i] = 0.5 * jerk * ti**2
            elif ti <= 2 * jerk_time:
                dt_local = ti - jerk_time
                alpha[i] = alpha_max - jerk * dt_local
                omega[i] = 0.5 * jerk * jerk_time**2 + alpha_max * dt_local - 0.5 * jerk * dt_local**2
            elif ti <= total_time / 2:
                omega[i] = omega_max
                alpha[i] = 0
            else:
                t_mirror = total_time - ti
                if t_mirror <= 2 * jerk_time:
                    if t_mirror <= jerk_time:
                        alpha[i] = -jerk * t_mirror
                        omega[i] = 0.5 * jerk * t_mirror**2
                    else:
                        dt_local = t_mirror - jerk_time
                        alpha[i] = -alpha_max + jerk * dt_local
                        omega[i] = 0.5 * jerk * jerk_time**2 + alpha_max * dt_local - 0.5 * jerk * dt_local**2
                else:
                    omega[i] = 0
                    alpha[i] = 0
                    
    elif profile_type == 'sin_accel':
        rpm_mean = params.get('rpm_mean', 500)
        rpm_amplitude = params.get('rpm_amplitude', 300)
        freq = params.get('freq', 0.5)
        
        omega_mean = rpm_mean * 2 * np.pi / 60
        omega_amp = rpm_amplitude * 2 * np.pi / 60
        
        omega = omega_mean + omega_amp * np.sin(2 * np.pi * freq * t)
        alpha = omega_amp * 2 * np.pi * freq * np.cos(2 * np.pi * freq * t)
        
    elif profile_type == 'step_response':
        rpm_final = params.get('rpm_final', 1000)
        time_constant = params.get('time_constant', 0.3)
        
        omega_final = rpm_final * 2 * np.pi / 60
        omega = omega_final * (1 - np.exp(-t / time_constant))
        alpha = (omega_final / time_constant) * np.exp(-t / time_constant)
    
    theta = np.cumsum(omega) * dt
    return t, wrap_2pi(theta), omega, alpha

def add_measurement_noise(theta_true, noise_std, seed=42):
    """측정 노이즈 추가"""
    np.random.seed(seed)
    theta_meas = wrap_2pi(theta_true + np.random.normal(0, noise_std, len(theta_true)))
    return theta_meas

# ============== 칼만필터 함수 ==============
def create_cv_kalman_filter(dt, process_noise, meas_noise, initial_theta, initial_omega):
    """CV (Constant Velocity) 칼만필터 생성"""
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.F = np.array([[1.0, dt], [0.0, 1.0]])
    kf.H = np.array([[1.0, 0.0]])
    kf.Q = process_noise * np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]])
    kf.R = np.array([[meas_noise]])
    kf.P = np.diag([1.0, 1.0])
    kf.x = np.array([[initial_theta], [initial_omega]])
    return kf

def create_ca_kalman_filter(dt, process_noise, meas_noise, initial_theta, initial_omega, initial_alpha):
    """CA (Constant Acceleration) 칼만필터 생성"""
    kf = KalmanFilter(dim_x=3, dim_z=1)
    kf.F = np.array([[1.0, dt, 0.5*dt*dt],
                     [0.0, 1.0, dt],
                     [0.0, 0.0, 1.0]])
    kf.H = np.array([[1.0, 0.0, 0.0]])
    kf.Q = process_noise * np.array([[dt**5/20, dt**4/8, dt**3/6],
                                      [dt**4/8, dt**3/3, dt**2/2],
                                      [dt**3/6, dt**2/2, dt]])
    kf.R = np.array([[meas_noise]])
    kf.P = np.diag([1.0, 10.0, 100.0])
    kf.x = np.array([[initial_theta], [initial_omega], [initial_alpha]])
    return kf

def run_cv_kalman_filter(kf, theta_meas):
    """CV 칼만필터 실행 (언랩 방식)"""
    theta_est = np.zeros_like(theta_meas)
    omega_est = np.zeros_like(theta_meas)
    
    # 언랩된 각도로 작업
    theta_unwrap = 0.0
    kf.x[0, 0] = theta_meas[0]  # 초기값 설정
    
    for i in range(len(theta_meas)):
        if i > 0:
            kf.predict()
            # 예측 후 상태도 언랩 유지
            kf.x[0, 0] = kf.x[0, 0]  # 언랩 상태 유지
            
        # 측정값과의 차이 계산 (wrap 고려)
        z = theta_meas[i]
        innovation = diffpi(z, wrap_2pi(kf.x[0, 0]))
        
        # 언랩 상태에 innovation 적용
        kf.x[0, 0] = kf.x[0, 0] + innovation
        kf.update(np.array([[kf.x[0, 0]]]))
        
        theta_est[i] = wrap_2pi(kf.x[0, 0])
        omega_est[i] = kf.x[1, 0]
    
    return theta_est, omega_est

def run_ca_kalman_filter(kf, theta_meas):
    """CA 칼만필터 실행 (언랩 방식)"""
    theta_est = np.zeros_like(theta_meas)
    omega_est = np.zeros_like(theta_meas)
    alpha_est = np.zeros_like(theta_meas)
    
    # 초기값 설정
    kf.x[0, 0] = theta_meas[0]
    
    for i in range(len(theta_meas)):
        if i > 0:
            kf.predict()
            
        # 측정값과의 차이 계산 (wrap 고려)
        z = theta_meas[i]
        innovation = diffpi(z, wrap_2pi(kf.x[0, 0]))
        
        # 언랩 상태에 innovation 적용
        kf.x[0, 0] = kf.x[0, 0] + innovation
        kf.update(np.array([[kf.x[0, 0]]]))
        
        theta_est[i] = wrap_2pi(kf.x[0, 0])
        omega_est[i] = kf.x[1, 0]
        alpha_est[i] = kf.x[2, 0]
    
    return theta_est, omega_est, alpha_est

# ============== 메인 실행 ==============
def main():
    # 파라미터
    dt = 0.001  # 1kHz
    total_time = 3.0
    noise_std = np.deg2rad(0.5)
    
    # 테스트할 프로파일들
    profiles = [
        ('trapezoid', {'rpm_start': 0, 'rpm_end': 1000, 'accel_time': 0.5}),
        ('s_curve', {'rpm_max': 1000, 'jerk_time': 0.2}),
        ('sin_accel', {'rpm_mean': 500, 'rpm_amplitude': 300, 'freq': 0.5}),
        ('step_response', {'rpm_final': 1000, 'time_constant': 0.3})
    ]
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    
    for idx, (profile_type, params) in enumerate(profiles):
        # 데이터 생성
        t, theta_true, omega_true, alpha_true = generate_true_angle_acc(dt, total_time, profile_type, **params)
        theta_meas = add_measurement_noise(theta_true, noise_std)
        
        # CV 칼만필터
        process_noise_cv = 1e-4 if profile_type in ['trapezoid', 's_curve'] else 1e-5
        kf_cv = create_cv_kalman_filter(dt, process_noise_cv, noise_std**2, theta_meas[0], omega_true[0])
        theta_cv, omega_cv = run_cv_kalman_filter(kf_cv, theta_meas)
        
        # CA 칼만필터
        process_noise_ca = 1e-3 if profile_type in ['trapezoid', 's_curve'] else 1e-4
        kf_ca = create_ca_kalman_filter(dt, process_noise_ca, noise_std**2, theta_meas[0], omega_true[0], alpha_true[0])
        theta_ca, omega_ca, alpha_ca = run_ca_kalman_filter(kf_ca, theta_meas)
        
        # 에러 계산
        err_meas = diffpi(theta_meas, theta_true)
        err_cv = diffpi(theta_cv, theta_true)
        err_ca = diffpi(theta_ca, theta_true)
        
        rms_meas = np.sqrt(np.mean(err_meas**2))
        rms_cv = np.sqrt(np.mean(err_cv**2))
        rms_ca = np.sqrt(np.mean(err_ca**2))
        
        # 플롯 - 속도
        axes[idx, 0].plot(t, omega_true * 60/(2*np.pi), 'k-', label='True', linewidth=2)
        axes[idx, 0].plot(t, omega_cv * 60/(2*np.pi), 'b-', label='CV', alpha=0.7)
        axes[idx, 0].plot(t, omega_ca * 60/(2*np.pi), 'g-', label='CA', alpha=0.7)
        axes[idx, 0].set_ylabel('RPM')
        axes[idx, 0].set_title(f'{profile_type.title()} - Speed')
        axes[idx, 0].legend(fontsize=8)
        axes[idx, 0].grid(True, alpha=0.3)
        
        # 플롯 - 가속도
        axes[idx, 1].plot(t, alpha_true * 60/(2*np.pi), 'k-', label='True', linewidth=2)
        axes[idx, 1].plot(t, alpha_ca * 60/(2*np.pi), 'g-', label='CA Est', alpha=0.7)
        axes[idx, 1].set_ylabel('RPM/s')
        axes[idx, 1].set_title('Acceleration')
        axes[idx, 1].legend(fontsize=8)
        axes[idx, 1].grid(True, alpha=0.3)
        
        # 플롯 - 에러
        axes[idx, 2].plot(t, np.degrees(err_meas), 'r-', label='Meas', linewidth=0.3, alpha=0.5)
        axes[idx, 2].plot(t, np.degrees(err_cv), 'b-', label='CV', linewidth=1)
        axes[idx, 2].plot(t, np.degrees(err_ca), 'g-', label='CA', linewidth=1)
        axes[idx, 2].set_ylabel('Error (deg)')
        axes[idx, 2].set_title('Angle Error')
        axes[idx, 2].legend(fontsize=8)
        axes[idx, 2].grid(True, alpha=0.3)
        axes[idx, 2].set_ylim([-2, 2])
        
        # 플롯 - RMS 비교
        models = ['Meas', 'CV', 'CA']
        rms_values = [np.degrees(rms_meas), np.degrees(rms_cv), np.degrees(rms_ca)]
        colors = ['r', 'b', 'g']
        bars = axes[idx, 3].bar(models, rms_values, color=colors, alpha=0.7)
        axes[idx, 3].set_ylabel('RMS Error (deg)')
        axes[idx, 3].set_title(f'RMS: CV={rms_values[1]:.3f}°, CA={rms_values[2]:.3f}°')
        axes[idx, 3].grid(True, alpha=0.3, axis='y')
        
        # 개선율 표시
        for i, (bar, val) in enumerate(zip(bars, rms_values)):
            if i > 0:
                improvement = rms_values[0] / val
                axes[idx, 3].text(bar.get_x() + bar.get_width()/2, val + 0.01,
                                f'{improvement:.1f}x', ha='center', fontsize=9)
        
        if idx == 3:
            axes[idx, 0].set_xlabel('Time (s)')
            axes[idx, 1].set_xlabel('Time (s)')
            axes[idx, 2].set_xlabel('Time (s)')
            axes[idx, 3].set_xlabel('Model')
    
    plt.suptitle('로봇 가속도 프로파일에 대한 CV vs CA 칼만필터 성능 비교', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 결과 출력
    print("\n=== 가속도 프로파일별 칼만필터 성능 비교 ===\n")
    for idx, (profile_type, params) in enumerate(profiles):
        t, theta_true, omega_true, alpha_true = generate_true_angle_acc(dt, total_time, profile_type, **params)
        theta_meas = add_measurement_noise(theta_true, noise_std)
        
        # CV
        process_noise_cv = 1e-4 if profile_type in ['trapezoid', 's_curve'] else 1e-5
        kf_cv = create_cv_kalman_filter(dt, process_noise_cv, noise_std**2, theta_meas[0], omega_true[0])
        theta_cv, omega_cv = run_cv_kalman_filter(kf_cv, theta_meas)
        
        # CA
        process_noise_ca = 1e-3 if profile_type in ['trapezoid', 's_curve'] else 1e-4
        kf_ca = create_ca_kalman_filter(dt, process_noise_ca, noise_std**2, theta_meas[0], omega_true[0], alpha_true[0])
        theta_ca, omega_ca, alpha_ca = run_ca_kalman_filter(kf_ca, theta_meas)
        
        err_meas = diffpi(theta_meas, theta_true)
        err_cv = diffpi(theta_cv, theta_true)
        err_ca = diffpi(theta_ca, theta_true)
        
        rms_meas = np.degrees(np.sqrt(np.mean(err_meas**2)))
        rms_cv = np.degrees(np.sqrt(np.mean(err_cv**2)))
        rms_ca = np.degrees(np.sqrt(np.mean(err_ca**2)))
        
        print(f"{profile_type:15s}: Meas={rms_meas:.3f}° | CV={rms_cv:.3f}° ({rms_meas/rms_cv:.1f}x) | CA={rms_ca:.3f}° ({rms_meas/rms_ca:.1f}x)")

if __name__ == "__main__":
    main()