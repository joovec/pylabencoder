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

# ============== 데이터 생성 함수 ==============
def generate_true_angle(rpm, dt, total_time):
    """등속 회전 각도 데이터 생성"""
    t = np.arange(0, total_time, dt)
    omega = rpm * 2 * np.pi / 60.0  # rad/s
    theta_true = wrap_2pi(omega * t)
    return t, theta_true, omega

def generate_true_angle_acc(dt, total_time, profile_type='trapezoid', **params):
    """
    실제 로봇 관절의 가속도 운동을 고려한 각도 생성
    
    Parameters:
    -----------
    dt : float
        샘플링 주기 (초)
    total_time : float
        총 시간 (초)
    profile_type : str
        'trapezoid': 사다리꼴 속도 프로파일 (기본 로봇 모션)
        's_curve': S-커브 프로파일 (부드러운 서보 모션)
        'sin_accel': 사인파 가속 (주기적 운동)
        'step_response': 스텝 응답 (급격한 명령 변화)
    **params : dict
        프로파일별 파라미터
        
    Returns:
    --------
    t : array
        시간 배열
    theta : array
        각도 배열 (rad, wrapped to [0, 2π))
    omega : array
        각속도 배열 (rad/s)
    alpha : array
        각가속도 배열 (rad/s²)
    """
    t = np.arange(0, total_time, dt)
    
    if profile_type == 'trapezoid':
        # 사다리꼴 속도 프로파일
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
        # S-커브 프로파일 (jerk 제한)
        rpm_max = params.get('rpm_max', 1000)
        jerk_time = params.get('jerk_time', 0.2)
        
        omega_max = rpm_max * 2 * np.pi / 60
        alpha_max = omega_max / (2 * jerk_time + 0.5)  # 간소화된 계산
        jerk = alpha_max / jerk_time
        
        omega = np.zeros_like(t)
        alpha = np.zeros_like(t)
        
        for i, ti in enumerate(t):
            if ti <= jerk_time:
                # Jerk up
                alpha[i] = jerk * ti
                omega[i] = 0.5 * jerk * ti**2
            elif ti <= 2 * jerk_time:
                # Jerk down  
                dt = ti - jerk_time
                alpha[i] = alpha_max - jerk * dt
                omega[i] = 0.5 * jerk * jerk_time**2 + alpha_max * dt - 0.5 * jerk * dt**2
            elif ti <= total_time / 2:
                # Constant velocity
                omega[i] = omega_max
                alpha[i] = 0
            else:
                # 대칭 감속
                t_mirror = total_time - ti
                if t_mirror <= 2 * jerk_time:
                    if t_mirror <= jerk_time:
                        alpha[i] = -jerk * t_mirror
                        omega[i] = 0.5 * jerk * t_mirror**2
                    else:
                        dt = t_mirror - jerk_time
                        alpha[i] = -alpha_max + jerk * dt
                        omega[i] = 0.5 * jerk * jerk_time**2 + alpha_max * dt - 0.5 * jerk * dt**2
                else:
                    omega[i] = 0
                    alpha[i] = 0
                    
    elif profile_type == 'sin_accel':
        # 사인파 가속 프로파일
        rpm_mean = params.get('rpm_mean', 500)
        rpm_amplitude = params.get('rpm_amplitude', 300)
        freq = params.get('freq', 0.5)
        
        omega_mean = rpm_mean * 2 * np.pi / 60
        omega_amp = rpm_amplitude * 2 * np.pi / 60
        
        omega = omega_mean + omega_amp * np.sin(2 * np.pi * freq * t)
        alpha = omega_amp * 2 * np.pi * freq * np.cos(2 * np.pi * freq * t)
        
    elif profile_type == 'step_response':
        # 스텝 응답 (1차 시스템 응답)
        rpm_final = params.get('rpm_final', 1000)
        time_constant = params.get('time_constant', 0.3)
        
        omega_final = rpm_final * 2 * np.pi / 60
        omega = omega_final * (1 - np.exp(-t / time_constant))
        alpha = (omega_final / time_constant) * np.exp(-t / time_constant)
        
    else:
        raise ValueError(f"Unknown profile type: {profile_type}")
    
    # 각도 적분
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
    
    # 상태 전이 행렬 F
    kf.F = np.array([[1.0, dt],
                     [0.0, 1.0]])
    
    # 관측 행렬 H
    kf.H = np.array([[1.0, 0.0]])
    
    # 프로세스 노이즈 Q
    kf.Q = process_noise * np.array([[dt**3/3, dt**2/2],
                                      [dt**2/2, dt]])
    
    # 측정 노이즈 R
    kf.R = np.array([[meas_noise]])
    
    # 초기 공분산 P
    kf.P = np.diag([1.0, 1.0])
    
    # 초기 상태 x
    kf.x = np.array([[initial_theta], [initial_omega]])
    
    return kf

def run_kalman_filter(kf, theta_meas):
    """칼만필터 실행"""
    theta_est = np.zeros_like(theta_meas)
    omega_est = np.zeros_like(theta_meas)
    
    for i in range(len(theta_meas)):
        if i > 0:
            kf.predict()
        
        # 측정값 업데이트 (각도 불연속 처리)
        z = theta_meas[i]
        innovation = diffpi(z, kf.x[0, 0])
        kf.update(np.array([[kf.x[0, 0] + innovation]]))
        
        # 결과 저장
        theta_est[i] = wrap_2pi(kf.x[0, 0])
        omega_est[i] = kf.x[1, 0]
    
    return theta_est, omega_est

# ============== 성능 평가 함수 ==============
def calculate_errors(theta_meas, theta_est, theta_true):
    """에러 계산"""
    err_meas = diffpi(theta_meas, theta_true)
    err_est = diffpi(theta_est, theta_true)
    
    rms_meas = np.sqrt(np.mean(err_meas**2))
    rms_est = np.sqrt(np.mean(err_est**2))
    
    return err_meas, err_est, rms_meas, rms_est

# ============== 시각화 함수 ==============
def plot_results(t, theta_true, theta_meas, theta_est, err_meas, err_est, omega_est, rpm):
    """결과 플롯"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # 각도 플롯
    axes[0].plot(t, np.degrees(theta_true), 'k-', label='True', linewidth=2)
    axes[0].plot(t, np.degrees(theta_meas), 'r.', label='Measured', markersize=1, alpha=0.5)
    axes[0].plot(t, np.degrees(theta_est), 'b-', label='CV Kalman', linewidth=1.5)
    axes[0].set_ylabel('Angle (deg)')
    axes[0].set_title(f'등속 {rpm} RPM - CV 칼만필터 vs 측정값')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 에러 플롯
    axes[1].plot(t, np.degrees(err_meas), 'r-', label='Measurement Error', linewidth=0.5, alpha=0.7)
    axes[1].plot(t, np.degrees(err_est), 'b-', label='CV Kalman Error', linewidth=1)
    axes[1].set_ylabel('Error (deg)')
    axes[1].set_title('각도 에러 비교')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([-2, 2])
    
    # 속도 플롯
    axes[2].plot(t, [rpm]*len(t), 'k--', label='True RPM', linewidth=2)
    axes[2].plot(t, omega_est * 60 / (2*np.pi), 'b-', label='Estimated RPM', linewidth=1.5)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('RPM')
    axes[2].set_title('속도 추정')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def print_performance(rms_meas, rms_est):
    """성능 출력"""
    print(f"측정값 RMS 에러: {np.degrees(rms_meas):.4f}°")
    print(f"칼만필터 RMS 에러: {np.degrees(rms_est):.4f}°")
    print(f"개선율: {rms_meas/rms_est:.2f}x")

# ============== 메인 실행 함수 ==============
def main():
    # 파라미터 설정
    rpm = 500.0          # 등속 500 RPM
    dt = 0.001           # 1kHz 샘플링
    total_time = 2.0     # 총 시간
    noise_std = np.deg2rad(0.5)  # 0.5도 노이즈
    process_noise = 1e-5  # 프로세스 노이즈
    
    # 데이터 생성
    t, theta_true, omega_true = generate_true_angle(rpm, dt, total_time)
    theta_meas = add_measurement_noise(theta_true, noise_std)
    
    # 칼만필터 생성 및 실행
    kf = create_cv_kalman_filter(dt, process_noise, noise_std**2, theta_meas[0], omega_true)
    theta_est, omega_est = run_kalman_filter(kf, theta_meas)
    
    # 에러 계산
    err_meas, err_est, rms_meas, rms_est = calculate_errors(theta_meas, theta_est, theta_true)
    
    # 결과 시각화
    plot_results(t, theta_true, theta_meas, theta_est, err_meas, err_est, omega_est, rpm)
    
    # 성능 출력
    print_performance(rms_meas, rms_est)
    
    plt.show()

if __name__ == "__main__":
    main()