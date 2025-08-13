import numpy as np
import matplotlib.pyplot as plt

def wrap_2pi(angle):
    """입력 각도를 [0, 2pi) 범위로 래핑"""
    return np.mod(angle, 2*np.pi)

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
        속도 프로파일 타입 ('trapezoid', 's_curve', 'sin_accel', 'multi_phase')
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
        theta, omega, alpha = _trapezoid_profile(t, **params)
    elif profile_type == 's_curve':
        theta, omega, alpha = _s_curve_profile(t, **params)
    elif profile_type == 'sin_accel':
        theta, omega, alpha = _sin_accel_profile(t, **params)
    elif profile_type == 'multi_phase':
        theta, omega, alpha = _multi_phase_profile(t, **params)
    else:
        raise ValueError(f"Unknown profile type: {profile_type}")
    
    return t, wrap_2pi(theta), omega, alpha

def _trapezoid_profile(t, rpm_start=0, rpm_end=1000, accel_time=0.5, decel_time=0.5):
    """
    사다리꼴 속도 프로파일 (가장 기본적인 로봇 모션)
    
    Parameters:
    -----------
    rpm_start, rpm_end : float
        시작/종료 RPM
    accel_time : float
        가속 시간 (초)
    decel_time : float
        감속 시간 (초)
    """
    omega_start = rpm_start * 2 * np.pi / 60
    omega_end = rpm_end * 2 * np.pi / 60
    
    total_time = t[-1]
    const_time = total_time - accel_time - decel_time
    
    if const_time < 0:
        # 시간이 부족하면 삼각 프로파일
        accel_time = total_time / 2
        decel_time = total_time / 2
        const_time = 0
    
    omega = np.zeros_like(t)
    alpha = np.zeros_like(t)
    
    # 가속 구간
    accel_rate = (omega_end - omega_start) / accel_time
    
    for i, ti in enumerate(t):
        if ti <= accel_time:
            # 가속 구간
            omega[i] = omega_start + accel_rate * ti
            alpha[i] = accel_rate
        elif ti <= accel_time + const_time:
            # 등속 구간
            omega[i] = omega_end
            alpha[i] = 0
        else:
            # 감속 구간
            t_decel = ti - accel_time - const_time
            decel_rate = omega_end / decel_time
            omega[i] = omega_end - decel_rate * t_decel
            alpha[i] = -decel_rate
            if omega[i] < 0:
                omega[i] = 0
                alpha[i] = 0
    
    # 각도 적분
    theta = np.cumsum(omega) * (t[1] - t[0])
    
    return theta, omega, alpha

def _s_curve_profile(t, rpm_max=1000, jerk_time=0.2):
    """
    S-커브 속도 프로파일 (부드러운 가속/감속)
    저크(jerk) 제한이 있는 실제 서보 모터의 동작
    
    Parameters:
    -----------
    rpm_max : float
        최대 RPM
    jerk_time : float
        저크 적용 시간 (초)
    """
    omega_max = rpm_max * 2 * np.pi / 60
    total_time = t[-1]
    
    # 7-segment S-curve 간소화 버전
    t1 = jerk_time  # jerk up
    t2 = total_time / 4 - jerk_time  # const accel
    t3 = jerk_time  # jerk down
    t4 = total_time / 2 - (t1 + t2 + t3)  # const velocity
    
    omega = np.zeros_like(t)
    alpha = np.zeros_like(t)
    
    # 최대 가속도 계산
    alpha_max = omega_max / (2 * jerk_time + t2)
    jerk = alpha_max / jerk_time
    
    for i, ti in enumerate(t):
        if ti <= t1:
            # Jerk up (가속도 증가)
            alpha[i] = jerk * ti
            omega[i] = 0.5 * jerk * ti**2
        elif ti <= t1 + t2:
            # Constant acceleration
            dt = ti - t1
            alpha[i] = alpha_max
            omega[i] = 0.5 * jerk * t1**2 + alpha_max * dt
        elif ti <= t1 + t2 + t3:
            # Jerk down (가속도 감소)
            dt = ti - t1 - t2
            alpha[i] = alpha_max - jerk * dt
            omega[i] = 0.5 * jerk * t1**2 + alpha_max * t2 + alpha_max * dt - 0.5 * jerk * dt**2
        elif ti <= total_time / 2:
            # Constant velocity
            omega[i] = omega_max
            alpha[i] = 0
        else:
            # 대칭적인 감속 (간소화)
            t_mirror = total_time - ti
            if t_mirror <= t1 + t2 + t3:
                # 감속 구간 (가속 구간의 미러)
                if t_mirror <= t1:
                    alpha[i] = -jerk * t_mirror
                    omega[i] = 0.5 * jerk * t_mirror**2
                elif t_mirror <= t1 + t2:
                    dt = t_mirror - t1
                    alpha[i] = -alpha_max
                    omega[i] = 0.5 * jerk * t1**2 + alpha_max * dt
                else:
                    dt = t_mirror - t1 - t2
                    alpha[i] = -alpha_max + jerk * dt
                    omega[i] = 0.5 * jerk * t1**2 + alpha_max * t2 + alpha_max * dt - 0.5 * jerk * dt**2
            else:
                omega[i] = 0
                alpha[i] = 0
    
    theta = np.cumsum(omega) * (t[1] - t[0])
    return theta, omega, alpha

def _sin_accel_profile(t, rpm_mean=500, rpm_amplitude=300, freq=0.5):
    """
    사인파 가속 프로파일 (주기적 운동)
    로봇 팔의 반복 작업이나 진동 테스트용
    
    Parameters:
    -----------
    rpm_mean : float
        평균 RPM
    rpm_amplitude : float
        RPM 진폭
    freq : float
        주파수 (Hz)
    """
    omega_mean = rpm_mean * 2 * np.pi / 60
    omega_amp = rpm_amplitude * 2 * np.pi / 60
    
    # 사인파 속도
    omega = omega_mean + omega_amp * np.sin(2 * np.pi * freq * t)
    
    # 가속도는 속도의 미분
    alpha = omega_amp * 2 * np.pi * freq * np.cos(2 * np.pi * freq * t)
    
    # 각도는 속도의 적분
    theta = np.cumsum(omega) * (t[1] - t[0])
    
    return theta, omega, alpha

def _multi_phase_profile(t, phases=None):
    """
    다단계 운동 프로파일 (실제 pick-and-place 작업)
    
    Parameters:
    -----------
    phases : list of dict
        각 단계별 파라미터
        [{'duration': 0.5, 'rpm_end': 500, 'type': 'accel'},
         {'duration': 1.0, 'rpm_end': 500, 'type': 'const'},
         {'duration': 0.3, 'rpm_end': 1000, 'type': 'accel'},
         {'duration': 0.5, 'rpm_end': 1000, 'type': 'const'},
         {'duration': 0.7, 'rpm_end': 0, 'type': 'decel'}]
    """
    if phases is None:
        # 기본 pick-and-place 시나리오
        phases = [
            {'duration': 0.3, 'rpm_end': 300, 'type': 'accel'},   # 시작 가속
            {'duration': 0.4, 'rpm_end': 300, 'type': 'const'},   # 접근
            {'duration': 0.2, 'rpm_end': 0, 'type': 'decel'},     # 정지 (pick)
            {'duration': 0.1, 'rpm_end': 0, 'type': 'const'},     # 대기
            {'duration': 0.4, 'rpm_end': 800, 'type': 'accel'},   # 빠른 이동
            {'duration': 0.5, 'rpm_end': 800, 'type': 'const'},   # 이동
            {'duration': 0.3, 'rpm_end': 100, 'type': 'decel'},   # 감속
            {'duration': 0.3, 'rpm_end': 100, 'type': 'const'},   # 정밀 접근
            {'duration': 0.2, 'rpm_end': 0, 'type': 'decel'},     # 정지 (place)
        ]
    
    omega = np.zeros_like(t)
    alpha = np.zeros_like(t)
    
    current_time = 0
    current_rpm = 0
    
    for phase in phases:
        duration = phase['duration']
        rpm_end = phase['rpm_end']
        phase_type = phase['type']
        
        omega_start = current_rpm * 2 * np.pi / 60
        omega_end = rpm_end * 2 * np.pi / 60
        
        # 해당 구간 인덱스 찾기
        mask = (t >= current_time) & (t < current_time + duration)
        t_local = t[mask] - current_time
        
        if phase_type == 'accel' or phase_type == 'decel':
            # 선형 가속/감속
            if duration > 0:
                accel = (omega_end - omega_start) / duration
                omega[mask] = omega_start + accel * t_local
                alpha[mask] = accel
        elif phase_type == 'const':
            # 등속
            omega[mask] = omega_end
            alpha[mask] = 0
        
        current_time += duration
        current_rpm = rpm_end
        
        if current_time >= t[-1]:
            break
    
    theta = np.cumsum(omega) * (t[1] - t[0])
    return theta, omega, alpha

# 테스트 및 시각화
if __name__ == "__main__":
    import matplotlib as mpl
    mpl.rcParams['axes.unicode_minus'] = False
    mpl.rcParams['font.family'] = 'Malgun Gothic'
    
    dt = 0.001  # 1kHz
    total_time = 3.0
    
    profiles = [
        ('trapezoid', {'rpm_start': 0, 'rpm_end': 1000, 'accel_time': 0.5, 'decel_time': 0.5}),
        ('s_curve', {'rpm_max': 1000, 'jerk_time': 0.2}),
        ('sin_accel', {'rpm_mean': 500, 'rpm_amplitude': 300, 'freq': 0.5}),
        ('multi_phase', {})
    ]
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    
    for idx, (profile_type, params) in enumerate(profiles):
        t, theta, omega, alpha = generate_true_angle_acc(dt, total_time, profile_type, **params)
        
        # 각도
        axes[idx, 0].plot(t, np.degrees(theta), 'b-', linewidth=1.5)
        axes[idx, 0].set_ylabel('Angle (deg)')
        axes[idx, 0].set_title(f'{profile_type.title()} - Angle')
        axes[idx, 0].grid(True, alpha=0.3)
        
        # 속도
        axes[idx, 1].plot(t, omega * 60 / (2*np.pi), 'g-', linewidth=1.5)
        axes[idx, 1].set_ylabel('Speed (RPM)')
        axes[idx, 1].set_title(f'{profile_type.title()} - Speed')
        axes[idx, 1].grid(True, alpha=0.3)
        
        # 가속도
        axes[idx, 2].plot(t, alpha * 60 / (2*np.pi), 'r-', linewidth=1.5)
        axes[idx, 2].set_ylabel('Accel (RPM/s)')
        axes[idx, 2].set_title(f'{profile_type.title()} - Acceleration')
        axes[idx, 2].grid(True, alpha=0.3)
        
        if idx == 3:
            axes[idx, 0].set_xlabel('Time (s)')
            axes[idx, 1].set_xlabel('Time (s)')
            axes[idx, 2].set_xlabel('Time (s)')
    
    plt.suptitle('실제 로봇 관절 운동 프로파일', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()