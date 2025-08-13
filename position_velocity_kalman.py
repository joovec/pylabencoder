import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.family'] = 'Malgun Gothic'

def generate_motion_profile(dt, total_time, v_max=10.0, accel_time=0.5):
    """1차원 운동 프로파일 생성 (사다리꼴 속도)
    
    Parameters:
    -----------
    dt : float
        샘플링 주기 (초)
    total_time : float
        총 시간 (초)
    v_max : float
        최대 속도 (m/s)
    accel_time : float
        가속/감속 시간 (초)
    """
    t = np.arange(0, total_time, dt)
    
    position = np.zeros_like(t)
    velocity = np.zeros_like(t)
    acceleration = np.zeros_like(t)
    
    accel_rate = v_max / accel_time
    
    for i, ti in enumerate(t):
        if ti <= accel_time:
            # 가속 구간
            acceleration[i] = accel_rate
            velocity[i] = accel_rate * ti
            position[i] = 0.5 * accel_rate * ti**2
        elif ti <= total_time - accel_time:
            # 등속 구간
            acceleration[i] = 0
            velocity[i] = v_max
            position[i] = 0.5 * accel_rate * accel_time**2 + v_max * (ti - accel_time)
        else:
            # 감속 구간
            t_decel = ti - (total_time - accel_time)
            acceleration[i] = -accel_rate
            velocity[i] = v_max - accel_rate * t_decel
            # 등속 끝 위치 + 감속 구간 이동 거리
            pos_const_end = 0.5 * accel_rate * accel_time**2 + v_max * (total_time - 2*accel_time)
            position[i] = pos_const_end + v_max * t_decel - 0.5 * accel_rate * t_decel**2
            
            if velocity[i] < 0:
                velocity[i] = 0
                acceleration[i] = 0
    
    return t, position, velocity, acceleration

def run_cv_filter(pos_meas, v_init, dt, q_value, noise_std):
    """CV (Constant Velocity) 칼만필터 실행"""
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.F = np.array([[1.0, dt], 
                     [0.0, 1.0]])
    kf.H = np.array([[1.0, 0.0]])  # 위치만 측정
    kf.Q = q_value * np.array([[dt**3/3, dt**2/2], 
                               [dt**2/2, dt]])
    kf.R = np.array([[noise_std**2]])
    kf.P = np.diag([noise_std**2, 1.0])
    kf.x = np.array([[pos_meas[0]], [v_init]])
    
    pos_est = np.zeros_like(pos_meas)
    vel_est = np.zeros_like(pos_meas)
    
    for i in range(len(pos_meas)):
        if i > 0:
            kf.predict()
        kf.update(np.array([[pos_meas[i]]]))
        pos_est[i] = kf.x[0, 0]
        vel_est[i] = kf.x[1, 0]
    
    return pos_est, vel_est

def run_ca_filter(pos_meas, v_init, a_init, dt, q_value, noise_std):
    """CA (Constant Acceleration) 칼만필터 실행"""
    kf = KalmanFilter(dim_x=3, dim_z=1)
    kf.F = np.array([[1.0, dt, 0.5*dt*dt],
                     [0.0, 1.0, dt],
                     [0.0, 0.0, 1.0]])
    kf.H = np.array([[1.0, 0.0, 0.0]])  # 위치만 측정
    kf.Q = q_value * np.array([[dt**5/20, dt**4/8, dt**3/6],
                               [dt**4/8, dt**3/3, dt**2/2],
                               [dt**3/6, dt**2/2, dt]])
    kf.R = np.array([[noise_std**2]])
    kf.P = np.diag([noise_std**2, 1.0, 10.0])
    kf.x = np.array([[pos_meas[0]], [v_init], [a_init]])
    
    pos_est = np.zeros_like(pos_meas)
    vel_est = np.zeros_like(pos_meas)
    acc_est = np.zeros_like(pos_meas)
    
    for i in range(len(pos_meas)):
        if i > 0:
            kf.predict()
        kf.update(np.array([[pos_meas[i]]]))
        pos_est[i] = kf.x[0, 0]
        vel_est[i] = kf.x[1, 0]
        acc_est[i] = kf.x[2, 0]
    
    return pos_est, vel_est, acc_est

def main():
    # 파라미터
    dt_values = [0.001, 0.01, 0.1]  # 1kHz, 100Hz, 10Hz
    q_cv_values = np.logspace(-6, 2, 9)  # 10^-6 ~ 10^2
    q_ca_values = np.logspace(-6, 2, 9)
    
    total_time = 2.0
    noise_std = 0.01  # 1cm 위치 노이즈
    
    # 결과 저장
    results_cv = {}
    results_ca = {}
    
    # Figure 1: Q 값 튜닝 히트맵
    fig1, axes1 = plt.subplots(2, 3, figsize=(15, 8))
    
    for dt_idx, dt in enumerate(dt_values):
        print(f"\n샘플링: dt={dt*1000:.1f}ms ({1/dt:.0f}Hz)")
        
        # 데이터 생성
        t, pos_true, vel_true, acc_true = generate_motion_profile(
            dt, total_time, v_max=10.0, accel_time=0.4
        )
        
        # 노이즈 추가
        np.random.seed(42)
        pos_meas = pos_true + np.random.normal(0, noise_std, len(pos_true))
        
        # CV 필터 테스트
        cv_rms_pos = []
        cv_rms_vel = []
        best_cv = {'q': None, 'rms_pos': float('inf'), 'rms_vel': float('inf')}
        
        for q in q_cv_values:
            pos_est, vel_est = run_cv_filter(pos_meas, vel_true[0], dt, q, noise_std)
            
            rms_pos = np.sqrt(np.mean((pos_est - pos_true)**2))
            rms_vel = np.sqrt(np.mean((vel_est - vel_true)**2))
            
            cv_rms_pos.append(rms_pos)
            cv_rms_vel.append(rms_vel)
            
            if rms_pos < best_cv['rms_pos']:
                best_cv = {'q': q, 'rms_pos': rms_pos, 'rms_vel': rms_vel,
                          'pos_est': pos_est, 'vel_est': vel_est}
        
        results_cv[dt] = best_cv
        
        # CA 필터 테스트
        ca_rms_pos = []
        ca_rms_vel = []
        best_ca = {'q': None, 'rms_pos': float('inf'), 'rms_vel': float('inf')}
        
        for q in q_ca_values:
            pos_est, vel_est, acc_est = run_ca_filter(pos_meas, vel_true[0], 0, dt, q, noise_std)
            
            rms_pos = np.sqrt(np.mean((pos_est - pos_true)**2))
            rms_vel = np.sqrt(np.mean((vel_est - vel_true)**2))
            
            ca_rms_pos.append(rms_pos)
            ca_rms_vel.append(rms_vel)
            
            if rms_pos < best_ca['rms_pos']:
                best_ca = {'q': q, 'rms_pos': rms_pos, 'rms_vel': rms_vel,
                          'pos_est': pos_est, 'vel_est': vel_est, 'acc_est': acc_est}
        
        results_ca[dt] = best_ca
        
        # Q 튜닝 곡선 플롯
        axes1[0, dt_idx].loglog(q_cv_values, cv_rms_pos, 'b-o', label='CV Position')
        axes1[0, dt_idx].loglog(q_ca_values, ca_rms_pos, 'g-s', label='CA Position')
        axes1[0, dt_idx].axvline(best_cv['q'], color='b', linestyle='--', alpha=0.5)
        axes1[0, dt_idx].axvline(best_ca['q'], color='g', linestyle='--', alpha=0.5)
        axes1[0, dt_idx].set_xlabel('Q value')
        axes1[0, dt_idx].set_ylabel('RMS Position Error (m)')
        axes1[0, dt_idx].set_title(f'dt={dt*1000:.0f}ms ({1/dt:.0f}Hz)')
        axes1[0, dt_idx].legend()
        axes1[0, dt_idx].grid(True, alpha=0.3, which='both')
        
        axes1[1, dt_idx].loglog(q_cv_values, cv_rms_vel, 'b-o', label='CV Velocity')
        axes1[1, dt_idx].loglog(q_ca_values, ca_rms_vel, 'g-s', label='CA Velocity')
        axes1[1, dt_idx].axvline(best_cv['q'], color='b', linestyle='--', alpha=0.5)
        axes1[1, dt_idx].axvline(best_ca['q'], color='g', linestyle='--', alpha=0.5)
        axes1[1, dt_idx].set_xlabel('Q value')
        axes1[1, dt_idx].set_ylabel('RMS Velocity Error (m/s)')
        axes1[1, dt_idx].legend()
        axes1[1, dt_idx].grid(True, alpha=0.3, which='both')
        
        print(f"  CV 최적: Q={best_cv['q']:.1e}, 위치 RMS={best_cv['rms_pos']*100:.2f}cm, 속도 RMS={best_cv['rms_vel']:.3f}m/s")
        print(f"  CA 최적: Q={best_ca['q']:.1e}, 위치 RMS={best_ca['rms_pos']*100:.2f}cm, 속도 RMS={best_ca['rms_vel']:.3f}m/s")
    
    plt.suptitle('1차원 운동: Q 값 튜닝 (위치만 측정)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Figure 2: 최적 Q에서의 성능 비교
    fig2, axes2 = plt.subplots(3, 3, figsize=(15, 10))
    
    for dt_idx, dt in enumerate(dt_values):
        t, pos_true, vel_true, acc_true = generate_motion_profile(
            dt, total_time, v_max=10.0, accel_time=0.4
        )
        np.random.seed(42)
        pos_meas = pos_true + np.random.normal(0, noise_std, len(pos_true))
        
        cv_best = results_cv[dt]
        ca_best = results_ca[dt]
        
        # 위치 에러 비교
        err_meas = pos_meas - pos_true  # 측정 에러
        err_cv = cv_best['pos_est'] - pos_true  # CV 에러
        err_ca = ca_best['pos_est'] - pos_true  # CA 에러
        
        rms_meas = np.sqrt(np.mean(err_meas**2))
        rms_cv = np.sqrt(np.mean(err_cv**2))
        rms_ca = np.sqrt(np.mean(err_ca**2))
        
        axes2[0, dt_idx].plot(t, err_meas*100, 'r-', 
                             label=f'측정 에러 ({rms_meas*100:.2f}cm)', linewidth=1, alpha=0.7)
        axes2[0, dt_idx].plot(t, err_cv*100, 'b-', 
                             label=f'CV 에러 ({rms_cv*100:.2f}cm)', linewidth=1.5)
        axes2[0, dt_idx].plot(t, err_ca*100, 'g-', 
                             label=f'CA 에러 ({rms_ca*100:.2f}cm)', linewidth=1.5)
        axes2[0, dt_idx].axhline(0, color='k', linestyle='-', alpha=0.3)
        axes2[0, dt_idx].set_ylabel('Position Error (cm)')
        axes2[0, dt_idx].set_title(f'dt={dt*1000:.0f}ms - 위치 에러 비교')
        axes2[0, dt_idx].legend(fontsize=8)
        axes2[0, dt_idx].grid(True, alpha=0.3)
        axes2[0, dt_idx].set_ylim([-5, 5])
        
        # 속도 에러 비교
        vel_err_cv = cv_best['vel_est'] - vel_true
        vel_err_ca = ca_best['vel_est'] - vel_true
        
        rms_vel_cv = np.sqrt(np.mean(vel_err_cv**2))
        rms_vel_ca = np.sqrt(np.mean(vel_err_ca**2))
        
        # 속도는 직접 측정하지 않으므로 측정 에러가 없음 (위치로만 추정)
        axes2[1, dt_idx].plot(t, vel_err_cv, 'b-', 
                             label=f'CV 에러 ({rms_vel_cv:.3f}m/s)', linewidth=1.5)
        axes2[1, dt_idx].plot(t, vel_err_ca, 'g-', 
                             label=f'CA 에러 ({rms_vel_ca:.3f}m/s)', linewidth=1.5)
        axes2[1, dt_idx].axhline(0, color='k', linestyle='-', alpha=0.3)
        axes2[1, dt_idx].set_ylabel('Velocity Error (m/s)')
        axes2[1, dt_idx].set_title('속도 에러 비교')
        axes2[1, dt_idx].legend(fontsize=8)
        axes2[1, dt_idx].grid(True, alpha=0.3)
        axes2[1, dt_idx].set_ylim([-2, 2])
        
        # 가속도 에러 비교 (CA만 추정 가능)
        if 'acc_est' in ca_best:
            acc_err_ca = ca_best['acc_est'] - acc_true
            rms_acc_ca = np.sqrt(np.mean(acc_err_ca**2))
            
            axes2[2, dt_idx].plot(t, acc_err_ca, 'g-', 
                                 label=f'CA 에러 ({rms_acc_ca:.3f}m/s²)', linewidth=1.5)
        axes2[2, dt_idx].axhline(0, color='k', linestyle='-', alpha=0.3)
        axes2[2, dt_idx].set_xlabel('Time (s)')
        axes2[2, dt_idx].set_ylabel('Acceleration Error (m/s²)')
        axes2[2, dt_idx].set_title('가속도 에러 비교')
        axes2[2, dt_idx].legend(fontsize=8)
        axes2[2, dt_idx].grid(True, alpha=0.3)
        axes2[2, dt_idx].set_ylim([-10, 10])
        
        # 구간 표시
        for ax in [axes2[0, dt_idx], axes2[1, dt_idx], axes2[2, dt_idx]]:
            ax.axvspan(0, 0.4, alpha=0.1, color='red')    # 가속
            ax.axvspan(1.6, 2.0, alpha=0.1, color='blue')  # 감속
    
    plt.suptitle('에러 비교: 측정값 vs CV vs CA', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Figure 4: 개선율 비교
    fig4, axes4 = plt.subplots(1, 3, figsize=(15, 5))
    
    dt_labels = ['1ms (1kHz)', '10ms (100Hz)', '100ms (10Hz)']
    
    # 위치 에러 개선율
    pos_improve_cv = []
    pos_improve_ca = []
    vel_improve_cv = []
    vel_improve_ca = []
    
    for dt in dt_values:
        t_temp, pos_true_temp, vel_true_temp, _ = generate_motion_profile(
            dt, total_time, v_max=10.0, accel_time=0.4
        )
        np.random.seed(42)
        pos_meas_temp = pos_true_temp + np.random.normal(0, noise_std, len(pos_true_temp))
        
        # 측정값 RMS
        rms_pos_meas = np.sqrt(np.mean((pos_meas_temp - pos_true_temp)**2))
        
        # CV/CA 결과
        cv_result = results_cv[dt]
        ca_result = results_ca[dt]
        
        # 개선율 계산
        pos_improve_cv.append(rms_pos_meas / cv_result['rms_pos'])
        pos_improve_ca.append(rms_pos_meas / ca_result['rms_pos'])
        vel_improve_cv.append(cv_result['rms_vel'])  # 직접 값 (기준 없음)
        vel_improve_ca.append(ca_result['rms_vel'])
    
    # 위치 개선율
    x = np.arange(len(dt_labels))
    width = 0.35
    
    axes4[0].bar(x - width/2, pos_improve_cv, width, label='CV', color='blue', alpha=0.7)
    axes4[0].bar(x + width/2, pos_improve_ca, width, label='CA', color='green', alpha=0.7)
    axes4[0].set_xlabel('샘플링 주파수')
    axes4[0].set_ylabel('위치 개선율 (배수)')
    axes4[0].set_title('위치 추정 개선율')
    axes4[0].set_xticks(x)
    axes4[0].set_xticklabels(dt_labels)
    axes4[0].legend()
    axes4[0].grid(True, alpha=0.3, axis='y')
    
    # 값 표시
    for i, (cv_val, ca_val) in enumerate(zip(pos_improve_cv, pos_improve_ca)):
        axes4[0].text(i - width/2, cv_val + 0.1, f'{cv_val:.1f}x', ha='center', fontsize=9)
        axes4[0].text(i + width/2, ca_val + 0.1, f'{ca_val:.1f}x', ha='center', fontsize=9)
    
    # 속도 RMS (절대값)
    axes4[1].bar(x - width/2, vel_improve_cv, width, label='CV', color='blue', alpha=0.7)
    axes4[1].bar(x + width/2, vel_improve_ca, width, label='CA', color='green', alpha=0.7)
    axes4[1].set_xlabel('샘플링 주파수')
    axes4[1].set_ylabel('속도 RMS 에러 (m/s)')
    axes4[1].set_title('속도 추정 정확도')
    axes4[1].set_xticks(x)
    axes4[1].set_xticklabels(dt_labels)
    axes4[1].legend()
    axes4[1].grid(True, alpha=0.3, axis='y')
    
    # 값 표시
    for i, (cv_val, ca_val) in enumerate(zip(vel_improve_cv, vel_improve_ca)):
        axes4[1].text(i - width/2, cv_val + 0.01, f'{cv_val:.2f}', ha='center', fontsize=9)
        axes4[1].text(i + width/2, ca_val + 0.01, f'{ca_val:.2f}', ha='center', fontsize=9)
    
    # Q값 비교
    q_cv_vals = [results_cv[dt]['q'] for dt in dt_values]
    q_ca_vals = [results_ca[dt]['q'] for dt in dt_values]
    
    axes4[2].semilogy(x - width/2, q_cv_vals, 'bo-', label='CV 최적 Q', markersize=8, linewidth=2)
    axes4[2].semilogy(x + width/2, q_ca_vals, 'gs-', label='CA 최적 Q', markersize=8, linewidth=2)
    axes4[2].set_xlabel('샘플링 주파수')
    axes4[2].set_ylabel('최적 Q 값')
    axes4[2].set_title('최적 Q 값 비교')
    axes4[2].set_xticks(x)
    axes4[2].set_xticklabels(dt_labels)
    axes4[2].legend()
    axes4[2].grid(True, alpha=0.3, which='both')
    
    plt.suptitle('칼만필터 성능 요약', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 결과 테이블 출력
    print("\n" + "="*70)
    print("최적 Q 값 및 성능 요약")
    print("="*70)
    print(f"{'dt (ms)':<10} {'Model':<8} {'Q value':<12} {'Pos RMS (cm)':<15} {'Vel RMS (m/s)':<15}")
    print("-"*70)
    
    for dt in dt_values:
        cv = results_cv[dt]
        ca = results_ca[dt]
        print(f"{dt*1000:<10.0f} {'CV':<8} {cv['q']:<12.1e} {cv['rms_pos']*100:<15.3f} {cv['rms_vel']:<15.3f}")
        print(f"{'':<10} {'CA':<8} {ca['q']:<12.1e} {ca['rms_pos']*100:<15.3f} {ca['rms_vel']:<15.3f}")
        print("-"*70)
    
    # Figure 3: 구간별 성능 분석
    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 8))
    
    dt_test = 0.01  # 100Hz로 테스트
    t, pos_true, vel_true, acc_true = generate_motion_profile(
        dt_test, total_time, v_max=10.0, accel_time=0.4
    )
    np.random.seed(42)
    pos_meas = pos_true + np.random.normal(0, noise_std, len(pos_true))
    
    # 구간 인덱스
    accel_idx = (t <= 0.4)
    const_idx = (t > 0.4) & (t <= 1.6)
    decel_idx = (t > 1.6)
    
    # 다양한 Q 값에서 구간별 성능
    q_test = np.logspace(-4, 1, 20)
    cv_accel_rms = []
    cv_const_rms = []
    cv_decel_rms = []
    ca_accel_rms = []
    ca_const_rms = []
    ca_decel_rms = []
    
    for q in q_test:
        # CV
        pos_cv, vel_cv = run_cv_filter(pos_meas, vel_true[0], dt_test, q, noise_std)
        err_cv = vel_cv - vel_true
        cv_accel_rms.append(np.sqrt(np.mean(err_cv[accel_idx]**2)))
        cv_const_rms.append(np.sqrt(np.mean(err_cv[const_idx]**2)))
        cv_decel_rms.append(np.sqrt(np.mean(err_cv[decel_idx]**2)))
        
        # CA
        pos_ca, vel_ca, acc_ca = run_ca_filter(pos_meas, vel_true[0], 0, dt_test, q, noise_std)
        err_ca = vel_ca - vel_true
        ca_accel_rms.append(np.sqrt(np.mean(err_ca[accel_idx]**2)))
        ca_const_rms.append(np.sqrt(np.mean(err_ca[const_idx]**2)))
        ca_decel_rms.append(np.sqrt(np.mean(err_ca[decel_idx]**2)))
    
    # 구간별 플롯
    axes3[0, 0].loglog(q_test, cv_accel_rms, 'b-', label='CV', linewidth=2)
    axes3[0, 0].loglog(q_test, ca_accel_rms, 'g-', label='CA', linewidth=2)
    axes3[0, 0].set_xlabel('Q value')
    axes3[0, 0].set_ylabel('Velocity RMS (m/s)')
    axes3[0, 0].set_title('가속 구간 (0-0.4s)')
    axes3[0, 0].legend()
    axes3[0, 0].grid(True, alpha=0.3, which='both')
    
    axes3[0, 1].loglog(q_test, cv_const_rms, 'b-', label='CV', linewidth=2)
    axes3[0, 1].loglog(q_test, ca_const_rms, 'g-', label='CA', linewidth=2)
    axes3[0, 1].set_xlabel('Q value')
    axes3[0, 1].set_ylabel('Velocity RMS (m/s)')
    axes3[0, 1].set_title('등속 구간 (0.4-1.6s)')
    axes3[0, 1].legend()
    axes3[0, 1].grid(True, alpha=0.3, which='both')
    
    axes3[1, 0].loglog(q_test, cv_decel_rms, 'b-', label='CV', linewidth=2)
    axes3[1, 0].loglog(q_test, ca_decel_rms, 'g-', label='CA', linewidth=2)
    axes3[1, 0].set_xlabel('Q value')
    axes3[1, 0].set_ylabel('Velocity RMS (m/s)')
    axes3[1, 0].set_title('감속 구간 (1.6-2.0s)')
    axes3[1, 0].legend()
    axes3[1, 0].grid(True, alpha=0.3, which='both')
    
    # 전체 성능 vs Q
    total_cv_rms = []
    total_ca_rms = []
    
    for q in q_test:
        pos_cv, vel_cv = run_cv_filter(pos_meas, vel_true[0], dt_test, q, noise_std)
        pos_ca, vel_ca, _ = run_ca_filter(pos_meas, vel_true[0], 0, dt_test, q, noise_std)
        total_cv_rms.append(np.sqrt(np.mean((vel_cv - vel_true)**2)))
        total_ca_rms.append(np.sqrt(np.mean((vel_ca - vel_true)**2)))
    
    axes3[1, 1].loglog(q_test, total_cv_rms, 'b-', label='CV', linewidth=2)
    axes3[1, 1].loglog(q_test, total_ca_rms, 'g-', label='CA', linewidth=2)
    axes3[1, 1].set_xlabel('Q value')
    axes3[1, 1].set_ylabel('Velocity RMS (m/s)')
    axes3[1, 1].set_title('전체 구간')
    axes3[1, 1].legend()
    axes3[1, 1].grid(True, alpha=0.3, which='both')
    
    plt.suptitle('구간별 Q 민감도 분석 (100Hz)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.show()

if __name__ == "__main__":
    main()