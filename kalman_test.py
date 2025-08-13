import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
# 유니코드 마이너스 폰트 경고 제거
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False




from src.pytemplate.core.ang import *
from src.pytemplate.core.anal import linear_error_analysis
from src.pytemplate.utils.csv_reader import load_csv_data, list_csv_files
from src.pytemplate.utils.plot_helper import setup_korean_font, plot_timeseries, plot_arrays, plot_histograms

# --- 새 함수: 시간/각도 생성 ---
def generate_true_angle(rpm: float = 100, spp: int = 32, target_rev: int = 10):
    """주어진 rpm 으로 target_rev 회전하는 동안의 시간(t)과 각도(th) 생성.

    Args:
        rpm: 분당 회전수
        spp: 한 회전당 샘플 수 (samples per revolution)
        target_rev: 총 회전수
    Returns:
        t: 시간 배열 (초)
        th: 0~2π 로 wrap 된 각도 배열 (rad)
    """
    w = 2 * np.pi * rpm / 60.0            # 각속도 (rad/s)
    dt_spp = 60.0 / (spp * rpm)           # 샘플링 주기
    total_time = 60.0 / rpm * target_rev  # 총 시간
    t = np.arange(0.0, total_time, dt_spp)
    th = wrap_2pi(w * t)
    return t, th

# --- 시스템매틱(고조파) 엔코더 에러 추가 함수 ---
def add_harmonic_systematic_error(angle: np.ndarray,
                                  harmonics=(2,3,5),
                                  amps_deg=(1.0,0.7,0.5),
                                  phases=None,
                                  max_total_deg: float = 2.0,
                                  seed: int | None = 0):
    """엔코더 각도에 2,3,5 고조파 기반 시스템매틱 에러 추가.

    error(theta) = Σ A_k * sin(h_k * theta + φ_k)
    - 기본 앰프(도) 합이 2도 근처가 되도록 설정 (최대 max_total_deg 로 스케일)
    - angle: 0~2π 래핑된 진짜 각도 배열 (rad)

    Returns:
        angle_with_error: 시스템매틱 에러 적용 후 각도 (0~2π)
        error_array: 적용된 에러 (rad, -max_total_deg~+max_total_deg 근처)
    """
    amps_rad = np.deg2rad(np.array(amps_deg, dtype=float))
    if phases is None:
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, 2*np.pi, size=len(harmonics))
    phases = np.array(phases, dtype=float)
    error = np.zeros_like(angle, dtype=float)
    for h, a, ph in zip(harmonics, amps_rad, phases):
        error += a * np.sin(h * angle + ph)
    # 스케일이 최대 허용 초과 시 정규화
    peak_est = np.max(np.abs(error))
    max_allowed = np.deg2rad(max_total_deg)
    if peak_est > max_allowed and peak_est > 0:
        error *= (max_allowed / peak_est)
    angle_err = wrap_2pi(angle + error)
    return angle_err, error

# --- 가우시안 노이즈 추가 함수 ---
def add_gaussian_noise(angle: np.ndarray, std: float, seed: int | None = None) -> np.ndarray:
    """각도 배열에 가우시안 노이즈 추가 후 0~2π 래핑.

    Args:
        angle: 원본 각도 (0~2π 범위)
        std: 노이즈 표준편차 (rad)
        seed: 재현성 위한 시드
    Returns:
        noisy_angle: 노이즈가 추가된 각도 (0~2π 범위)
    """
    rng = np.random.default_rng(seed)
    noisy = wrap_2pi(angle + rng.normal(0.0, std, size=angle.shape))
    return noisy

# --- 양자화 함수 (엔코더 분해능 시뮬레이션) ---
def quantize_angle(angle: np.ndarray, bits: int) -> np.ndarray:
    """주어진 비트 수 분해능(2^bits) 로 각도를 양자화 (round to nearest code).
    angle: 0~2π 범위 (rad)
    returns: 0~2π 양자화 결과
    """
    levels = 1 << bits
    step = 2 * np.pi / levels
    # 라운드 후 wrap
    q = np.floor(angle / step + 0.5) * step
    return wrap_2pi(q)

# --- 선형 칼만필터 생성 (상태: [theta, omega], 측정: theta) ---
def create_angle_kf(dt: float, process_var: float, meas_var: float) -> KalmanFilter:
    """상태벡터 x=[theta, omega]^T, 측정 z=[theta] 모델 칼만필터 생성.

    F = [[1, dt],[0,1]] (등속 회전 모델)
    H = [[1, 0]] (각도만 관측)
    Q = q * [[dt^3/3, dt^2/2],[dt^2/2, dt]]  (q=process_var : 각가속도(각속도 변화) 분산)
    R = [meas_var]
    """
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.F = np.array([[1.0, dt],
                     [0.0, 1.0]])
    kf.H = np.array([[1.0, 0.0]])
    q = process_var
    kf.Q = q * np.array([[dt**3/3.0, dt**2/2.0],
                         [dt**2/2.0, dt]])
    kf.R = np.array([[meas_var]])
    kf.P = np.diag([10.0, 10.0])  # 초기 공분산 (불확실 크게)
    return kf

# --- CA(등가속) 모델 칼만필터 생성 (상태: [theta, omega, alpha], 측정: theta) ---
def create_angle_kf_ca(dt: float, process_var: float, meas_var: float) -> KalmanFilter:
    """Constant Acceleration model.
    상태 x = [θ, ω, α]^T
    F = [[1, dt, 0.5*dt^2],
         [0,  1,      dt  ],
         [0,  0,       1  ]]
    H = [1, 0, 0]
    Q (white jerk spectral density q=process_var):
      Q = q * [[dt^5/20, dt^4/8, dt^3/6],
               [dt^4/8,  dt^3/3, dt^2/2],
               [dt^3/6,  dt^2/2, dt     ]]
    """
    kf = KalmanFilter(dim_x=3, dim_z=1)
    kf.F = np.array([[1.0, dt, 0.5*dt*dt],
                     [0.0, 1.0, dt],
                     [0.0, 0.0, 1.0]])
    kf.H = np.array([[1.0, 0.0, 0.0]])
    q = process_var
    kf.Q = q * np.array([[dt**5/20.0, dt**4/8.0, dt**3/6.0],
                         [dt**4/8.0,  dt**3/3.0, dt**2/2.0],
                         [dt**3/6.0,  dt**2/2.0, dt]])
    kf.R = np.array([[meas_var]])
    kf.P = np.diag([1.0, 10.0, 100.0])  # 초기 공분산 적절히 감소
    return kf

# 신규 언랩 함수 (필터에서 언랩 각 사용)
def unwrap_sequence(wrapped: np.ndarray) -> np.ndarray:
    out = np.zeros_like(wrapped)
    acc = wrapped[0]
    out[0] = acc
    for i in range(1, len(wrapped)):
        acc += diffpi(wrapped[i], wrapped[i-1])
        out[i] = acc
    return out

if __name__ == "__main__":
    setup_korean_font()

    # ================= 로봇 액추에이터 가속도 시나리오 (0~1000 RPM) ONLY =================
    print('\n=== Robot Actuator Acceleration Scenarios (0~1000RPM) : CV vs CA (tuning + spec S-curve) ===')

    # ----- 일반적인 서보/모터 스펙 (예시 값) -----
    # (실제 시스템 값으로 교체 가능)
    T_peak_Nm = 2.5          # 피크 토크 (Nm)
    J_load = 0.002           # 부하 관성 (kg*m^2)
    gear_ratio = 10.0        # 감속비 (motor:load = 1:gear_ratio)
    J_motor = 0.00005        # 모터 로터 관성 (kg*m^2)
    # 부하 관성을 모터측 환산
    J_equiv = J_motor + J_load / (gear_ratio**2)
    alpha_max = T_peak_Nm / J_equiv              # rad/s^2 실제 최대 가속 근사
    alpha_max_rpm_s = alpha_max * 60/(2*np.pi)   # RPM/s 변환
    # jerk 한계 (경험적): 최대 가속을 40~60ms 정도에 도달하도록 설정
    t_rise_accel = 0.05                          # 가속 상승 목표 시간 (s)
    j_max = alpha_max / t_rise_accel             # rad/s^3
    j_max_rpm_s2 = j_max * 60/(2*np.pi)          # RPM/s^2
    print(f'[Spec] alpha_max ≈ {alpha_max_rpm_s:.1f} RPM/s, j_max ≈ {j_max_rpm_s2:.1f} RPM/s^2')

    # ----- 파라미터 -----
    noise_std = np.deg2rad(1)            # 센서 랜덤 노이즈 표준편차 (rad)
    dt2 = 0.0005                         # 2 kHz
    total_time = 4.0
    t2 = np.arange(0.0, total_time, dt2)
    bits2 = 10
    quant_step2 = 2*np.pi / (1<<bits2)
    quant_var2 = (quant_step2**2)/12.0   # 이상적 양자화 분산
    meas_var2 = noise_std**2 + quant_var2

    # q 후보 (CV: accel variance, CA: jerk spectral density)
    # 가속도 시나리오에 맞게 더 큰 범위 추가
    q_cv_list = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
    q_ca_list = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

    # Adaptive CA 옵션
    use_adaptive_ca = True
    adapt_innov_thresh_sigma = 4.0   # |innovation| > threshold * sqrt(R) 시 boost (덜 민감하게)
    adapt_boost_factor = 3.0         # 가속도 관련 process noise 임시 확대 배수 (적절히 감소)
    adapt_decay = 0.98               # 매 스텝 감쇠 (더 빠른 감쇠)

    def rpm_to_rad(rpm_val):
        return rpm_val * 2*np.pi / 60.0

    # ----- Jerk 제한 S-curve 가속 프로파일 생성 함수 -----
    def scurve_speed_profile(t: np.ndarray, rpm_start: float, rpm_end: float, accel_window: float,
                             alpha_lim_rpm_s: float, j_lim_rpm_s2: float):
        """단측(가속만) jerk 제한 S-curve 로 rpm_start -> rpm_end 가속 후 유지.
        - 7-seg full 구현이 아니라 (+j, 0, -j) 3세그 가속 (필요시 plateau) 구성.
        - Δv 가 너무 작아 plateau 불필요하면 삼각형( jerk up/down ) 패턴.
        accel_window: 가속이 완료되도록 허용되는 최대 시간 (s)
        alpha_lim_rpm_s, j_lim_rpm_s2: 한계 가속/jerk (RPM 계)
        """
        rpm_delta = rpm_end - rpm_start
        if rpm_delta <= 0:
            return np.full_like(t, rpm_start)
        # 기본 t_j (jerk 상승시간)
        t_j = alpha_lim_rpm_s / j_lim_rpm_s2
        # plateau 없는 삼각 조건 판단
        # 삼각 S-curve (jerk up + jerk down) 속도 변화: Δv_tri = alpha^2 / j
        delta_v_tri = alpha_lim_rpm_s**2 / j_lim_rpm_s2
        if rpm_delta <= delta_v_tri + 1e-9:
            # 목표 속도를 위해 필요한 피크 가속 alpha_peak 계산
            alpha_peak = (rpm_delta * j_lim_rpm_s2)**0.5
            t_j_eff = alpha_peak / j_lim_rpm_s2
            T_acc = 2 * t_j_eff
            if T_acc > accel_window:
                # accel_window 안에 맞추려면 jerk 확장 (alpha_peak 유지) 대신 alpha 축소
                alpha_peak = (rpm_delta * j_lim_rpm_s2 * (accel_window/(2*t_j_eff)))**0.5
                t_j_eff = alpha_peak / j_lim_rpm_s2
                T_acc = 2 * t_j_eff
            # 시간축 따라 속도 생성
            rpm_arr = np.zeros_like(t, dtype=float)
            for i, ti in enumerate(t):
                if ti <= t_j_eff:  # jerk up
                    a = j_lim_rpm_s2 * ti
                    v_inc = 0.5 * j_lim_rpm_s2 * ti**2
                elif ti <= 2*t_j_eff:  # jerk down
                    tau = ti - t_j_eff
                    a = alpha_peak - j_lim_rpm_s2 * tau
                    v_inc = 0.5 * j_lim_rpm_s2 * t_j_eff**2 + alpha_peak * tau - 0.5 * j_lim_rpm_s2 * tau**2
                else:
                    v_inc = rpm_delta
                rpm_arr[i] = rpm_start + min(v_inc, rpm_delta)
            return rpm_arr
        # plateau 존재하는 경우
        t_a = (rpm_delta - delta_v_tri) / alpha_lim_rpm_s  # 상수 가속 유지 시간
        T_acc = 2*t_j + t_a
        if T_acc > accel_window:
            # accel_window 안에 맞추도록 alpha 효율적 축소 -> scale 비율 s
            s = accel_window / T_acc
            alpha_eff = alpha_lim_rpm_s * s
            j_eff = j_lim_rpm_s2 * s    # 단순 스케일 (보수적)
            t_j_eff = alpha_eff / j_eff
            delta_v_tri_eff = alpha_eff**2 / j_eff
            t_a_eff = (rpm_delta - delta_v_tri_eff) / alpha_eff if rpm_delta > delta_v_tri_eff else 0.0
            if t_a_eff < 0: t_a_eff = 0.0
            T_acc_eff = 2*t_j_eff + t_a_eff
            # 재생성
            rpm_arr = np.zeros_like(t, dtype=float)
            for i, ti in enumerate(t):
                if ti <= t_j_eff:  # jerk up
                    v_inc = 0.5 * j_eff * ti**2
                elif ti <= t_j_eff + t_a_eff:  # const accel
                    tau = ti - t_j_eff
                    v_inc = 0.5 * j_eff * t_j_eff**2 + alpha_eff * tau
                elif ti <= 2*t_j_eff + t_a_eff:  # jerk down
                    tau = ti - (t_j_eff + t_a_eff)
                    v_inc = (0.5 * j_eff * t_j_eff**2 + alpha_eff * t_a_eff +
                              alpha_eff * tau - 0.5 * j_eff * tau**2)
                else:
                    v_inc = rpm_delta
                rpm_arr[i] = rpm_start + min(v_inc, rpm_delta)
            return rpm_arr
        # 정상 케이스 (스펙 한계 내에서 목표 달성)
        rpm_arr = np.zeros_like(t, dtype=float)
        for i, ti in enumerate(t):
            if ti <= t_j:  # jerk up
                v_inc = 0.5 * j_lim_rpm_s2 * ti**2
            elif ti <= t_j + t_a:  # const accel
                tau = ti - t_j
                v_inc = 0.5 * j_lim_rpm_s2 * t_j**2 + alpha_lim_rpm_s * tau
            elif ti <= 2*t_j + t_a:  # jerk down
                tau = ti - (t_j + t_a)
                v_inc = (0.5 * j_lim_rpm_s2 * t_j**2 + alpha_lim_rpm_s * t_a +
                          alpha_lim_rpm_s * tau - 0.5 * j_lim_rpm_s2 * tau**2)
            else:
                v_inc = rpm_delta
            rpm_arr[i] = rpm_start + min(v_inc, rpm_delta)
        return rpm_arr

    # 기존 간단 프로파일 + 스펙기반 S-curve 추가
    def prof_const_500(t): return np.full_like(t, 500.0)
    def prof_scurve_0_1000(t):
        accel_window = 1.5  # 1.5초 안에 0->1000RPM
        return scurve_speed_profile(t, 0.0, 1000.0, accel_window, alpha_max_rpm_s, j_max_rpm_s2)
    def prof_scurve_200_800(t):
        accel_window = 1.0  # 1초 안에 200->800RPM
        base = scurve_speed_profile(t, 200.0, 800.0, accel_window, alpha_max_rpm_s, j_max_rpm_s2)
        return base
    def prof_sinusoidal(t): return 500.0 + 300.0 * np.sin(2*np.pi * t / total_time)

    # 현실성 낮은 ramp/trapezoid/step 제거
    profiles = [
        ("Const 500RPM", prof_const_500),
        ("S-curve 0->1000", prof_scurve_0_1000),
        ("S-curve 200->800", prof_scurve_200_800),
        ("Sin(500±300)", prof_sinusoidal),
    ]

    def quantize_angle(angle: np.ndarray, bits: int) -> np.ndarray:  # local
        levels = 1 << bits
        step = 2 * np.pi / levels
        q = np.floor(angle / step + 0.5) * step
        return wrap_2pi(q)

    def integrate_angle_from_rpm(rpm_func, t):
        rpm_series = rpm_func(t)
        omega = rpm_to_rad(rpm_series)
        theta = np.cumsum(omega) * dt2
        return wrap_2pi(theta), rpm_series, omega

    agg_rows = []
    tuning_records = []  # (profile, model, q, rms_deg)

    for name, fn in profiles:
        th2, rpm_series, omega_series = integrate_angle_from_rpm(fn, t2)          # wrapped true angle
        th2_unwrap = unwrap_sequence(th2)                                         # unwrapped true angle
        # 시스템매틱 + 노이즈 + 양자화 (wrapped)
        th2_sys, sys2_err = add_harmonic_systematic_error(th2, harmonics=(2,3,5), amps_deg=(1.0,0.7,0.5), max_total_deg=2.0, seed=11)
        noisy2 = add_gaussian_noise(th2_sys, noise_std, seed=12)
        meas2 = quantize_angle(noisy2, bits2)
        meas2_unwrap = unwrap_sequence(meas2)

        # Measurement error baseline (wrap 기반 비교 유지)
        err_meas = diffpi(meas2, th2)
        rms_meas = np.degrees(np.sqrt(np.mean(err_meas**2)))

        # -------- CV q 스윕 (언랩 잔차) --------
        best_cv = (None, 1e9, None, None)
        for q_cv in q_cv_list:
            kf_cv = create_angle_kf(dt2, q_cv, meas_var2)
            # 상태 각도 언랩 값 사용
            kf_cv.x = np.array([[meas2_unwrap[0]], [0.0]])
            H_cv = kf_cv.H; I_cv = np.eye(2)
            est_th_cv = np.zeros_like(meas2_unwrap)
            est_om_cv = np.zeros_like(meas2_unwrap)
            for i, z_unwrap in enumerate(meas2_unwrap):
                if i>0:
                    kf_cv.predict()
                y = z_unwrap - kf_cv.x[0,0]   # 언랩 잔차
                S = H_cv @ kf_cv.P @ H_cv.T + kf_cv.R
                K = kf_cv.P @ H_cv.T / S
                kf_cv.x = kf_cv.x + K * y
                # 언랩 상태 -> wrap 제거 필요 없음
                kf_cv.P = (I_cv - K @ H_cv) @ kf_cv.P
                est_th_cv[i] = kf_cv.x[0,0]
                est_om_cv[i] = kf_cv.x[1,0]
            # 에러는 wrap 비교 (기존 기준 유지)
            est_th_cv_wrapped = wrap_2pi(est_th_cv)
            err_cv = diffpi(est_th_cv_wrapped, th2)
            rms_cv = np.degrees(np.sqrt(np.mean(err_cv**2)))
            tuning_records.append((name, 'CV', q_cv, rms_cv))
            if rms_cv < best_cv[1]:
                best_cv = (q_cv, rms_cv, est_th_cv, est_om_cv)

        # -------- CA q 스윕 (언랩 잔차) --------
        best_ca = (None, 1e9, None, None, None)
        for q_ca in q_ca_list:
            kf_ca = create_angle_kf_ca(dt2, q_ca, meas_var2)
            kf_ca.x = np.array([[meas2_unwrap[0]], [0.0], [0.0]])
            H_ca = kf_ca.H; I_ca = np.eye(3)
            est_th_ca = np.zeros_like(meas2_unwrap)
            est_om_ca = np.zeros_like(meas2_unwrap)
            est_al_ca = np.zeros_like(meas2_unwrap)
            for i, z_unwrap in enumerate(meas2_unwrap):
                if i>0:
                    kf_ca.predict()
                y = z_unwrap - kf_ca.x[0,0]
                S = H_ca @ kf_ca.P @ H_ca.T + kf_ca.R
                K = kf_ca.P @ H_ca.T / S
                kf_ca.x = kf_ca.x + K * y
                kf_ca.P = (I_ca - K @ H_ca) @ kf_ca.P
                est_th_ca[i] = kf_ca.x[0,0]
                est_om_ca[i] = kf_ca.x[1,0]
                est_al_ca[i] = kf_ca.x[2,0]
            est_th_ca_wrapped = wrap_2pi(est_th_ca)
            err_ca = diffpi(est_th_ca_wrapped, th2)
            rms_ca = np.degrees(np.sqrt(np.mean(err_ca**2)))
            tuning_records.append((name, 'CA', q_ca, rms_ca))
            if rms_ca < best_ca[1]:
                best_ca = (q_ca, rms_ca, est_th_ca, est_om_ca, est_al_ca)

        # -------- Adaptive CA (언랩 기반) --------
        if use_adaptive_ca and best_ca[0] is not None:
            base_q_ca = best_ca[0]
            kf_ca_ad = create_angle_kf_ca(dt2, base_q_ca, meas_var2)
            kf_ca_ad.x = np.array([[meas2_unwrap[0]], [0.0], [0.0]])
            H_ca = kf_ca_ad.H; I_ca = np.eye(3)
            est_th_ca_ad = np.zeros_like(meas2_unwrap)
            est_om_ca_ad = np.zeros_like(meas2_unwrap)
            est_al_ca_ad = np.zeros_like(meas2_unwrap)
            boost = 1.0
            thresh = adapt_innov_thresh_sigma * np.sqrt(meas_var2)
            base_Q_unit = np.array([[dt2**5/20.0, dt2**4/8.0, dt2**3/6.0],
                                    [dt2**4/8.0,  dt2**3/3.0, dt2**2/2.0],
                                    [dt2**3/6.0,  dt2**2/2.0, dt2]])
            for i, z_unwrap in enumerate(meas2_unwrap):
                if i>0:
                    kf_ca_ad.Q = base_q_ca * boost * base_Q_unit
                    kf_ca_ad.predict()
                y = z_unwrap - kf_ca_ad.x[0,0]
                S = H_ca @ kf_ca_ad.P @ H_ca.T + kf_ca_ad.R
                K = kf_ca_ad.P @ H_ca.T / S
                kf_ca_ad.x = kf_ca_ad.x + K * y
                kf_ca_ad.P = (I_ca - K @ H_ca) @ kf_ca_ad.P
                est_th_ca_ad[i] = kf_ca_ad.x[0,0]
                est_om_ca_ad[i] = kf_ca_ad.x[1,0]
                est_al_ca_ad[i] = kf_ca_ad.x[2,0]
                if abs(y) > thresh:
                    boost = max(boost, adapt_boost_factor)
                boost = 1.0 + (boost - 1.0) * adapt_decay
            est_th_ca_ad_wrapped = wrap_2pi(est_th_ca_ad)
            err_ca_ad = diffpi(est_th_ca_ad_wrapped, th2)
            rms_ca_ad = np.degrees(np.sqrt(np.mean(err_ca_ad**2)))
        else:
            rms_ca_ad = None
            est_th_ca_ad = est_om_ca_ad = est_al_ca_ad = None

        # 최종 선택
        q_cv_best, rms_cv_best, est_th_cv_best_unwrap, est_om_cv_best = best_cv
        q_ca_best, rms_ca_best, est_th_ca_best_unwrap, est_om_ca_best, est_al_ca_best = best_ca
        gain_cv = rms_meas / rms_cv_best if rms_cv_best>0 else np.nan
        gain_ca = rms_meas / rms_ca_best if rms_ca_best>0 else np.nan
        gain_ca_ad = (rms_meas / rms_ca_ad) if (rms_ca_ad and rms_ca_ad>0) else None
        agg_rows.append((name, rms_meas, rms_cv_best, rms_ca_best, rms_ca_ad, gain_cv, gain_ca, gain_ca_ad, q_cv_best, q_ca_best))

        # Plot (wrapped 시각화)
        est_th_cv_plot = wrap_2pi(est_th_cv_best_unwrap)
        est_th_ca_plot = wrap_2pi(est_th_ca_best_unwrap)
        if est_th_ca_ad is not None:
            est_th_ca_ad_plot = wrap_2pi(est_th_ca_ad)

        fig, axes = plt.subplots(5,1, figsize=(10,12), sharex=True)
        axes[0].plot(t2, rpm_series, label='True RPM', color='k')
        axes[0].plot(t2, est_om_cv_best * 60/(2*np.pi), label=f'CV RPM (q={q_cv_best:g})', alpha=0.7)
        axes[0].plot(t2, est_om_ca_best * 60/(2*np.pi), label=f'CA RPM (q={q_ca_best:g})', alpha=0.7)
        if rms_ca_ad is not None:
            axes[0].plot(t2, est_om_ca_ad * 60/(2*np.pi), label='CA RPM adaptive', alpha=0.7, linestyle='--')
        axes[0].set_ylabel('RPM'); axes[0].set_title(f'{name} Speed'); axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(t2, np.degrees(err_meas), label='Meas Err', color='gray', alpha=0.35)
        axes[1].plot(t2, np.degrees(diffpi(est_th_cv_plot, th2)), label='CV Err', color='tab:blue', linewidth=0.8)
        axes[1].plot(t2, np.degrees(diffpi(est_th_ca_plot, th2)), label='CA Err', color='tab:green', linewidth=0.8)
        if rms_ca_ad is not None:
            axes[1].plot(t2, np.degrees(diffpi(est_th_ca_ad_plot, th2)), label='CA Err adaptive', color='tab:olive', linewidth=0.8)
        axes[1].set_ylabel('Angle Err (deg)'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

        accel_true = np.gradient(rpm_to_rad(rpm_series), dt2) * 60/(2*np.pi)
        accel_ca_best = est_al_ca_best * 60/(2*np.pi)
        axes[2].plot(t2, accel_true, label='True Accel', color='purple')
        axes[2].plot(t2, accel_ca_best, label='CA α', color='tab:green', alpha=0.8)
        if rms_ca_ad is not None:
            axes[2].plot(t2, est_al_ca_ad * 60/(2*np.pi), label='CA α adaptive', color='tab:olive', alpha=0.8)
        axes[2].set_ylabel('Accel (RPM/s)'); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

        axes[3].plot(t2, np.abs(np.degrees(diffpi(est_th_cv_plot, th2))), label='|CV Err|', color='tab:blue', alpha=0.6)
        axes[3].plot(t2, np.abs(np.degrees(diffpi(est_th_ca_plot, th2))), label='|CA Err|', color='tab:green', alpha=0.6)
        if rms_ca_ad is not None:
            axes[3].plot(t2, np.abs(np.degrees(diffpi(est_th_ca_ad_plot, th2))), label='|CA Err| adaptive', color='tab:olive', alpha=0.6)
        axes[3].set_ylabel('|Err| (deg)'); axes[3].legend(fontsize=8); axes[3].grid(alpha=0.3)

        # RMS vs q tuning curves
        cv_q_vals = [r[2] for r in tuning_records if r[0]==name and r[1]=='CV']
        cv_rms_vals = [r[3] for r in tuning_records if r[0]==name and r[1]=='CV']
        ca_q_vals = [r[2] for r in tuning_records if r[0]==name and r[1]=='CA']
        ca_rms_vals = [r[3] for r in tuning_records if r[0]==name and r[1]=='CA']
        axes[4].semilogx(cv_q_vals, cv_rms_vals, marker='o', label='CV RMS')
        axes[4].semilogx(ca_q_vals, ca_rms_vals, marker='s', label='CA RMS')
        axes[4].axhline(rms_meas, color='gray', linestyle='--', label='Meas RMS')
        axes[4].axvline(q_cv_best, color='tab:blue', linestyle=':', alpha=0.6)
        axes[4].axvline(q_ca_best, color='tab:green', linestyle=':', alpha=0.6)
        axes[4].set_xlabel('q'); axes[4].set_ylabel('RMS (deg)'); axes[4].legend(fontsize=8); axes[4].grid(alpha=0.3, which='both')

        fig.suptitle(f'{name}: Meas {rms_meas:.3f}° | CV {rms_cv_best:.3f}° (q={q_cv_best:g}, Gx{rms_meas/rms_cv_best:.2f}) | CA {rms_ca_best:.3f}° (q={q_ca_best:g}, Gx{rms_meas/rms_ca_best:.2f})' + (f' | CA_ad {rms_ca_ad:.3f}° (Gx{rms_meas/rms_ca_ad:.2f})' if rms_ca_ad else ''))
        fig.tight_layout(rect=[0,0,1,0.94])
        print(f'{name}: Meas={rms_meas:.4f}° | CV best q={q_cv_best:g} RMS={rms_cv_best:.4f}° | CA best q={q_ca_best:g} RMS={rms_ca_best:.4f}°' + (f' | CA adaptive RMS={rms_ca_ad:.4f}°' if rms_ca_ad else ''))

    # Aggregate comparison
    labels = [r[0] for r in agg_rows]
    x = np.arange(len(labels))
    width = 0.22
    rms_meas_vals = [r[1] for r in agg_rows]
    rms_cv_vals = [r[2] for r in agg_rows]
    rms_ca_vals = [r[3] for r in agg_rows]
    rms_ca_ad_vals = [r[4] if r[4] is not None else np.nan for r in agg_rows]
    plt.figure(figsize=(11,5))
    plt.bar(x - 1.5*width, rms_meas_vals, width, label='Meas', color='gray', alpha=0.5)
    plt.bar(x - 0.5*width, rms_cv_vals, width, label='CV best', color='tab:blue')
    plt.bar(x + 0.5*width, rms_ca_vals, width, label='CA best', color='tab:green')
    plt.bar(x + 1.5*width, rms_ca_ad_vals, width, label='CA adaptive', color='tab:olive')
    plt.xticks(x, labels, rotation=15)
    plt.ylabel('RMS Angle Error (deg)')
    plt.title('Scenario RMS Comparison (Best Tuned)')
    plt.legend(fontsize=8); plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()

    print('\nSummary (Best):')
    for row in agg_rows:
        name, rms_meas, rms_cv_b, rms_ca_b, rms_ca_ad, g_cv, g_ca, g_ca_ad, qcvb, qcab = row
        print(f'{name:14s} | Meas {rms_meas:6.3f}° | CV {rms_cv_b:6.3f}° (q={qcvb:g}) | CA {rms_ca_b:6.3f}° (q={qcab:g})' + (f' | CA_ad {rms_ca_ad:6.3f}°' if rms_ca_ad else ''))

    plt.show()