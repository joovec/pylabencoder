import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter




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

if __name__ == "__main__":
    setup_korean_font()

    # 파라미터
    rpm = 100
    spp = 1024
    target_rev = 10
    noise_std = np.deg2rad(1)  # 측정 표준편차 (rad)

    # 시간/각도 생성
    t, th = generate_true_angle(rpm=rpm, spp=spp, target_rev=target_rev)

    # 시스템매틱 엔코더 에러 추가 (2,3,5 고조파)
    th_sys, sys_err = add_harmonic_systematic_error(th, harmonics=(2,3,5), amps_deg=(1.0,0.7,0.5), max_total_deg=2.0, seed=7)

    # 랜덤 노이즈 추가
    noisy_th = add_gaussian_noise(th_sys, noise_std, seed=42)

    dt = 60.0 / (spp * rpm)

    meas_var = noise_std**2

    # ---------------- q 리스트 반복 실행 (Kalman vs systematic vs measurement error 비교) ----------------
    q_list = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4]
    kalman_errors = []  # (q, err_est(rad), est_theta, est_omega, rms_err)

    # 공통 에러 (측정, 시스템매틱)
    err_meas = diffpi(noisy_th, th)          # measurement error vs true
    sys_err_true = diffpi(th_sys, th)        # pure systematic (rad)

    for q in q_list:
        kf = create_angle_kf(dt, q, meas_var)
        kf.x = np.array([[noisy_th[0]], [0.0]])
        H = kf.H; I = np.eye(2)
        est_theta = np.zeros_like(noisy_th)
        est_omega = np.zeros_like(noisy_th)
        for i, z in enumerate(noisy_th):
            if i > 0:
                kf.predict()
            y = diffpi(z, kf.x[0, 0])
            S = H @ kf.P @ H.T + kf.R
            K = kf.P @ H.T / S
            kf.x = kf.x + K * y
            kf.x[0, 0] = wrap_2pi(kf.x[0, 0])
            kf.P = (I - K @ H) @ kf.P
            est_theta[i] = kf.x[0, 0]
            est_omega[i] = kf.x[1, 0]
        err_est = diffpi(est_theta, th)
        rms_err = float(np.sqrt(np.mean(err_est**2)))
        kalman_errors.append((q, err_est, est_theta, est_omega, rms_err))

    # 단위 변환
    th_deg = np.degrees(th)
    th_sys_deg = np.degrees(th_sys)
    noisy_deg = np.degrees(noisy_th)
    sys_err_deg = np.degrees(sys_err_true)
    err_meas_deg = np.degrees(err_meas)

    # 플롯 1: 각도 (참, 시스템매틱, 측정)
    plt.figure(figsize=(12,6))
    plt.plot(t, th_deg, label='True', linewidth=2, color='k')
    plt.plot(t, th_sys_deg, label='Systematic Added', alpha=0.7)
    plt.plot(t, noisy_deg, label='Systematic+Noise (Measured)', alpha=0.4)
    plt.xlabel('Time (s)'); plt.ylabel('Angle (deg)'); plt.title('Angle Signals')
    plt.legend(); plt.grid(alpha=0.3)

    # 플롯 2: 에러 비교 (Systematic, Measurement, Kalman(q...))
    plt.figure(figsize=(12,6))
    plt.plot(t, sys_err_deg, label='Systematic Error (true)', linewidth=2, color='orange')
    plt.plot(t, err_meas_deg, label='Measurement Error', alpha=0.5, color='gray')
    cmap = plt.cm.get_cmap('viridis', len(kalman_errors))
    for idx, (q, err_est, *_rest) in enumerate(kalman_errors):
        plt.plot(t, np.degrees(err_est), label=f'Kalman q={q:g}', color=cmap(idx), linewidth=1)
    plt.xlabel('Time (s)'); plt.ylabel('Error (deg)'); plt.title('Error vs Time for Different q')
    plt.legend(ncol=2, fontsize=8); plt.grid(alpha=0.3)

    # 플롯 3: Kalman RMS vs q (log q)
    plt.figure(figsize=(8,4))
    qs = [q for q, *_ in kalman_errors]
    rms_deg = [np.degrees(r) for *_, r in kalman_errors]
    plt.semilogx(qs, rms_deg, marker='o')
    meas_rms_deg = np.degrees(np.sqrt(np.mean(err_meas**2)))
    sys_rms_deg = np.degrees(np.sqrt(np.mean(sys_err_true**2)))
    plt.axhline(meas_rms_deg, color='gray', linestyle='--', label=f'Meas RMS {meas_rms_deg:.3f}°')
    plt.axhline(sys_rms_deg, color='orange', linestyle='--', label=f'Systematic RMS {sys_rms_deg:.3f}°')
    plt.xlabel('q (process noise)'); plt.ylabel('RMS Error (deg)')
    plt.title('Kalman RMS Error vs q')
    plt.legend(); plt.grid(alpha=0.3, which='both')

    # 플롯 4: 선택 q 에 대한 추정 각속도 (rpm)
    plt.figure(figsize=(12,4))
    select_idx = [0, len(kalman_errors)//2, -1]
    for si in select_idx:
        q, _err_est, _est_theta, est_omega, _rms = kalman_errors[si]
        plt.plot(t, est_omega * 60/(2*np.pi), label=f'q={q:g}')
    plt.hlines(rpm, t[0], t[-1], colors='k', linestyles='--', label='True RPM')
    plt.xlabel('Time (s)'); plt.ylabel('RPM'); plt.title('Estimated RPM (selected q)')
    plt.legend(); plt.grid(alpha=0.3)

    # 콘솔 요약 출력
    print('\nKalman q vs RMS (rad / deg):')
    for q, _err_est, _est_theta, _est_omega, rms_err in kalman_errors:
        print(f'q={q:>8g}  RMS={rms_err:.6e} rad ({np.degrees(rms_err):.4f}°)')

    plt.tight_layout()
    plt.show()