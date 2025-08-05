import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import sys
sys.path.append('src')
from pytemplate.core import ang
from pytemplate.core.anal import linear_error_analysis
from pytemplate.core.kalman import AngleKalmanFilter, SingleAngleKalmanFilter
from pytemplate.utils.csv_reader import load_csv_data, list_csv_files
from pytemplate.utils.plot_helper import setup_korean_font, plot_timeseries, plot_arrays, plot_histograms

def main():
    setup_korean_font()
    
    # CSV 파일 읽기
    df = load_csv_data('wheel_angle_results.csv')
    
    # 독립적인 두 센서만 사용 (M_abs, N_abs)
    m_abs_radians = df['M_abs'].values / 32  # 라디안 변환
    n_abs_radians = df['N_abs'].values / 31  # 라디안 변환

    # 분산 기반 가중치 계산
    # 1단계: 각 센서의 임시 에러 계산 (가중치 결정용)
    m_temp_error = linear_error_analysis(m_abs_radians)
    n_temp_error = linear_error_analysis(n_abs_radians)
    
    # 2단계: 각 센서의 분산 계산
    var_m = np.var(m_temp_error)
    var_n = np.var(n_temp_error)
    
    # 3단계: 가중치 계산 (분산의 역수)
    w_m = 1/var_m
    w_n = 1/var_n
    
    # 4단계: 정규화 (합이 1이 되도록)
    total_weight = w_m + w_n
    w_m_norm = w_m / total_weight
    w_n_norm = w_n / total_weight
    
    # 5단계: 동일 가중치 차이 기반 평균 계산
    da12 = ang.diffpi(n_abs_radians, m_abs_radians)
    
    # 차이 기반 평균: a_avg = a1 + C2*da12  
    # 동일 가중치: C1=1, C2=0.5
    C2 = 0.5
    
    # 동일 가중치 평균 (기준은 m_abs_radians)
    weighted_avg_radians = m_abs_radians + C2*da12
    
    # 가중치 정보 출력
    print(f"분산 정보:")
    print(f"M_abs: 분산 = {var_m:.8f}")
    print(f"N_abs: 분산 = {var_n:.8f}")
    print(f"동일 가중치 차이 기반 평균: C1=1, C2={C2:.1f}")
    
    # 칼만필터 적용
    print("\n=== 칼만필터 적용 ===")
    
    # 센서 노이즈를 분산 기반으로 설정
    kf = AngleKalmanFilter(dt=1.0, 
                          process_noise=1e-6,
                          sensor1_noise=var_m, 
                          sensor2_noise=var_n)
    
    # 이중 센서 칼만필터로 데이터 처리
    kalman_filtered_radians = kf.filter_data(m_abs_radians, n_abs_radians)
    
    # 단일 센서 칼만필터 적용 (M_abs만 사용)
    single_kf = SingleAngleKalmanFilter(dt=1.0, 
                                       process_noise=1e-7,
                                       sensor_noise=var_m)
    single_kalman_filtered_radians = single_kf.filter_data(m_abs_radians)
    
    # 선형 에러 분석 비교
    m_error = np.rad2deg(linear_error_analysis(m_abs_radians))
    n_error = np.rad2deg(linear_error_analysis(n_abs_radians))
    weighted_error = np.rad2deg(linear_error_analysis(weighted_avg_radians))
    kalman_error = np.rad2deg(linear_error_analysis(kalman_filtered_radians))
    single_kalman_error = np.rad2deg(linear_error_analysis(single_kalman_filtered_radians))
    
    # 에러 시계열 비교
    plot_arrays([m_error, kalman_error, single_kalman_error], 
                labels=['M_abs 원본', '이중센서 칼만', '단일센서 칼만'], 
                title='원본 vs 이중센서칼만 vs 단일센서칼만 선형 에러 비교')
    
    # 히스토그램 비교
    plot_histograms([m_error, kalman_error, single_kalman_error], 
                   labels=['M_abs 원본', '이중센서 칼만', '단일센서 칼만'], 
                   title='에러 분포 비교: 원본 vs 이중센서칼만 vs 단일센서칼만')
    
    # RMS 및 개선도 출력
    m_rms = np.sqrt(np.mean(m_error**2))
    weighted_rms = np.sqrt(np.mean(weighted_error**2))
    kalman_rms = np.sqrt(np.mean(kalman_error**2))
    single_kalman_rms = np.sqrt(np.mean(single_kalman_error**2))
    
    weighted_improvement = (m_rms - weighted_rms) / m_rms * 100
    kalman_improvement = (m_rms - kalman_rms) / m_rms * 100
    single_kalman_improvement = (m_rms - single_kalman_rms) / m_rms * 100
    
    print(f"M_abs 원본 에러 RMS: {m_rms:.6f} 도")
    print(f"가중평균 에러 RMS: {weighted_rms:.6f} 도 (개선도: {weighted_improvement:.2f}%)")
    print(f"이중센서 칼만 에러 RMS: {kalman_rms:.6f} 도 (개선도: {kalman_improvement:.2f}%)")
    print(f"단일센서 칼만 에러 RMS: {single_kalman_rms:.6f} 도 (개선도: {single_kalman_improvement:.2f}%)")
    print("분석 그래프들이 열렸습니다!")
    
    # 창이 열려있는 동안 프로그램이 종료되지 않도록 대기
    plt.show()  


if __name__ == "__main__":
    main()
