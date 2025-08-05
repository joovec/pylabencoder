# -*- coding: utf-8 -*-
"""엔코더 각도용 칼만필터 구현."""
import numpy as np
from . import ang


class AngleKalmanFilter:
    """2개 엔코더 센서를 위한 각도 칼만필터.
    
    상태: [각도, 각속도]
    관측: [센서1_각도, 센서2_각도]
    """
    
    def __init__(self, dt=1.0, process_noise=1e-5, sensor1_noise=1e-6, sensor2_noise=1e-6):
        """칼만필터 초기화.
        
        Args:
            dt: 샘플링 시간 간격
            process_noise: 프로세스 노이즈 분산
            sensor1_noise: 센서1 관측 노이즈 분산  
            sensor2_noise: 센서2 관측 노이즈 분산
        """
        # 상태 벡터: [각도, 각속도]
        self.x = np.zeros(2)  # 초기 상태
        
        # 상태 전이 행렬 (등속도 모델)
        self.F = np.array([[1.0, dt],
                          [0.0, 1.0]])
        
        # 관측 행렬 (두 센서 모두 각도만 관측)
        self.H = np.array([[1.0, 0.0],
                          [1.0, 0.0]])
        
        # 프로세스 노이즈 공분산 행렬
        self.Q = np.array([[dt**4/4, dt**3/2],
                          [dt**3/2, dt**2]]) * process_noise
        
        # 관측 노이즈 공분산 행렬
        self.R = np.array([[sensor1_noise, 0.0],
                          [0.0, sensor2_noise]])
        
        # 상태 공분산 행렬 (초기값)
        self.P = np.eye(2) * 1.0
        
        # 첫 번째 업데이트 플래그
        self.initialized = False
        
    def predict(self):
        """예측 단계."""
        # 상태 예측
        self.x = self.F @ self.x
        
        # 각도를 [-π, π] 범위로 래핑
        self.x[0] = ang.wrap_pi(self.x[0])
        
        # 공분산 예측
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, z1, z2):
        """업데이트 단계.
        
        Args:  
            z1: 센서1 관측값 (각도)
            z2: 센서2 관측값 (각도)
        """
        # 관측 벡터
        z = np.array([z1, z2])
        
        # 첫 번째 업데이트인 경우 초기화
        if not self.initialized:
            # 두 센서의 평균으로 초기 각도 설정
            avg_angle = z1 + 0.5 * ang.diffpi(z2, z1)
            self.x[0] = avg_angle
            self.x[1] = 0.0  # 초기 각속도는 0
            self.initialized = True
            return self.x[0]
        
        # 예상 관측값 (두 센서 모두 현재 추정 각도)
        h = np.array([self.x[0], self.x[0]])
        
        # 잔차 계산 (각도 차이 고려)
        y = np.array([ang.diffpi(z1, h[0]), 
                     ang.diffpi(z2, h[1])])
        
        # 잔차 공분산
        S = self.H @ self.P @ self.H.T + self.R
        
        # 칼만 게인
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 상태 업데이트
        self.x = self.x + K @ y
        
        # 각도를 [-π, π] 범위로 래핑
        self.x[0] = ang.wrap_pi(self.x[0])
        
        # 공분산 업데이트
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P
        
        return self.x[0]  # 추정된 각도 반환
        
    def filter_data(self, sensor1_data, sensor2_data):
        """전체 데이터를 칼만필터로 처리.
        
        Args:
            sensor1_data: 센서1 각도 데이터 배열
            sensor2_data: 센서2  각도 데이터 배열
            
        Returns:
            numpy array: 필터링된 각도 데이터
        """
        filtered_angles = np.zeros(len(sensor1_data))
        
        for i in range(len(sensor1_data)):
            # 예측 단계
            if i > 0:  # 첫 번째는 예측 없이 시작
                self.predict()
            
            # 업데이트 단계
            filtered_angles[i] = self.update(sensor1_data[i], sensor2_data[i])
            
        return filtered_angles


class SingleAngleKalmanFilter:
    """단일 엔코더 센서를 위한 3차 칼만필터.
    
    상태: [각도, 각속도, 각가속도]
    관측: [센서_각도]
    """
    
    def __init__(self, dt=1.0, process_noise=1e-6, sensor_noise=1e-6):
        """칼만필터 초기화.
        
        Args:
            dt: 샘플링 시간 간격
            process_noise: 프로세스 노이즈 분산 (각가속도 변화)
            sensor_noise: 센서 관측 노이즈 분산
        """
        # 상태 벡터: [각도, 각속도, 각가속도]
        self.x = np.zeros(3)
        
        # 상태 전이 행렬 (등가속도 모델)
        self.F = np.array([[1.0, dt, 0.5*dt**2],
                          [0.0, 1.0, dt],
                          [0.0, 0.0, 1.0]])
        
        # 관측 행렬 (각도만 관측)
        self.H = np.array([[1.0, 0.0, 0.0]])
        
        # 프로세스 노이즈 공분산 행렬 (각가속도의 변화만 고려)
        G = np.array([[0.5*dt**2], [dt], [1.0]])  # 노이즈 입력 행렬
        self.Q = G @ G.T * process_noise
        
        # 관측 노이즈 공분산 행렬
        self.R = np.array([[sensor_noise]])
        
        # 상태 공분산 행렬 (초기값)
        self.P = np.eye(3) * 1.0
        
        # 첫 번째 업데이트 플래그
        self.initialized = False
        
    def predict(self):
        """예측 단계."""
        # 상태 예측
        self.x = self.F @ self.x
        
        # 각도를 [-π, π] 범위로 래핑
        self.x[0] = ang.wrap_pi(self.x[0])
        
        # 공분산 예측
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, z):
        """업데이트 단계.
        
        Args:
            z: 센서 관측값 (각도)
        """
        # 첫 번째 업데이트인 경우 초기화
        if not self.initialized:
            self.x[0] = z  # 초기 각도
            self.x[1] = 0.0  # 초기 각속도
            self.x[2] = 0.0  # 초기 각가속도
            self.initialized = True
            return self.x[0]
        
        # 예상 관측값
        h = self.H @ self.x
        
        # 잔차 계산 (각도 차이 고려)
        y = ang.diffpi(z, h[0])
        
        # 잔차 공분산
        S = self.H @ self.P @ self.H.T + self.R
        
        # 칼만 게인
        K = (self.P @ self.H.T) / S[0, 0]
        
        # 상태 업데이트
        self.x = self.x + K.flatten() * y
        
        # 각도를 [-π, π] 범위로 래핑
        self.x[0] = ang.wrap_pi(self.x[0])
        
        # 공분산 업데이트
        I = np.eye(3)
        self.P = (I - K @ self.H) @ self.P
        
        return float(self.x[0])  # 추정된 각도 반환 (스칼라)
        
    def filter_data(self, sensor_data):
        """전체 데이터를 칼만필터로 처리.
        
        Args:
            sensor_data: 센서 각도 데이터 배열
            
        Returns:
            numpy array: 필터링된 각도 데이터
        """
        filtered_angles = np.zeros(len(sensor_data))
        
        for i in range(len(sensor_data)):
            # 예측 단계
            if i > 0:  # 첫 번째는 예측 없이 시작
                self.predict()
            
            # 업데이트 단계
            filtered_angles[i] = self.update(sensor_data[i])
            
        return filtered_angles


if __name__ == "__main__":
    # 테스트
    print("AngleKalmanFilter 테스트")
    
    # 테스트 데이터 생성
    true_angles = np.linspace(0, 4*np.pi, 100)  # 2바퀴 회전
    noise1 = np.random.normal(0, 0.01, 100)
    noise2 = np.random.normal(0, 0.015, 100)
    
    sensor1_data = true_angles + noise1
    sensor2_data = true_angles + noise2
    
    # 이중 센서 칼만필터 적용
    kf = AngleKalmanFilter(dt=1.0, process_noise=1e-4, 
                          sensor1_noise=1e-4, sensor2_noise=2e-4)
    
    filtered = kf.filter_data(sensor1_data, sensor2_data)
    
    print(f"이중센서 - 원본 RMS: {np.sqrt(np.mean((true_angles - sensor1_data)**2)):.6f}")
    print(f"이중센서 - 필터 RMS: {np.sqrt(np.mean((true_angles - filtered)**2)):.6f}")
    
    # 단일 센서 칼만필터 적용
    single_kf = SingleAngleKalmanFilter(dt=1.0, process_noise=1e-5, sensor_noise=1e-4)
    single_filtered = single_kf.filter_data(sensor1_data)
    
    print(f"단일센서 - 원본 RMS: {np.sqrt(np.mean((true_angles - sensor1_data)**2)):.6f}")
    print(f"단일센서 - 필터 RMS: {np.sqrt(np.mean((true_angles - single_filtered)**2)):.6f}")