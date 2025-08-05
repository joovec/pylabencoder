# -*- coding: utf-8 -*-
"""분석 함수들 - 엔코더 데이터 분석용."""
import numpy as np
from sklearn.linear_model import LinearRegression


def linear_error_analysis(angle_data):
    """각도 데이터의 선형 에러를 분석하는 함수.
    
    Args:
        angle_data: numpy 배열 또는 pandas Series (각도 데이터)
    
    Returns:
        numpy array: 언랩된 데이터와 선형 회귀의 차이 (선형 에러)
    """
    # pandas Series인 경우 numpy 배열로 변환
    if hasattr(angle_data, 'values'):
        angles = angle_data.values
    else:
        angles = np.array(angle_data)
    
    # 1단계: numpy.unwrap()으로 언랩
    unwrapped = np.unwrap(angles)
    
    # 2단계: 선형 회귀 (1차)
    x = np.arange(len(unwrapped)).reshape(-1, 1)  # 인덱스를 x축으로
    y = unwrapped.reshape(-1, 1)
    
    # sklearn 선형 회귀 모델
    model = LinearRegression()
    model.fit(x, y)
    
    # 선형 회귀 예측값
    linear_fit = model.predict(x).flatten()
    
    # 3단계: 차이 계산 (선형 에러)
    linear_error = unwrapped - linear_fit
    
    return linear_error


if __name__ == "__main__":
    # 테스트
    test_angles = np.array([0, 1, 2, 6.2, 0.5, 1.5])  # 라디안
    error = linear_error_analysis(test_angles)
    print("테스트 결과:", error)