# -*- coding: utf-8 -*-
"""
autil.py - 추가 유틸리티 함수들
"""
import numpy as np




def diffpi(angle1, angle2):
    """
    두 각도의 차이를 구하는 함수 (불연속 구간 고려)
    
    Parameters:
    angle1, angle2 : float or np.ndarray
        입력각도 라디안
        
    Returns:
    float or np.ndarray
        각도 차이 ([-pi, pi) 범위로 정규화됨)
    """
    diff = angle1 - angle2
    return np.mod(diff + np.pi, 2*np.pi) - np.pi

def wrap_pi(angle):
    """
    입력 각도를 [-pi, pi) 범위로 래핑하는 함수
    
    Parameters:
    angle : float or np.ndarray
        입력 각도 라디안
        
    Returns:
    float or np.ndarray
        [-pi, pi) 범위로 정규화된 각도
    """
    return np.mod(angle + np.pi, 2*np.pi) - np.pi

def wrap_2pi(angle):
    """
    입력 각도를 [0, 2pi) 범위로 래핑하는 함수
    
    Parameters:
    angle : float or np.ndarray
        입력 각도 라디안
        
    Returns:
    float or np.ndarray
        [0, 2pi) 범위로 정규화된 각도
    """
    return np.mod(angle, 2*np.pi)


def neighor_angle_diff(angle_array):

    angle_array=np.array(angle_array)
 
    return diffpi(angle_array[1:], angle_array[:-1])

def angle_diff_cumsum(angle_array):
    return np.insert(np.cumsum(diffpi(angle_array[1:], angle_array[:-1])),0,0)

def angle_diff_sum(angle_array):
    return np.sum(neighor_angle_diff(angle_array))


def main():
    angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    print("Angle differences:", neighor_angle_diff(angles))
    print("Cumulative angle differences:", angle_diff_cumsum(angles))
    print("Total angle difference:", angle_diff_sum(angles))


if __name__ == "__main__":
    main()