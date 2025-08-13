import numpy as np

def diffpi(angle1, angle2):
    """두 각도의 차이를 구하는 함수 (불연속 구간 고려)"""
    diff = angle1 - angle2
    return np.mod(diff + np.pi, 2*np.pi) - np.pi

def wrap_pi(angle):
    """입력 각도를 [-pi, pi) 범위로 래핑"""
    return np.mod(angle + np.pi, 2*np.pi) - np.pi

def wrap_2pi(angle):
    """입력 각도를 [0, 2pi) 범위로 래핑"""
    return np.mod(angle, 2*np.pi)

# 테스트 케이스
print("=== wrap_2pi 테스트 ===")
test_angles = [-2*np.pi, -np.pi, 0, np.pi, 2*np.pi, 3*np.pi]
for angle in test_angles:
    wrapped = wrap_2pi(angle)
    print(f"wrap_2pi({angle:.2f}) = {wrapped:.2f} ({np.degrees(wrapped):.1f}°)")

print("\n=== wrap_pi 테스트 ===")
for angle in test_angles:
    wrapped = wrap_pi(angle)
    print(f"wrap_pi({angle:.2f}) = {wrapped:.2f} ({np.degrees(wrapped):.1f}°)")

print("\n=== diffpi 테스트 (불연속 구간) ===")
# 0도 근처 불연속
angle1 = np.deg2rad(5)
angle2 = np.deg2rad(355)
diff = diffpi(angle1, angle2)
print(f"diffpi(5°, 355°) = {np.degrees(diff):.1f}° (예상: 10°)")

# 180도 근처 불연속
angle1 = np.deg2rad(175)
angle2 = np.deg2rad(-175)
diff = diffpi(angle1, angle2)
print(f"diffpi(175°, -175°) = {np.degrees(diff):.1f}° (예상: -10°)")

# 일반적인 경우
angle1 = np.deg2rad(90)
angle2 = np.deg2rad(30)
diff = diffpi(angle1, angle2)
print(f"diffpi(90°, 30°) = {np.degrees(diff):.1f}° (예상: 60°)")

print("\n=== 칼만필터에서의 활용 예시 ===")
# 측정값이 359도, 예측값이 1도인 경우
z_meas = np.deg2rad(359)
x_pred = np.deg2rad(1)
innovation = diffpi(z_meas, x_pred)
print(f"측정값 359°, 예측값 1° -> innovation = {np.degrees(innovation):.1f}° (예상: -2°)")

# 반대 경우
z_meas = np.deg2rad(1)
x_pred = np.deg2rad(359)
innovation = diffpi(z_meas, x_pred)
print(f"측정값 1°, 예측값 359° -> innovation = {np.degrees(innovation):.1f}° (예상: 2°)")