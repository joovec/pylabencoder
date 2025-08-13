# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

이 프로젝트는 칼만 필터(Kalman Filter)를 활용한 엔코더 각도 추정 및 분석을 위한 Python 프로젝트입니다. `uv`를 패키지 매니저로 사용하며, 표준 Python 패키지 구조를 따릅니다. 주요 기능은 다양한 칼만 필터 구현체와 각도 데이터 처리 유틸리티를 포함합니다.

## 자주 사용하는 명령어

### 개발 환경 설정
```bash
# 개발 도구를 포함한 모든 의존성 설치
uv sync --group dev
```

### 테스트 실행
```bash
# 모든 테스트 실행
uv run pytest

# 자세한 출력과 함께 테스트 실행
uv run pytest -v

# 특정 테스트 파일 실행
uv run pytest tests/unit/test_base.py -v

# 특정 테스트 함수만 실행
uv run pytest tests/unit/test_base.py::test_add -v

# 패턴으로 테스트 검색 및 실행
uv run pytest -k "test_add" -v
```

### 코드 품질 관리
```bash
# black으로 코드 포맷팅
uv run black src/ tests/

# ruff로 코드 스타일 검사
uv run ruff check .

# ruff 이슈 자동 수정
uv run ruff check --fix .
```

### Python 파일 실행
```bash
# main.py 실행
uv run python main.py

# 모듈 파일 직접 실행
uv run python src/pytemplate/core/ang.py

# Windows 한글 인코딩 문제 해결
powershell -Command '$env:PYTHONIOENCODING="utf-8"; uv run python src/pytemplate/core/ang.py'
```

## 아키텍처

프로젝트는 src 레이아웃 패턴을 따르며, 칼만 필터 기반 각도 추정에 특화되어 있습니다:

### 핵심 패키지 구조
- **src/pytemplate/**: 메인 패키지 코드
  - **core/**: 핵심 기능 모듈들
    - `ang.py`: 각도 조작 유틸리티 (wrap_pi, wrap_2pi, diffpi, 각도 차이 계산)
    - `kalman.py`: 칼만 필터 구현체
      - `AngleKalmanFilter`: 2개 엔코더 센서용 각도 칼만 필터
      - `SingleAngleKalmanFilter`: 단일 엔코더용 칼만 필터
    - `anal.py`: 각도 데이터 분석 함수 (linear_error_analysis)
  - **utils/**: 유틸리티 모듈
    - `csv_reader.py`: CSV 데이터 로드/저장 (프로젝트 루트 기준 경로 처리)
    - `plot_helper.py`: 한글 폰트 설정 및 플롯 유틸리티

### 실험 스크립트
프로젝트 루트에 다양한 칼만 필터 실험 스크립트들이 위치:
- `simple_kalman.py`: filterpy 라이브러리를 사용한 기본 구현
- `circular_kalman_*.py`: 원형 운동 칼만 필터 변형들
- `cv_*.py`: Constant Velocity 모델 실험들
- `adaptive_cascaded_kalman.py`: 적응형 캐스케이드 칼만 필터

### 데이터
- **csvdata/**: 실제 엔코더 각도 데이터 (wheel_angle_results.csv)

## 주요 기술 사항

1. **Python 버전**: Python >=3.13 필요
2. **핵심 의존성**: 
   - 과학 계산: numpy, matplotlib, pandas, scipy, scikit-learn
   - 칼만 필터: filterpy (표준 칼만 필터 구현)
   - 플로팅: ipympl (인터랙티브 matplotlib)
3. **테스트 프레임워크**: pytest (coverage 지원)
4. **코드 스타일**: black 포맷터, ruff 린터 설정됨
5. **인코딩**: Windows에서 한글 지원을 위해 UTF-8 인코딩 헤더 추가

## Import 패턴

main.py나 실험 스크립트에서:
```python
# core 모듈에서 import
from src.pytemplate.core.ang import wrap_pi, diffpi
from src.pytemplate.core.kalman import AngleKalmanFilter
from src.pytemplate.utils.csv_reader import load_csv_data
from src.pytemplate.utils.plot_helper import setup_korean_font

# filterpy 사용 시
from filterpy.kalman import KalmanFilter
```

테스트 파일 내에서:
```python
# src 접두사 없이 패키지명 사용
from pytemplate.core.ang import wrap_pi
```

## 칼만 필터 사용 패턴

프로젝트는 두 가지 칼만 필터 접근법을 제공:

1. **커스텀 구현** (`src.pytemplate.core.kalman`): 엔코더 각도에 특화된 구현
2. **filterpy 라이브러리**: 범용 칼만 필터 실험용

### 🔴 중요: 순환 혁신(Circular Innovation) 방식 사용

**각도 칼만 필터 구현 시 반드시 순환 혁신 방식을 사용할 것:**

```python
# ✅ 올바른 방식: 순환 혁신
predicted_angle = ang.wrap_pi(kf.x[0, 0])
innovation = ang.diffpi(measured_angle, predicted_angle)  # 모듈러 공간에서 차이
kf.x = kf.x + K * innovation

# ❌ 사용하지 말 것: 언래핑
unwrapped_angle += angle_diff  # 누적 오차와 오버플로 위험
```

**순환 혁신의 장점:**
- 오버플로 방지: 값이 무한정 증가하지 않음
- 누적 오차 없음: 각 측정마다 독립적 계산
- 메모리 효율적: bounded 범위 유지

각도 데이터 처리 시 주의사항:
- 각도는 -π ~ π 또는 0 ~ 2π 범위로 정규화 필요
- `ang.wrap_pi()`, `ang.wrap_2pi()` 함수 활용
- 각도 차이 계산 시 `ang.diffpi()` 사용 (wrap-around 처리)
- 칼만 필터 혁신 단계에서 반드시 모듈러 차이 사용