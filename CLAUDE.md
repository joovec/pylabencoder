# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

이 프로젝트는 `uv`를 패키지 매니저로 사용하는 Python 템플릿 프로젝트입니다. 표준 Python 패키지 구조를 따르며, 소스 코드는 `src/pytemplate/`에, 테스트는 `tests/`에 위치합니다.

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

프로젝트는 src 레이아웃 패턴을 따릅니다:

- **src/pytemplate/**: 메인 패키지 코드
  - **core/**: 핵심 기능 모듈들
    - `ang.py`: 각도 조작 유틸리티 (wrap_pi, wrap_2pi, 각도 차이 계산)
- **tests/**: pytest를 사용한 테스트 스위트
  - **unit/**: 개별 모듈 단위 테스트
  - **integration/**: 통합 테스트 (현재 비어있음)
- **main.py**: 한글 폰트 설정이 포함된 matplotlib 진입점 스크립트

## 주요 기술 사항

1. **Python 버전**: Python >=3.13 필요
2. **의존성**: 과학 계산을 위한 numpy, matplotlib, pandas, scipy 사용
3. **테스트 프레임워크**: pytest (coverage 지원)
4. **코드 스타일**: black 포맷터, ruff 린터 설정됨
5. **인코딩**: Windows에서 한글 지원을 위해 UTF-8 인코딩 헤더 추가

## Import 패턴

main.py나 노트북에서:
```python
# core 모듈에서 import
from src.pytemplate.core.ang import wrap_pi, diffpi
```

테스트 파일 내에서:
```python
# src 접두사 없이 패키지명 사용
from pytemplate.core.ang import wrap_pi
```