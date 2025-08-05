# -*- coding: utf-8 -*-
"""CSV 파일 읽기 유틸리티 모듈."""
import os
import pandas as pd
from pathlib import Path


def get_project_root():
    """프로젝트 루트 디렉토리를 찾는 함수."""
    # 현재 파일(csv_reader.py)의 위치에서 상위로 올라가며 프로젝트 루트 찾기
    # src/pytemplate/utils/csv_reader.py -> 3단계 상위가 프로젝트 루트
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent.parent
    return project_root


def load_csv_data(filename, csv_dir='csvdata'):
    """CSV 파일을 프로젝트의 csvdata 폴더에서 읽어오는 함수.
    
    Args:
        filename (str): 읽을 CSV 파일명
        csv_dir (str): CSV 파일이 있는 디렉토리명 (기본값: 'csvdata')
    
    Returns:
        pd.DataFrame: 읽어온 데이터프레임, 실패시 None
    """
    project_root = get_project_root()
    csv_path = project_root / csv_dir / filename
    
    if not csv_path.exists():
        print(f"파일을 찾을 수 없습니다: {csv_path}")
        return None
    
    try:
        # CSV 파일 읽기
        df = pd.read_csv(csv_path)
        print(f"CSV 파일 로드 성공: {filename}")
        print(f"데이터 shape: {df.shape}")
        print(f"컬럼: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"CSV 파일 읽기 오류: {e}")
        return None


def save_csv_data(df, filename, csv_dir='csvdata'):
    """데이터프레임을 CSV 파일로 저장하는 함수.
    
    Args:
        df (pd.DataFrame): 저장할 데이터프레임
        filename (str): 저장할 파일명
        csv_dir (str): 저장할 디렉토리명 (기본값: 'csvdata')
    
    Returns:
        bool: 저장 성공 여부
    """
    project_root = get_project_root()
    csv_path = project_root / csv_dir / filename
    
    # 디렉토리가 없으면 생성
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        df.to_csv(csv_path, index=False)
        print(f"CSV 파일 저장 성공: {filename}")
        return True
    except Exception as e:
        print(f"CSV 파일 저장 오류: {e}")
        return False


def list_csv_files(csv_dir='csvdata'):
    """csvdata 폴더의 모든 CSV 파일 목록을 반환하는 함수.
    
    Args:
        csv_dir (str): CSV 파일이 있는 디렉토리명 (기본값: 'csvdata')
    
    Returns:
        list: CSV 파일명 리스트
    """
    project_root = get_project_root()
    csv_path = project_root / csv_dir
    
    if not csv_path.exists():
        print(f"디렉토리를 찾을 수 없습니다: {csv_path}")
        return []
    
    csv_files = [f.name for f in csv_path.glob('*.csv')]
    return csv_files


if __name__ == "__main__":
    # 모듈 테스트
    print("프로젝트 루트:", get_project_root())
    print("CSV 파일 목록:", list_csv_files())
    
    # 테스트용 CSV 읽기
    df = load_csv_data('wheel_angle_results.csv')
    if df is not None:
        print("\n처음 5행:")
        print(df.head())