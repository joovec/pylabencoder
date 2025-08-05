# -*- coding: utf-8 -*-
"""Matplotlib 플롯 헬퍼 유틸리티."""
import matplotlib.pyplot as plt
import platform


def setup_korean_font():
    """운영체제별로 한글 폰트를 설정하는 함수."""
    system = platform.system()
    if system == 'Windows':
        font_names = ['Malgun Gothic', 'Microsoft YaHei', 'SimHei']
    elif system == 'Darwin':  # macOS
        font_names = ['AppleGothic', 'Helvetica']
    else:  # Linux
        font_names = ['DejaVu Sans', 'Liberation Sans']
    
    for font_name in font_names:
        try:
            plt.rcParams['font.family'] = font_name
            break
        except:
            continue
    
    plt.rcParams['axes.unicode_minus'] = False


def set_plot_style(style='default', figsize=(10, 6), dpi=100):
    """플롯 스타일을 설정하는 함수.
    
    Args:
        style: matplotlib 스타일 ('default', 'seaborn', 'ggplot' 등)
        figsize: 그림 크기 (width, height)
        dpi: 해상도
    """
    if style != 'default':
        plt.style.use(style)
    
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3


def save_figure(filename, dpi=300, bbox_inches='tight'):
    """그림을 파일로 저장하는 함수.
    
    Args:
        filename: 저장할 파일명
        dpi: 저장 해상도
        bbox_inches: 여백 설정
    """
    plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
    print(f"그림 저장 완료: {filename}")


def plot_timeseries(df, column_names, figsize=(15, 8), title=None):
    """데이터프레임의 컬럼(들)을 시계열로 플롯하는 함수.
    
    Args:
        df: 데이터프레임
        column_names: 플롯할 컬럼명 (문자열 또는 리스트)
        figsize: 그림 크기 (width, height)
        title: 그래프 제목 (None이면 자동 생성)
    """
    # 문자열이면 리스트로 변환
    if isinstance(column_names, str):
        column_names = [column_names]
    
    # 하나의 그래프에 모든 컬럼 겹쳐서 그리기
    plt.figure(figsize=figsize)
    
    colors = ['b', 'r', 'g', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, col in enumerate(column_names):
        color = colors[i % len(colors)]  # 색상 순환
        plt.plot(df[col], color=color, linewidth=0.5, alpha=0.7, label=col)
    
    # 제목 설정
    if title is None:
        if len(column_names) == 1:
            title = f'{column_names[0]} 시계열 데이터'
        else:
            title = '시계열 데이터 비교'
    
    plt.title(title, fontsize=16)
    plt.xlabel('인덱스', fontsize=14)
    plt.ylabel('값', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.show(block=False)  # 논블로킹 모드로 창 띄우기


def plot_arrays(arrays, labels=None, figsize=(15, 8), title=None):
    """numpy 1차원 배열들을 시계열로 플롯하는 함수.
    
    Args:
        arrays: numpy 1차원 배열들의 리스트
        labels: 각 배열의 라벨 리스트 (None이면 자동 생성)
        figsize: 그림 크기 (width, height)
        title: 그래프 제목 (None이면 자동 생성)
    """
    plt.figure(figsize=figsize)
    
    colors = ['b', 'r', 'g', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, array in enumerate(arrays):
        color = colors[i % len(colors)]
        label = labels[i] if labels else f'Array {i+1}'
        plt.plot(array, color=color, linewidth=0.5, alpha=0.7, label=label)
    
    # 제목 설정
    if title is None:
        if len(arrays) == 1:
            title = '배열 시계열 데이터'
        else:
            title = '배열 시계열 데이터 비교'
    
    plt.title(title, fontsize=16)
    plt.xlabel('인덱스', fontsize=14)
    plt.ylabel('값', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.show(block=False)


def plot_histograms(arrays, labels=None, bins=50, alpha=0.7, figsize=(10, 6), title=None):
    """여러 1차원 배열들의 히스토그램을 오버랩해서 그리는 함수.
    
    Args:
        arrays: 1차원 배열들의 리스트
        labels: 각 배열의 라벨 리스트 (None이면 자동 생성)
        bins: 히스토그램 구간 수 (기본값: 50)
        alpha: 투명도 (기본값: 0.7)
        figsize: 그림 크기 (width, height)
        title: 그래프 제목 (None이면 자동 생성)
    """
    plt.figure(figsize=figsize)
    
    colors = ['b', 'r', 'g', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, array in enumerate(arrays):
        color = colors[i % len(colors)]
        label = labels[i] if labels else f'Array {i+1}'
        plt.hist(array, bins=bins, alpha=alpha, color=color, 
                edgecolor='black', linewidth=0.5, label=label)
    
    # 제목 설정
    if title is None:
        if len(arrays) == 1:
            title = '히스토그램'
        else:
            title = '히스토그램 비교'
    
    plt.title(title, fontsize=16)
    plt.xlabel('값', fontsize=14)
    plt.ylabel('빈도', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.show(block=False)


if __name__ == "__main__":
    # 테스트
    setup_korean_font()
    print("한글 폰트 설정 완료")
    print(f"현재 폰트: {plt.rcParams['font.family']}")