# -*- coding: utf-8 -*-
"""
칼만 필터 및 다양한 필터 성능 비교 시각화 모듈

CV, CA, CJ 칼만 필터와 IIR, EMA 필터의 성능 비교 그래프
모션별 개별 그래프 및 히스토그램 포함
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
from src.pytemplate.utils.plot_helper import setup_korean_font
from src.pytemplate.core import ang


class FilterVisualization:
    """필터 성능 시각화 클래스"""
    
    def __init__(self):
        """시각화 초기화"""
        # 한글 폰트 설정을 가장 먼저 수행
        setup_korean_font()
        
        # 색상 팔레트 (확장)
        self.colors = {
            'true': '#2E8B57',      # 진실값 - 진한 녹색
            'measured': '#708090',   # 측정값 - 회색
            'cv': '#4169E1',        # CV - 파란색
            'ca': '#FF6347',        # CA - 토마토색
            'cj': '#9932CC',        # CJ - 보라색
            'iir': '#FF8C00',       # IIR - 주황색
            'ema': '#DC143C',       # EMA - 빨간색
            'fir': '#228B22'        # FIR - 녹색
        }
        
        # 라인 스타일
        self.linestyles = {
            'true': '-',
            'measured': '-',
            'cv': '-',
            'ca': '--',
            'cj': '-.',
            'iir': ':',
            'ema': '-',
            'fir': '--'
        }
        
        # 마커 스타일
        self.markers = {
            'true': None,
            'measured': '.',
            'cv': None,
            'ca': None,
            'cj': None,
            'iir': None,
            'ema': None,
            'fir': None
        }
    
    def plot_all_filters_comparison(self, 
                                  time: np.ndarray,
                                  true_angles: np.ndarray,
                                  measured_angles: np.ndarray,
                                  filter_results: Dict[str, np.ndarray],
                                  motion_info: Dict,
                                  filter_configs: Dict,
                                  save_path: Optional[str] = None) -> None:
        """모든 필터 비교 그래프
        
        Args:
            time: 시간 배열
            true_angles: 실제 각도
            measured_angles: 측정 각도
            filter_results: 필터별 결과 {'cv': angles, 'ca': angles, ...}
            motion_info: 모션 프로파일 정보
            filter_configs: 필터 설정 정보
            save_path: 저장 경로 (None이면 화면 출력)
        """
        # 한글 폰트 재설정 (확실하게)
        setup_korean_font()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'필터 성능 비교 - {motion_info.get("motion_type", "Unknown")}', fontsize=16, fontweight='bold')
        
        # 1. 각도 비교
        ax1 = axes[0, 0]
        ax1.plot(time, np.degrees(true_angles), 
                color=self.colors['true'], linestyle=self.linestyles['true'], 
                linewidth=2, label='실제값', alpha=0.9)
        ax1.plot(time, np.degrees(measured_angles), 
                color=self.colors['measured'], linestyle=self.linestyles['measured'], 
                marker=self.markers['measured'], markersize=0.5, alpha=0.6, label='측정값')
        
        # 필터별 결과 플롯
        for filter_name, angles in filter_results.items():
            if filter_name in self.colors:
                ax1.plot(time, np.degrees(angles), 
                        color=self.colors[filter_name], linestyle=self.linestyles[filter_name], 
                        linewidth=1.5, label=f'{filter_name.upper()}', alpha=0.8)
        
        ax1.set_ylabel('각도 (°)')
        ax1.set_title('각도 추정 비교')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 각도 오차 비교
        ax2 = axes[0, 1]
        
        # 오차 계산 (순환 차이)
        measured_error = np.array([ang.diffpi(meas, true) for meas, true in zip(measured_angles, true_angles)])
        filter_errors = {}
        
        # RMSE 계산
        measured_rmse = np.sqrt(np.mean(measured_error**2))
        
        ax2.plot(time, np.degrees(measured_error), 
                color=self.colors['measured'], linestyle=self.linestyles['measured'], 
                marker=self.markers['measured'], markersize=0.3, linewidth=1, alpha=0.6, 
                label=f'측정 오차 (RMS: {measured_rmse*180/np.pi:.2f}°)')
        
        for filter_name, angles in filter_results.items():
            if filter_name in self.colors:
                error = np.array([ang.diffpi(est, true) for est, true in zip(angles, true_angles)])
                filter_errors[filter_name] = error
                rmse = np.sqrt(np.mean(error**2))
                
                ax2.plot(time, np.degrees(error), 
                        color=self.colors[filter_name], linestyle=self.linestyles[filter_name], 
                        linewidth=1.5, label=f'{filter_name.upper()} 오차 (RMS: {rmse*180/np.pi:.2f}°)')
        
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
        ax2.set_ylabel('각도 오차 (°)')
        ax2.set_title('각도 추정 오차 비교')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 각속도 비교 (차분으로 계산)
        ax3 = axes[1, 0]
        
        dt = time[1] - time[0]
        true_vel_diff = ang.neighor_angle_diff(true_angles) / dt
        measured_vel_diff = ang.neighor_angle_diff(measured_angles) / dt
        
        # 첫 번째 값은 0으로 설정
        true_vel_diff = np.insert(true_vel_diff, 0, 0.0)
        measured_vel_diff = np.insert(measured_vel_diff, 0, 0.0)
        
        ax3.plot(time, true_vel_diff * 30/np.pi, 
                color=self.colors['true'], linestyle=self.linestyles['true'], 
                linewidth=2, label='실제값', alpha=0.9)
        ax3.plot(time, measured_vel_diff * 30/np.pi, 
                color=self.colors['measured'], linestyle=self.linestyles['measured'], 
                marker=self.markers['measured'], markersize=0.3, alpha=0.5, label='측정값 차분')
        
        for filter_name, angles in filter_results.items():
            if filter_name in self.colors:
                filter_vel_diff = ang.neighor_angle_diff(angles) / dt
                filter_vel_diff = np.insert(filter_vel_diff, 0, 0.0)
                ax3.plot(time, filter_vel_diff * 30/np.pi, 
                        color=self.colors[filter_name], linestyle=self.linestyles[filter_name], 
                        linewidth=1.5, label=f'{filter_name.upper()}', alpha=0.8)
        
        ax3.set_ylabel('각속도 (RPM)')
        ax3.set_xlabel('시간 (s)')
        ax3.set_title('각속도 추정 비교')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 모션 프로파일 정보
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # 5. 필터 설정 정보
        ax5 = axes[0, 2]
        ax5.axis('off')
        
        # 6. 성능 지표 테이블
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # 모션 프로파일 정보
        motion_text = f"""
모션 프로파일 정보:
• {motion_info.get('motion_type', 'Unknown')}
• 최대 각속도: {motion_info.get('max_velocity', 'N/A')}
• 전체 시간: {motion_info.get('total_duration', 'N/A')}
• 측정 노이즈: {motion_info.get('noise_std', 'N/A')}
• 양자화: {motion_info.get('quantization_bits', 'N/A')}
• 총 회전각: {motion_info.get('total_angle', 'N/A')}
        """
        
        # 필터 설정 정보
        filter_text = f"""
필터 설정:
• 샘플링: {filter_configs.get('dt', 0.01):.3f}s ({1/filter_configs.get('dt', 0.01):.0f}Hz)
• 프로세스 노이즈 Q: {filter_configs.get('process_noise', 'N/A'):.1e}
• 측정 노이즈 R: {filter_configs.get('measurement_noise', 'N/A'):.1e}
• Q/R 비율: {filter_configs.get('q_r_ratio', 'N/A'):.1e}

사용된 노이즈 값:
• 시뮬레이션 노이즈: {motion_info.get('noise_std', 'N/A')}
• 필터 측정 노이즈: {np.sqrt(filter_configs.get('measurement_noise', 0)):.4f} rad
  ({np.sqrt(filter_configs.get('measurement_noise', 0))*180/np.pi:.2f}°)
        """
        
        ax4.text(0.05, 0.95, motion_text.strip(), transform=ax4.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        ax5.text(0.05, 0.95, filter_text.strip(), transform=ax5.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        # 성능 지표 테이블
        header = ['지표', '측정값'] + [name.upper() for name in filter_results.keys()]
        
        # RMSE 행
        rmse_row = ['RMSE (°)', f'{measured_rmse*180/np.pi:.2f}']
        for filter_name in filter_results.keys():
            if filter_name in filter_errors:
                rmse = np.sqrt(np.mean(filter_errors[filter_name]**2))
                rmse_row.append(f'{rmse*180/np.pi:.2f}')
        
        # MAX 오차 행
        max_row = ['MAX (°)', f'{np.max(np.abs(measured_error))*180/np.pi:.2f}']
        for filter_name in filter_results.keys():
            if filter_name in filter_errors:
                max_error = np.max(np.abs(filter_errors[filter_name]))
                max_row.append(f'{max_error*180/np.pi:.2f}')
        
        # 개선율 행
        improvement_row = ['개선율 (%)', '기준']
        for filter_name in filter_results.keys():
            if filter_name in filter_errors:
                filter_rmse = np.sqrt(np.mean(filter_errors[filter_name]**2))
                improvement = (measured_rmse - filter_rmse) / measured_rmse * 100
                improvement_row.append(f'{improvement:.1f}')
        
        table_data = [header, rmse_row, max_row, improvement_row]
        
        table = ax6.table(cellText=table_data,
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.12] + [0.1] * len(header[1:]))
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.5)
        
        # 헤더 색상
        for i in range(len(header)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 측정값 열 강조
        for i in range(1, 4):
            table[(i, 1)].set_facecolor('#FFE4B5')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"그래프가 저장되었습니다: {save_path}")
        else:
            plt.show()
    
    def plot_error_histogram_with_measured(self,
                                          measured_errors: np.ndarray,
                                          filter_errors: Dict[str, np.ndarray]) -> None:
        """오차 히스토그램 (측정값 포함)
        
        Args:
            measured_errors: 측정값 오차 배열
            filter_errors: 필터별 오차 딕셔너리
        """
        setup_korean_font()
        
        n_filters = len(filter_errors) + 1  # +1 for measured
        cols = min(4, n_filters)
        rows = (n_filters + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        fig.suptitle('오차 분포 히스토그램', fontsize=16, fontweight='bold')
        
        # 측정값 히스토그램
        ax = axes[0]
        measured_rmse = np.sqrt(np.mean(measured_errors**2))
        measured_std = np.std(measured_errors)
        
        ax.hist(np.degrees(measured_errors), bins=30, alpha=0.7, 
               color=self.colors['measured'], edgecolor='black', linewidth=0.5)
        ax.axvline(0, color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax.set_xlabel('오차 (°)')
        ax.set_ylabel('빈도')
        ax.set_title(f'측정값 오차\\nRMSE: {measured_rmse*180/np.pi:.3f}°, STD: {measured_std*180/np.pi:.3f}°')
        ax.grid(True, alpha=0.3)
        
        # 필터별 히스토그램
        for i, (filter_name, errors) in enumerate(filter_errors.items(), 1):
            if i >= len(axes):
                break
                
            ax = axes[i]
            rmse = np.sqrt(np.mean(errors**2))
            std = np.std(errors)
            
            ax.hist(np.degrees(errors), bins=30, alpha=0.7, 
                   color=self.colors.get(filter_name, 'gray'), 
                   edgecolor='black', linewidth=0.5)
            ax.axvline(0, color='red', linestyle='--', alpha=0.8, linewidth=2)
            ax.set_xlabel('오차 (°)')
            ax.set_ylabel('빈도')
            ax.set_title(f'{filter_name.upper()} 필터\\nRMSE: {rmse*180/np.pi:.3f}°, STD: {std*180/np.pi:.3f}°')
            ax.grid(True, alpha=0.3)
        
        # 빈 subplot 숨기기
        for i in range(n_filters, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_motion_specific_analysis(self,
                                    motion_data: Dict,
                                    filter_results: Dict[str, np.ndarray],
                                    save_dir: Optional[str] = None) -> None:
        """모션별 개별 분석 그래프 생성
        
        Args:
            motion_data: 모션 데이터 딕셔너리
            filter_results: 필터 결과 딕셔너리
            save_dir: 저장 디렉토리
        """
        setup_korean_font()
        
        time = motion_data['time']
        true_angles = motion_data['true_angles']
        measured_angles = motion_data['measured_angles']
        motion_info = motion_data['info']
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        motion_name = motion_info.get('motion_type', 'Unknown')
        fig.suptitle(f'{motion_name} - 필터 성능 비교', fontsize=16, fontweight='bold')
        
        # 각도 비교 플롯
        ax.plot(time, np.degrees(true_angles), 
               color=self.colors['true'], linestyle='-', 
               linewidth=3, label='실제값', alpha=0.9)
        ax.plot(time, np.degrees(measured_angles), 
               color=self.colors['measured'], linestyle='-', 
               marker='.', markersize=1, alpha=0.6, label='측정값')
        
        # 필터별 결과
        for filter_name, angles in filter_results.items():
            if filter_name in self.colors:
                ax.plot(time, np.degrees(angles), 
                       color=self.colors[filter_name], 
                       linestyle=self.linestyles[filter_name], 
                       linewidth=2, label=f'{filter_name.upper()}', alpha=0.8)
        
        ax.set_xlabel('시간 (s)')
        ax.set_ylabel('각도 (°)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 정보 텍스트 박스
        info_text = f"""
모션: {motion_name}
최대 속도: {motion_info.get('max_velocity', 'N/A')}
시간: {motion_info.get('total_duration', 'N/A')}
        """
        
        ax.text(0.02, 0.98, info_text.strip(), transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        if save_dir:
            motion_type = motion_info.get('motion_type', 'unknown').replace(' ', '_').lower()
            save_path = f"{save_dir}/{motion_type}_analysis.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"모션별 그래프 저장: {save_path}")
        else:
            plt.show()


def create_visualizer() -> FilterVisualization:
    """시각화 객체 생성 헬퍼 함수"""
    return FilterVisualization()