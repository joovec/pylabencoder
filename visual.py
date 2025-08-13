# -*- coding: utf-8 -*-
"""
칼만 필터 성능 비교 시각화 모듈

CV, CA, CJ 칼만 필터와 측정값의 각도 오차 비교 그래프
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
from src.pytemplate.utils.plot_helper import setup_korean_font
from src.pytemplate.core import ang


class KalmanVisualization:
    """칼만 필터 시각화 클래스"""
    
    def __init__(self):
        """시각화 초기화"""
        # 한글 폰트 설정을 가장 먼저 수행
        setup_korean_font()
        
        # 색상 팔레트
        self.colors = {
            'true': '#2E8B57',      # 진실값 - 진한 녹색
            'measured': '#708090',   # 측정값 - 회색
            'cv': '#4169E1',        # CV - 파란색
            'ca': '#FF6347',        # CA - 토마토색
            'cj': '#9932CC'         # CJ - 보라색
        }
        
        # 라인 스타일
        self.linestyles = {
            'true': '-',
            'measured': '-',
            'cv': '-',
            'ca': '--',
            'cj': '-.'
        }
        
        # 마커 스타일
        self.markers = {
            'true': None,
            'measured': '.',
            'cv': None,
            'ca': None,
            'cj': None
        }
    
    def plot_angle_comparison(self, 
                            time: np.ndarray,
                            true_angles: np.ndarray,
                            measured_angles: np.ndarray,
                            cv_angles: np.ndarray,
                            ca_angles: np.ndarray,
                            cj_angles: np.ndarray,
                            motion_info: Dict,
                            filter_configs: Dict,
                            save_path: Optional[str] = None) -> None:
        """각도 비교 그래프
        
        Args:
            time: 시간 배열
            true_angles: 실제 각도
            measured_angles: 측정 각도
            cv_angles, ca_angles, cj_angles: 칼만 필터 추정 각도
            motion_info: 모션 프로파일 정보
            save_path: 저장 경로 (None이면 화면 출력)
        """
        # 한글 폰트 재설정 (확실하게)
        setup_korean_font()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('칼만 필터 성능 비교 - 사다리꼴 각속도 프로파일', fontsize=16, fontweight='bold')
        
        # 1. 각도 비교
        ax1 = axes[0, 0]
        ax1.plot(time, np.degrees(true_angles), 
                color=self.colors['true'], linestyle=self.linestyles['true'], 
                linewidth=2, label='실제값', alpha=0.9)
        ax1.plot(time, np.degrees(measured_angles), 
                color=self.colors['measured'], linestyle=self.linestyles['measured'], 
                marker=self.markers['measured'], markersize=0.5, alpha=0.6, label='측정값')
        ax1.plot(time, np.degrees(cv_angles), 
                color=self.colors['cv'], linestyle=self.linestyles['cv'], 
                linewidth=1.5, label='CV 칼만', alpha=0.8)
        ax1.plot(time, np.degrees(ca_angles), 
                color=self.colors['ca'], linestyle=self.linestyles['ca'], 
                linewidth=1.5, label='CA 칼만', alpha=0.8)
        ax1.plot(time, np.degrees(cj_angles), 
                color=self.colors['cj'], linestyle=self.linestyles['cj'], 
                linewidth=1.5, label='CJ 칼만', alpha=0.8)
        
        ax1.set_ylabel('각도 (°)')
        ax1.set_title('각도 추정 비교')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 각도 오차 비교
        ax2 = axes[0, 1]
        
        # 오차 계산 (순환 차이)
        cv_error = np.array([ang.diffpi(est, true) for est, true in zip(cv_angles, true_angles)])
        ca_error = np.array([ang.diffpi(est, true) for est, true in zip(ca_angles, true_angles)])
        cj_error = np.array([ang.diffpi(est, true) for est, true in zip(cj_angles, true_angles)])
        measured_error = np.array([ang.diffpi(meas, true) for meas, true in zip(measured_angles, true_angles)])
        
        # RMSE 계산
        measured_rmse = np.sqrt(np.mean(measured_error**2))
        cv_rmse = np.sqrt(np.mean(cv_error**2))
        ca_rmse = np.sqrt(np.mean(ca_error**2))
        cj_rmse = np.sqrt(np.mean(cj_error**2))
        
        # 최대 오차 계산
        cv_max_error = np.max(np.abs(cv_error))
        ca_max_error = np.max(np.abs(ca_error))
        cj_max_error = np.max(np.abs(cj_error))
        
        ax2.plot(time, np.degrees(measured_error), 
                color=self.colors['measured'], linestyle=self.linestyles['measured'], 
                marker=self.markers['measured'], markersize=0.3, linewidth=1, alpha=0.6, 
                label=f'측정 오차 (RMS: {measured_rmse*180/np.pi:.2f}°)')
        ax2.plot(time, np.degrees(cv_error), 
                color=self.colors['cv'], linestyle=self.linestyles['cv'], 
                linewidth=1.5, label=f'CV 오차 (RMS: {cv_rmse*180/np.pi:.2f}°)')
        ax2.plot(time, np.degrees(ca_error), 
                color=self.colors['ca'], linestyle=self.linestyles['ca'], 
                linewidth=1.5, label=f'CA 오차 (RMS: {ca_rmse*180/np.pi:.2f}°)')
        ax2.plot(time, np.degrees(cj_error), 
                color=self.colors['cj'], linestyle=self.linestyles['cj'], 
                linewidth=1.5, label=f'CJ 오차 (RMS: {cj_rmse*180/np.pi:.2f}°)')
        
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
        ax2.set_ylabel('각도 오차 (°)')
        ax2.set_title('각도 추정 오차 비교')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 각속도 비교 (차분으로 계산)
        ax3 = axes[1, 0]
        
        # 각속도 계산 (순환 차분)
        dt = time[1] - time[0]
        true_vel_diff = ang.neighor_angle_diff(true_angles) / dt
        measured_vel_diff = ang.neighor_angle_diff(measured_angles) / dt
        cv_vel_diff = ang.neighor_angle_diff(cv_angles) / dt
        ca_vel_diff = ang.neighor_angle_diff(ca_angles) / dt
        cj_vel_diff = ang.neighor_angle_diff(cj_angles) / dt
        
        # 첫 번째 값은 0으로 설정
        true_vel_diff = np.insert(true_vel_diff, 0, 0.0)
        measured_vel_diff = np.insert(measured_vel_diff, 0, 0.0)
        cv_vel_diff = np.insert(cv_vel_diff, 0, 0.0)
        ca_vel_diff = np.insert(ca_vel_diff, 0, 0.0)
        cj_vel_diff = np.insert(cj_vel_diff, 0, 0.0)
        
        ax3.plot(time, true_vel_diff * 30/np.pi, 
                color=self.colors['true'], linestyle=self.linestyles['true'], 
                linewidth=2, label='실제값', alpha=0.9)
        ax3.plot(time, measured_vel_diff * 30/np.pi, 
                color=self.colors['measured'], linestyle=self.linestyles['measured'], 
                marker=self.markers['measured'], markersize=0.3, alpha=0.5, label='측정값 차분')
        ax3.plot(time, cv_vel_diff * 30/np.pi, 
                color=self.colors['cv'], linestyle=self.linestyles['cv'], 
                linewidth=1.5, label='CV 칼만', alpha=0.8)
        ax3.plot(time, ca_vel_diff * 30/np.pi, 
                color=self.colors['ca'], linestyle=self.linestyles['ca'], 
                linewidth=1.5, label='CA 칼만', alpha=0.8)
        ax3.plot(time, cj_vel_diff * 30/np.pi, 
                color=self.colors['cj'], linestyle=self.linestyles['cj'], 
                linewidth=1.5, label='CJ 칼만', alpha=0.8)
        
        ax3.set_ylabel('각속도 (RPM)')
        ax3.set_xlabel('시간 (s)')
        ax3.set_title('각속도 추정 비교')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 성능 지표 표시
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # 5. 필터 설정 표시
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
• 가속 시간: {motion_info.get('acceleration_time', 'N/A')}
• 전체 시간: {motion_info.get('total_duration', 'N/A')}
• 측정 노이즈: {motion_info.get('noise_std', 'N/A')}
• 양자화: {motion_info.get('quantization_bits', 'N/A')}
• 총 회전각: {motion_info.get('total_angle', 'N/A')}
• 최대 가속도: {motion_info.get('max_acceleration', 'N/A')}
        """
        
        # 필터 설정 정보
        filter_text = f"""
칼만 필터 설정:
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
        measured_max_error = np.max(np.abs(measured_error))
        cv_improvement = (measured_rmse - cv_rmse) / measured_rmse * 100
        ca_improvement = (measured_rmse - ca_rmse) / measured_rmse * 100
        cj_improvement = (measured_rmse - cj_rmse) / measured_rmse * 100
        
        table_data = [
            ['지표', '측정값', 'CV', 'CA', 'CJ'],
            ['RMSE (°)', f'{measured_rmse*180/np.pi:.2f}', f'{cv_rmse*180/np.pi:.2f}', f'{ca_rmse*180/np.pi:.2f}', f'{cj_rmse*180/np.pi:.2f}'],
            ['MAX (°)', f'{measured_max_error*180/np.pi:.2f}', f'{cv_max_error*180/np.pi:.2f}', f'{ca_max_error*180/np.pi:.2f}', f'{cj_max_error*180/np.pi:.2f}'],
            ['개선율 (%)', '기준', f'{cv_improvement:.1f}', f'{ca_improvement:.1f}', f'{cj_improvement:.1f}']
        ]
        
        table = ax6.table(cellText=table_data,
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.15, 0.15, 0.15, 0.15, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # 헤더 색상
        for i in range(5):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 최적 성능 강조 (측정값 제외하고 가장 작은 RMSE)
        filter_rmse_values = [cv_rmse, ca_rmse, cj_rmse]
        best_filter_idx = np.argmin(filter_rmse_values) + 2  # +2 because column 0 is labels, column 1 is measured
        for i in range(1, 4):
            table[(i, best_filter_idx)].set_facecolor('#90EE90')
        
        # 측정값 열 강조
        for i in range(1, 4):
            table[(i, 1)].set_facecolor('#FFE4B5')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"그래프가 저장되었습니다: {save_path}")
        else:
            plt.show()
    
    def plot_error_statistics(self,
                            cv_errors: np.ndarray,
                            ca_errors: np.ndarray, 
                            cj_errors: np.ndarray) -> None:
        """오차 통계 히스토그램
        
        Args:
            cv_errors, ca_errors, cj_errors: 각 필터의 오차 배열
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('칼만 필터 오차 분포', fontsize=16, fontweight='bold')
        
        filters = [('CV', cv_errors, self.colors['cv']), 
                  ('CA', ca_errors, self.colors['ca']),
                  ('CJ', cj_errors, self.colors['cj'])]
        
        for i, (name, errors, color) in enumerate(filters):
            ax = axes[i]
            
            # 히스토그램
            ax.hist(np.degrees(errors), bins=30, alpha=0.7, color=color, 
                   edgecolor='black', linewidth=0.5)
            
            # 통계 정보
            rmse = np.sqrt(np.mean(errors**2))
            std = np.std(errors)
            
            ax.axvline(0, color='red', linestyle='--', alpha=0.8, linewidth=2)
            ax.set_xlabel('오차 (°)')
            ax.set_ylabel('빈도')
            ax.set_title(f'{name} 칼만 필터\nRMSE: {rmse*180/np.pi:.3f}°, STD: {std*180/np.pi:.3f}°')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def create_visualizer() -> KalmanVisualization:
    """시각화 객체 생성 헬퍼 함수"""
    return KalmanVisualization()