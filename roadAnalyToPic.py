# 主要依赖库
import gpxpy
import gpxpy.gpx
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
import math
import warnings
import os
warnings.filterwarnings('ignore')

# 设置中文字体支持
def set_chinese_font():
    """设置matplotlib中文字体支持"""
    try:
        # 尝试使用系统中已有的中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        
        # 测试字体是否可用
        test_fig, test_ax = plt.subplots(figsize=(1, 1))
        test_ax.text(0.5, 0.5, '测试中文', ha='center', va='center')
        plt.close(test_fig)
        
        print("中文字体设置成功")
        return True
    except:
        print("警告: 中文字体设置失败，将使用英文显示")
        return False

# 设置中文字体
set_chinese_font()

class GPXRunningAnalyzer:
    def __init__(self, gpx_file_path):
        self.gpx_file_path = gpx_file_path
        self.points = []
        self.waypoints = []  # 存储路点信息
        self.df = None
        self.results = {}
        
        # 提取基础文件名（不含路径和扩展名）
        self.base_filename = os.path.splitext(os.path.basename(gpx_file_path))[0]
        
    def parse_gpx(self):
        """解析GPX文件，提取轨迹点和路点 - 使用二进制模式读取"""
        try:
            # 使用二进制模式读取文件，完全避免编码问题
            with open(self.gpx_file_path, 'rb') as gpx_file:
                gpx_content = gpx_file.read()
                
            # 直接使用二进制内容解析GPX
            gpx = gpxpy.parse(gpx_content)
                    
        except Exception as e:
            raise Exception(f"无法解析GPX文件: {str(e)}")
            
        # 提取轨迹点
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    # 确保所有必要字段都存在
                    point_data = {
                        'latitude': point.latitude,
                        'longitude': point.longitude,
                        'elevation': point.elevation if point.elevation is not None else 0,
                        'time': point.time
                    }
                    self.points.append(point_data)
        
        # 提取路点（检查点）
        for waypoint in gpx.waypoints:
            waypoint_data = {
                'latitude': waypoint.latitude,
                'longitude': waypoint.longitude,
                'elevation': waypoint.elevation if waypoint.elevation is not None else 0,
                'name': waypoint.name if waypoint.name else f"CP{len(self.waypoints)+1}",
                'description': waypoint.description if waypoint.description else ""
            }
            self.waypoints.append(waypoint_data)
            
        self.df = pd.DataFrame(self.points)
        
        # 检查是否有足够的数据点
        if len(self.df) < 2:
            raise ValueError("GPX文件包含的点太少，无法分析")
            
        return self.df
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """使用Haversine公式计算两点间距离"""
        R = 6371000  # 地球半径(米)
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        
        a = math.sin(delta_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(delta_lambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def find_nearest_track_point(self, wp_lat, wp_lon):
        """找到距离路点最近的轨迹点"""
        if self.df is None or len(self.df) == 0:
            return None
            
        min_distance = float('inf')
        nearest_index = 0
        
        for i, row in self.df.iterrows():
            dist = self.calculate_distance(wp_lat, wp_lon, row['latitude'], row['longitude'])
            if dist < min_distance:
                min_distance = dist
                nearest_index = i
                
        return nearest_index, min_distance
    
    def smooth_elevation(self, window_size=5):
        """使用移动平均平滑海拔数据"""
        if self.df is None:
            self.parse_gpx()
            
        # 检查是否有足够的数据点进行平滑
        if len(self.df) < window_size:
            window_size = len(self.df)
            
        self.df['elevation_smoothed'] = self.df['elevation'].rolling(
            window=window_size, center=True, min_periods=1).mean()
        
        # 填充可能的NaN值
        self.df['elevation_smoothed'] = self.df['elevation_smoothed'].fillna(method='bfill').fillna(method='ffill')
        
        return self.df
    
    def calculate_segment_data(self):
        """计算每个轨迹段的数据"""
        if self.df is None or len(self.df) == 0:
            self.parse_gpx()
            
        if 'elevation_smoothed' not in self.df.columns:
            self.smooth_elevation()
            
        distances = [0]
        elevations = [self.df['elevation_smoothed'].iloc[0]]
        
        for i in range(1, len(self.df)):
            lat1, lon1 = self.df.iloc[i-1]['latitude'], self.df.iloc[i-1]['longitude']
            lat2, lon2 = self.df.iloc[i]['latitude'], self.df.iloc[i]['longitude']
            dist = self.calculate_distance(lat1, lon1, lat2, lon2)
            distances.append(dist)
            elevations.append(self.df.iloc[i]['elevation_smoothed'])
        
        self.df['segment_distance'] = distances
        self.df['cumulative_distance'] = self.df['segment_distance'].cumsum()
        self.df['elevation_change'] = self.df['elevation_smoothed'].diff()
        
        # 处理路点，找到每个路点对应的轨迹点
        for wp in self.waypoints:
            nearest_idx, distance = self.find_nearest_track_point(wp['latitude'], wp['longitude'])
            if distance < 100:  # 只考虑距离轨迹100米以内的路点
                wp['track_index'] = nearest_idx
                wp['cumulative_distance'] = self.df.iloc[nearest_idx]['cumulative_distance']
                wp['elevation'] = self.df.iloc[nearest_idx]['elevation_smoothed']
        
        return self.df
    
    def calculate_slope(self):
        """计算每个段的坡度"""
        if self.df is None or 'segment_distance' not in self.df.columns:
            self.calculate_segment_data()
            
        # 避免除以零
        segment_dist_nonzero = self.df['segment_distance'].copy()
        segment_dist_nonzero[segment_dist_nonzero == 0] = np.nan
        
        # 计算坡度百分比（垂直变化/水平距离 * 100）
        # 使用更稳定的计算方法
        slopes = []
        for i, row in self.df.iterrows():
            if i == 0:
                slopes.append(0)
                continue
                
            dist = row['segment_distance']
            elev_change = row['elevation_change']
            
            # 使用更稳定的坡度计算方法
            if dist > 0:
                # 使用arctan计算角度，然后转换为百分比
                angle_rad = math.atan(elev_change / dist)
                slope_percent = math.tan(angle_rad) * 100
            else:
                slope_percent = 0
                
            slopes.append(slope_percent)
        
        self.df['slope_percent'] = slopes
        
        # 处理极端坡度值
        self.df['slope_percent'] = self.df['slope_percent'].clip(-100, 100)
        
        return self.df
    
    def get_slope_weight(self, slope):
        """根据坡度返回难度权重"""
        if slope > 15:
            return 4.0  # 极度陡峭上坡
        elif slope > 10:
            return 3.0  # 非常陡峭上坡
        elif slope > 5:
            return 2.0  # 陡峭上坡
        elif slope > 0:
            return 1.5  # 缓上坡
        elif slope > -5:
            return 1.0  # 平路或缓下坡
        elif slope > -10:
            return 1.2  # 陡下坡
        else:
            return 1.5  # 非常陡峭下坡
    
    def calculate_difficulty(self):
        """计算路线难度"""
        if self.df is None or len(self.df) == 0:
            self.parse_gpx()
            
        if 'slope_percent' not in self.df.columns:
            self.calculate_slope()
            
        total_distance = self.df['cumulative_distance'].iloc[-1] / 1000  # 转换为公里
        
        # 计算总爬升和总下降
        elevation_changes = self.df['elevation_change'].dropna()
        total_climb = elevation_changes[elevation_changes > 0].sum()
        total_descent = abs(elevation_changes[elevation_changes < 0].sum())
        
        # 计算加权等效距离
        weighted_distance = 0
        slope_distribution = {
            '>15%': 0, '10-15%': 0, '5-10%': 0, '0-5%': 0,
            '-5-0%': 0, '-10--5%': 0, '<-10%': 0
        }
        
        # 计算加权平均坡度（按距离加权）
        weighted_slope_sum = 0
        total_weighted_distance = 0
        
        for i, row in self.df.iterrows():
            if i == 0:
                continue
                
            slope = row['slope_percent']
            dist = row['segment_distance']
            
            # 计算加权距离
            weight = self.get_slope_weight(slope)
            weighted_distance += dist * weight
            
            # 统计坡度分布
            if slope > 15:
                slope_distribution['>15%'] += dist
            elif slope > 10:
                slope_distribution['10-15%'] += dist
            elif slope > 5:
                slope_distribution['5-10%'] += dist
            elif slope > 0:
                slope_distribution['0-5%'] += dist
            elif slope > -5:
                slope_distribution['-5-0%'] += dist
            elif slope > -10:
                slope_distribution['-10--5%'] += dist
            else:
                slope_distribution['<-10%'] += dist
            
            # 计算加权坡度（按距离加权）
            weighted_slope_sum += abs(slope) * dist
            total_weighted_distance += dist
        
        # 转换为公里
        weighted_distance_km = weighted_distance / 1000
        
        # 计算难度比率
        difficulty_ratio = weighted_distance_km / total_distance if total_distance > 0 else 1
        
        # 确定星级评分
        if difficulty_ratio < 1.1:
            stars = 1
            difficulty_text = "非常容易"
        elif difficulty_ratio < 1.3:
            stars = 2
            difficulty_text = "容易"
        elif difficulty_ratio < 1.6:
            stars = 3
            difficulty_text = "中等"
        elif difficulty_ratio < 2.0:
            stars = 4
            difficulty_text = "困难"
        else:
            stars = 5
            difficulty_text = "非常困难"
        
        # 计算最大和最小坡度
        slope_values = self.df['slope_percent'].dropna()
        max_slope = slope_values.max() if len(slope_values) > 0 else 0
        min_slope = slope_values.min() if len(slope_values) > 0 else 0
        
        # 计算平均坡度 - 按距离加权
        if total_weighted_distance > 0:
            avg_slope = weighted_slope_sum / total_weighted_distance
        else:
            avg_slope = 0
        
        # 存储结果
        self.results = {
            'total_distance_km': total_distance,
            'total_climb_m': total_climb,
            'total_descent_m': total_descent,
            'weighted_distance_km': weighted_distance_km,
            'difficulty_ratio': difficulty_ratio,
            'stars': stars,
            'difficulty_text': difficulty_text,
            'slope_distribution': slope_distribution,
            'max_slope': max_slope,
            'min_slope': min_slope,
            'avg_slope': avg_slope
        }
        
        return self.results
    
    def find_challenging_sections(self, threshold=8, min_length=200):
        """识别具有挑战性的路段"""
        if self.df is None or len(self.df) == 0:
            self.parse_gpx()
            
        if 'slope_percent' not in self.df.columns:
            self.calculate_slope()
            
        challenging_sections = []
        current_section = None
        
        for i, row in self.df.iterrows():
            slope = abs(row['slope_percent'])
            
            if slope >= threshold:
                if current_section is None:
                    current_section = {
                        'start_index': i,
                        'start_distance': row['cumulative_distance'],
                        'max_slope': slope
                    }
                else:
                    current_section['max_slope'] = max(current_section['max_slope'], slope)
            else:
                if current_section is not None:
                    current_section['end_index'] = i-1
                    current_section['end_distance'] = self.df.iloc[i-1]['cumulative_distance']
                    current_section['length'] = current_section['end_distance'] - current_section['start_distance']
                    
                    if current_section['length'] >= min_length:
                        challenging_sections.append(current_section)
                    
                    current_section = None
        
        # 处理最后一个路段
        if current_section is not None:
            current_section['end_index'] = len(self.df) - 1
            current_section['end_distance'] = self.df.iloc[-1]['cumulative_distance']
            current_section['length'] = current_section['end_distance'] - current_section['start_distance']
            
            if current_section['length'] >= min_length:
                challenging_sections.append(current_section)
        
        self.results['challenging_sections'] = challenging_sections
        return challenging_sections
    
    def plot_elevation_profile(self):
        """绘制海拔剖面图，标注CP点"""
        if self.df is None or len(self.df) == 0:
            self.parse_gpx()
            
        if self.results == {}:
            self.calculate_difficulty()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.df['cumulative_distance']/1000, self.df['elevation_smoothed'], 'b-', linewidth=2)
        
        # 标记挑战性路段
        for section in self.results.get('challenging_sections', []):
            start_km = section['start_distance']/1000
            end_km = section['end_distance']/1000
            ax.axvspan(start_km, end_km, color='red', alpha=0.3)
        
        # 标注CP点（检查点）
        if self.waypoints:
            for wp in self.waypoints:
                if 'cumulative_distance' in wp:  # 确保路点已关联到轨迹
                    x_pos = wp['cumulative_distance'] / 1000
                    y_pos = wp['elevation']
                    ax.plot(x_pos, y_pos, 'ro', markersize=8)  # 红色圆点标记
                    ax.annotate(wp['name'], 
                               xy=(x_pos, y_pos),
                               xytext=(5, 5),
                               textcoords='offset points',
                               fontsize=10,
                               color='red',
                               weight='bold')
                    ax.axvline(x=x_pos, color='green', linestyle='--', alpha=0.7)  # 垂直虚线
        
        ax.set_xlabel('距离 (km)')
        ax.set_ylabel('海拔 (m)')
        ax.set_title(f'{self.base_filename} - 海拔剖面图')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def plot_slope_distribution(self):
        """绘制坡度分布图"""
        if self.results == {}:
            self.calculate_difficulty()
        
        labels = list(self.results['slope_distribution'].keys())
        values = [self.results['slope_distribution'][k]/1000 for k in labels]  # 转换为公里
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(labels, values)
        ax.set_xlabel('坡度区间')
        ax.set_ylabel('距离 (km)')
        ax.set_title(f'{self.base_filename} - 坡度分布')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def generate_report(self):
        """生成分析报告"""
        if self.results == {}:
            self.calculate_difficulty()
            self.find_challenging_sections()
        
        report = f"""
        ====== {self.base_filename} - GPX跑步路线难度分析报告 ======
        
        基础信息:
        - 总距离: {self.results['total_distance_km']:.2f} km
        - 总爬升: {self.results['total_climb_m']:.0f} m
        - 总下降: {self.results['total_descent_m']:.0f} m
        - 平均坡度: {self.results['avg_slope']:.1f}%
        - 最大坡度: {self.results['max_slope']:.1f}%
        - 最小坡度: {self.results['min_slope']:.1f}%
        
        难度评估:
        - 加权等效距离: {self.results['weighted_distance_km']:.2f} km
        - 难度比率: {self.results['difficulty_ratio']:.2f}
        - 星级评分: {'★' * self.results['stars']} ({self.results['difficulty_text']})
        """
        
        # 添加CP点信息
        if self.waypoints:
            report += "\n检查点(CP点)信息:\n"
            for i, wp in enumerate(self.waypoints):
                if 'cumulative_distance' in wp:  # 确保路点已关联到轨迹
                    report += (f"- {wp['name']}: 距离起点 {wp['cumulative_distance']/1000:.2f} km, "
                              f"海拔 {wp['elevation']:.0f} m")
                    if wp['description']:
                        report += f", 描述: {wp['description']}"
                    report += "\n"
        
        report += "\n坡度分布:\n"
        for slope_range, dist in self.results['slope_distribution'].items():
            report += f"- {slope_range}: {dist/1000:.2f} km\n"
        
        if self.results.get('challenging_sections'):
            report += "\n挑战性路段:\n"
            for i, section in enumerate(self.results['challenging_sections']):
                report += (f"{i+1}. 从 {section['start_distance']/1000:.2f} km 到 "
                          f"{section['end_distance']/1000:.2f} km, "
                          f"长度: {section['length']:.0f} m, "
                          f"最大坡度: {section['max_slope']:.1f}%\n")
        
        return report
    
    def save_report_to_file(self, report):
        """将报告保存到文件"""
        report_filename = f"{self.base_filename}_report.txt"
        try:
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"分析报告已保存为: {report_filename}")
            return report_filename
        except Exception as e:
            print(f"保存报告时出错: {str(e)}")
            return None
    
    def analyze(self):
        """执行完整分析"""
        try:
            print("开始解析GPX文件...")
            self.parse_gpx()
            
            if self.df is None or len(self.df) == 0:
                raise ValueError("GPX文件解析失败或没有轨迹数据")
            
            print("预处理海拔数据...")
            self.smooth_elevation()
            
            print("计算距离和坡度...")
            self.calculate_segment_data()
            self.calculate_slope()
            
            print("计算路线难度...")
            self.calculate_difficulty()
            
            print("识别挑战性路段...")
            self.find_challenging_sections()
            
            print("生成报告...")
            report = self.generate_report()
            
            # 保存报告到文件
            report_filename = self.save_report_to_file(report)
            
            print("绘制图表...")
            elevation_fig = self.plot_elevation_profile()
            slope_fig = self.plot_slope_distribution()

            # 根据GPX文件名生成PNG文件名
            elevation_filename = f"{self.base_filename}_elevation_profile.png"
            slope_filename = f"{self.base_filename}_slope_distribution.png"
            
            # 保存图表
            elevation_fig.savefig(elevation_filename, dpi=300, bbox_inches='tight')
            slope_fig.savefig(slope_filename, dpi=300, bbox_inches='tight')
            print(f"图表已保存为 {elevation_filename} 和 {slope_filename}")
            
            # 先打印报告
            print("\n" + "="*60)
            print(report)
            print("="*60)
            
            # 然后询问用户是否要显示图表
            show_plots = input("\n是否要显示图表？(y/n): ").lower().strip()
            if show_plots == 'y':
                plt.show()
            
            return report, elevation_fig, slope_fig
            
        except Exception as e:
            error_msg = f"分析过程中出现错误: {str(e)}"
            print(error_msg)
            return error_msg, None, None

# 使用示例
if __name__ == "__main__":
    # 初始化分析器
    gpx_file_path = input("请输入GPX文件路径: ").strip().strip('"')
    analyzer = GPXRunningAnalyzer(gpx_file_path)
    
    # 执行完整分析
    report, elevation_fig, slope_fig = analyzer.analyze()