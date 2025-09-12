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
        """计算每个轨迹段的数据 - 使用路面距离而不是水平距离"""
        if self.df is None or len(self.df) == 0:
            self.parse_gpx()
            
        if 'elevation_smoothed' not in self.df.columns:
            self.smooth_elevation()
            
        distances = [0]  # 第一个点的距离为0
        elevations = [self.df['elevation_smoothed'].iloc[0]]
        
        # 确保数据点按时间排序 时间有可能有问题 比如没有精确到足以排序
        #if 'time' in self.df.columns and not self.df['time'].isnull().all():
        #    self.df = self.df.sort_values('time').reset_index(drop=True)
        
        # 计算每段距离 - 使用路面距离（考虑海拔变化的斜距）
        for i in range(1, len(self.df)):
            lat1, lon1 = self.df.iloc[i-1]['latitude'], self.df.iloc[i-1]['longitude']
            lat2, lon2 = self.df.iloc[i]['latitude'], self.df.iloc[i]['longitude']
            
            # 计算水平距离
            horizontal_dist = self.calculate_distance(lat1, lon1, lat2, lon2)
            
            # 计算海拔变化
            elev_change = self.df.iloc[i]['elevation_smoothed'] - self.df.iloc[i-1]['elevation_smoothed']
            
            # 计算路面距离（斜距）- 使用勾股定理
            road_distance = math.sqrt(horizontal_dist**2 + elev_change**2)
            
            distances.append(road_distance)
            elevations.append(self.df.iloc[i]['elevation_smoothed'])
        
        self.df['segment_distance'] = distances
        self.df['cumulative_distance'] = self.df['segment_distance'].cumsum()
        self.df['elevation_change'] = self.df['elevation_smoothed'].diff()

        # 计算水平距离（用于坡度计算）
        self.df['horizontal_distance'] = np.sqrt(
            np.maximum(0, self.df['segment_distance']**2 - self.df['elevation_change']**2)
        )
        # 处理可能的NaN值
        self.df['horizontal_distance'] = self.df['horizontal_distance'].fillna(0)
        
        # 找到最高海拔点（使用原始数据，而不是平滑后的数据）
        max_elevation_idx = self.df['elevation'].idxmax()
        self.results['max_elevation_point'] = {
            'distance': self.df.iloc[max_elevation_idx]['cumulative_distance'],
            'elevation': self.df.iloc[max_elevation_idx]['elevation'],
            'index': max_elevation_idx
        }
        
        # 处理路点，找到每个路点对应的轨迹点
        for wp in self.waypoints:
            nearest_idx, distance = self.find_nearest_track_point(wp['latitude'], wp['longitude'])
            if distance < 100:  # 只考虑距离轨迹100米以内的路点
                wp['track_index'] = nearest_idx
                wp['cumulative_distance'] = self.df.iloc[nearest_idx]['cumulative_distance']
                wp['elevation'] = self.df.iloc[nearest_idx]['elevation']
        
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
    
    # 识别连续大坡度路段
    def find_challenging_sections(self, threshold=8, min_length=200):
        """识别具有挑战性的路段，并记录起点和终点海拔"""
        if self.df is None or len(self.df) == 0:
            self.parse_gpx()
            
        if 'slope_percent' not in self.df.columns:
            self.calculate_slope()
            
        challenging_sections = []
        current_section = None
        
        for i, row in self.df.iterrows():
            slope = row['slope_percent']
            
            if abs(slope) >= threshold:
                if current_section is None:
                    current_section = {
                        'start_index': i,
                        'start_distance': row['cumulative_distance'],
                        'start_elevation': row['elevation_smoothed'],
                        'max_slope': slope,
                        'min_slope': slope,  # 添加最小坡度记录
                        'slope_sum': 0,      # 用于计算平均坡度
                        'point_count': 0,     # 用于计算平均坡度
                        'type': '连续大坡度'  # 连续大坡度
                    }
                else:
                    # 更新最大和最小坡度
                    if abs(slope) > abs(current_section['max_slope']):
                        current_section['max_slope'] = slope
                    if abs(slope) < abs(current_section['min_slope']):
                        current_section['min_slope'] = slope
                    
                    # 累加坡度和点数
                    current_section['slope_sum'] += abs(slope)
                    current_section['point_count'] += 1
            else:
                if current_section is not None:
                    current_section['end_index'] = i-1
                    current_section['end_distance'] = self.df.iloc[i-1]['cumulative_distance']
                    current_section['end_elevation'] = self.df.iloc[i-1]['elevation_smoothed']
                    current_section['length'] = current_section['end_distance'] - current_section['start_distance']
                    current_section['elevation_change'] = current_section['end_elevation'] - current_section['start_elevation']
                    
                    # 计算路段平均坡度（基于逐点计算）
                    if current_section['point_count'] > 0:
                        current_section['avg_slope'] = current_section['slope_sum'] / current_section['point_count']
                    else:
                        current_section['avg_slope'] = current_section['max_slope']
                    
                    # 计算基于海拔差和距离差的坡度
                    if current_section['length'] > 0:
                        # 使用arctan计算真实坡度
                        angle_rad = math.atan(current_section['elevation_change'] / current_section['length'])
                        current_section['section_slope'] = math.tan(angle_rad) * 100
                    else:
                        current_section['section_slope'] = 0
                    
                    if current_section['length'] >= min_length:
                        challenging_sections.append(current_section)
                    
                    current_section = None
        
        # 处理最后一个路段
        if current_section is not None:
            current_section['end_index'] = len(self.df) - 1
            current_section['end_distance'] = self.df.iloc[-1]['cumulative_distance']
            current_section['end_elevation'] = self.df.iloc[-1]['elevation_smoothed']
            current_section['length'] = current_section['end_distance'] - current_section['start_distance']
            current_section['elevation_change'] = current_section['end_elevation'] - current_section['start_elevation']
            
            # 计算路段平均坡度（基于逐点计算）
            if current_section['point_count'] > 0:
                current_section['avg_slope'] = current_section['slope_sum'] / current_section['point_count']
            else:
                current_section['avg_slope'] = current_section['max_slope']
            
            # 计算基于海拔差和距离差的坡度
            if current_section['length'] > 0:
                angle_rad = math.atan(current_section['elevation_change'] / current_section['length'])
                current_section['section_slope'] = math.tan(angle_rad) * 100
            else:
                current_section['section_slope'] = 0
            
            if current_section['length'] >= min_length:
                challenging_sections.append(current_section)
        
        self.results['challenging_sections'] = challenging_sections
        return challenging_sections

    # 识别段落大坡度路段
    def find_segment_challenging_sections(self, min_length=500, avg_slope_threshold=5, 
                                     steep_slope_threshold=15, steep_ratio_threshold=30):
        """识别段落大坡度路段（较长距离内平均坡度较大）
    
        参数:
            min_length: 段落最小长度（米）
            avg_slope_threshold: 平均坡度阈值（%）
            steep_slope_threshold: 陡坡阈值（%）
            steep_ratio_threshold: 陡坡比例阈值（%）
        """
        if self.df is None or len(self.df) == 0:
            self.parse_gpx()
            
        if 'slope_percent' not in self.df.columns:
            self.calculate_slope()
            
        segment_sections = []
        
        # 将路线分成多个段落
        total_distance = self.df['cumulative_distance'].iloc[-1]
        num_segments = int(total_distance / min_length) + 1
        
        for i in range(num_segments):
            start_dist = i * min_length
            end_dist = min((i + 1) * min_length, total_distance)
            
            # 找到段落开始和结束的索引
            start_idx = (self.df['cumulative_distance'] - start_dist).abs().idxmin()
            end_idx = (self.df['cumulative_distance'] - end_dist).abs().idxmin()
            
            if start_idx >= end_idx:
                continue
                
            segment_df = self.df.iloc[start_idx:end_idx+1]
            
            # 计算段落数据
            segment_length = segment_df['cumulative_distance'].iloc[-1] - segment_df['cumulative_distance'].iloc[0]
            elevation_change = segment_df['elevation_smoothed'].iloc[-1] - segment_df['elevation_smoothed'].iloc[0]
            
            # 计算平均坡度
            if segment_length > 0:
                angle_rad = math.atan(elevation_change / segment_length)
                avg_slope = math.tan(angle_rad) * 100
            else:
                avg_slope = 0
            
            # 计算段落内坡度≥steep_slope_threshold%的距离比例
            steep_distance = segment_df[abs(segment_df['slope_percent']) >= steep_slope_threshold]['horizontal_distance'].sum()
            steep_ratio = steep_distance / segment_length if segment_length > 0 else 0
            
            # 如果平均坡度超过阈值且有一定比例的陡坡，则认为是挑战性段落
            if abs(avg_slope) >= avg_slope_threshold and steep_ratio * 100 >= steep_ratio_threshold:
                segment_section = {
                    'type': '段落大坡度',
                    'start_index': start_idx,
                    'end_index': end_idx,
                    'start_distance': segment_df['cumulative_distance'].iloc[0],
                    'end_distance': segment_df['cumulative_distance'].iloc[-1],
                    'length': segment_length,
                    'start_elevation': segment_df['elevation_smoothed'].iloc[0],
                    'end_elevation': segment_df['elevation_smoothed'].iloc[-1],
                    'elevation_change': elevation_change,
                    'avg_slope': avg_slope,
                    'steep_ratio': steep_ratio * 100,  # 转换为百分比
                    'max_slope': segment_df['slope_percent'].max(),
                    'min_slope': segment_df['slope_percent'].min(),
                    'steep_slope_threshold': steep_slope_threshold,  # 记录使用的阈值
                    'steep_ratio_threshold': steep_ratio_threshold   # 记录使用的阈值  
                }
                segment_sections.append(segment_section)
        
        # 将段落大坡度路段添加到挑战性路段中
        if 'challenging_sections' not in self.results:
            self.results['challenging_sections'] = []
        
        # 标记原有的连续大坡度路段类型
        for section in self.results['challenging_sections']:
            section['type'] = '连续大坡度'
        
        # 添加新的段落大坡度路段
        self.results['challenging_sections'].extend(segment_sections)
        
        # 按起始距离排序
        self.results['challenging_sections'].sort(key=lambda x: x['start_distance'])
        
        return self.results['challenging_sections']

    def get_kilometer_data(self):
        """获取每公里的数据 - 使用路面距离而不是水平距离"""
        if self.df is None or len(self.df) == 0:
            self.parse_gpx()
            
        if 'cumulative_distance' not in self.df.columns:
            self.calculate_segment_data()
            
        # 确保数据按累计距离排序
        sorted_df = self.df.sort_values('cumulative_distance').reset_index(drop=True)
        
        total_distance_km = sorted_df['cumulative_distance'].iloc[-1] / 1000
        kilometer_data = []
        
        # 找到每公里位置的数据
        for km in range(1, int(total_distance_km) + 1):
            target_distance = km * 1000  # 转换为米
            
            # 找到最接近目标距离的点
            distance_diff = (sorted_df['cumulative_distance'] - target_distance).abs()
            closest_idx = distance_diff.idxmin()
            closest_row = sorted_df.iloc[closest_idx]
            
            # 计算该公里段的爬升和下降
            start_idx = (sorted_df['cumulative_distance'] - (target_distance - 1000)).abs().idxmin()
            end_idx = closest_idx
            
            # 确保索引有效
            if start_idx >= len(sorted_df):
                start_idx = len(sorted_df) - 1
            if end_idx >= len(sorted_df):
                end_idx = len(sorted_df) - 1
                
            if start_idx < end_idx:
                segment_df = sorted_df.iloc[start_idx:end_idx+1]
                
                # 计算该公里段的海拔变化
                elevation_changes = segment_df['elevation_change']
                
                # 计算爬升和下降
                climb = elevation_changes[elevation_changes > 0].sum()
                descent = abs(elevation_changes[elevation_changes < 0].sum())
                
                # 计算该公里段的平均海拔
                #avg_elevation = segment_df['elevation'].mean()
                
                # 计算该公里段开始和结束的海拔
                start_elevation = sorted_df.iloc[start_idx]['elevation']
                end_elevation = sorted_df.iloc[end_idx]['elevation']
            else:
                climb = 0
                descent = 0
                avg_elevation = closest_row['elevation']
                start_elevation = closest_row['elevation']
                end_elevation = closest_row['elevation']
            
            kilometer_data.append({
                '公里': km,
                '距离(km)': closest_row['cumulative_distance'] / 1000,
                '海拔(m)': closest_row['elevation_smoothed'],  # 使用终点海拔
                '开始海拔(m)': start_elevation,
                '结束海拔(m)': end_elevation,
                '爬升(m)': climb,
                '下降(m)': descent
            })
        
        # 找到最高海拔点（使用原始海拔数据，而不是平滑后的数据）
        max_elevation_idx = sorted_df['elevation'].idxmax()
        max_elevation_distance = sorted_df.iloc[max_elevation_idx]['cumulative_distance'] / 1000
        max_elevation_value = sorted_df.iloc[max_elevation_idx]['elevation']
        
        self.results['kilometer_data'] = kilometer_data
        self.results['max_elevation_point'] = {
            'distance_km': max_elevation_distance,
            'elevation': max_elevation_value
        }
        
        return kilometer_data
    
    def generate_excel_report(self):
        """生成Excel格式的详细数据报告，包含多个图表"""
        if self.df is None or len(self.df) == 0:
            self.parse_gpx()
            
        if self.results == {}:
            self.calculate_difficulty()
            
        if 'kilometer_data' not in self.results:
            self.get_kilometer_data()
            
        # 创建Excel文件名
        excel_filename = f"{self.base_filename}_elevation_profile.xlsx"
        
        # 创建Excel写入器
        with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
            # 1. 添加详细轨迹数据表
            detailed_df = self.df.copy()
            detailed_df['累计距离(km)'] = detailed_df['cumulative_distance'] / 1000
            detailed_df['海拔(m)'] = detailed_df['elevation_smoothed']
            detailed_df['坡度(%)'] = detailed_df['slope_percent']
            
            # 选择需要的列
            detailed_df = detailed_df[['累计距离(km)', '海拔(m)', '坡度(%)']]
            detailed_df.to_excel(writer, sheet_name='详细数据', index=False)
            
            # 2. 添加每公里数据表
            km_df = pd.DataFrame(self.results['kilometer_data'])
            km_df.to_excel(writer, sheet_name='每公里数据', index=False)
            
            # 3. 添加坡度分布数据表
            slope_data = []
            for slope_range, dist in self.results['slope_distribution'].items():
                slope_data.append({
                    '坡度区间': slope_range,
                    '距离(m)': dist,
                    '距离(km)': dist / 1000
                })
            slope_df = pd.DataFrame(slope_data)
            slope_df.to_excel(writer, sheet_name='坡度分布', index=False)
            
            # 4. 添加检查点数据表
            if self.waypoints:
                cp_data = []
                for wp in self.waypoints:
                    if 'cumulative_distance' in wp:
                        cp_data.append({
                            '名称': wp['name'],
                            '距离起点(km)': wp['cumulative_distance'] / 1000,
                            '海拔(m)': wp['elevation'],
                            '描述': wp['description']
                        })
                cp_df = pd.DataFrame(cp_data)
                cp_df.to_excel(writer, sheet_name='检查点', index=False)
            
            # 5. 添加挑战性路段数据表
            if self.results.get('challenging_sections'):
                challenge_data = []
                for i, section in enumerate(self.results['challenging_sections']):
                    challenge_data.append({
                        '序号': i + 1,
                        '类型': section.get('type', '未知'),  # 添加类型列
                        '起始距离(km)': section['start_distance'] / 1000,
                        '结束距离(km)': section['end_distance'] / 1000,
                        '长度(m)': section['length'],
                        '最大坡度(%)': section['max_slope'],
                        '平均坡度(%)': section['avg_slope'],
                        '路段坡度(%)': section.get('section_slope', 0),
                        '起点海拔(m)': section['start_elevation'],
                        '终点海拔(m)': section['end_elevation'],
                        '海拔变化(m)': section['elevation_change'],
                        '陡坡比例(%)': section.get('steep_ratio', 0),  # 添加陡坡比例列
                        '陡坡阈值(%)': section.get('steep_slope_threshold', 0),  # 添加陡坡阈值
                        '比例阈值(%)': section.get('steep_ratio_threshold', 0)   # 添加比例阈值
                    })
                challenge_df = pd.DataFrame(challenge_data)
                challenge_df.to_excel(writer, sheet_name='挑战性路段', index=False)
            
            # 6. 添加汇总数据表
            summary_data = {
                '指标': [
                    '总距离(km)', '总爬升(m)', '总下降(m)', 
                    '平均坡度(%)', '最大坡度(%)', '最小坡度(%)',
                    '加权等效距离(km)', '难度比率', '星级评分'
                ],
                '数值': [
                    self.results['total_distance_km'],
                    self.results['total_climb_m'],
                    self.results['total_descent_m'],
                    self.results['avg_slope'],
                    self.results['max_slope'],
                    self.results['min_slope'],
                    self.results['weighted_distance_km'],
                    self.results['difficulty_ratio'],
                    f"{'★' * self.results['stars']} ({self.results['difficulty_text']})"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='汇总数据', index=False)
            
            # 获取工作簿对象
            workbook = writer.book
            
            # 7. 在"详细数据"工作表中创建海拔剖面图
            detailed_worksheet = writer.sheets['详细数据']
            data_range = len(detailed_df)
            
            detailed_worksheet = writer.sheets['详细数据']
            data_range = len(detailed_df)

            # 计算合理的纵坐标范围
            elevation_data = detailed_df['海拔(m)']
            min_elevation = elevation_data.min()
            max_elevation = elevation_data.max()

            # 计算 y_min：向下取整到百位，并确保小于 min_elevation
            y_min = math.floor(min_elevation / 100) * 100
            if y_min >= min_elevation:
                y_min -= 100

            # 计算 y_max：向上取整到百位，并确保大于 max_elevation
            y_max = math.ceil(max_elevation / 100) * 100
            if y_max <= max_elevation:
                y_max += 100

            # 创建海拔剖面图
            elevation_chart = workbook.add_chart({'type': 'scatter', 'subtype': 'straight_with_markers'})
            elevation_chart.add_series({
                'name': '海拔剖面',
                'categories': ['详细数据', 1, 0, data_range, 0],
                'values': ['详细数据', 1, 1, data_range, 1],
                'marker': {'type': 'circle', 'size': 2},
                'line': {'width': 1.5}
            })
            elevation_chart.set_title({'name': f'{self.base_filename} - 海拔剖面图'})
            elevation_chart.set_x_axis({'name': '距离 (km)', 'min': 0})
            elevation_chart.set_y_axis({
                'name': '海拔 (m)',
                'min': y_min,  # 设置最小纵坐标值
                'max': y_max   # 设置最大纵坐标值
            })
            elevation_chart.set_size({'width': 800, 'height': 400})
            detailed_worksheet.insert_chart('E2', elevation_chart)
            
            # 8. 在"每公里数据"工作表中创建图表
            km_worksheet = writer.sheets['每公里数据']
            km_data_range = len(km_df)
            # 创建每公里爬升与下降柱状图 - 使用正确的数据列
            climb_descent_chart = workbook.add_chart({'type': 'column'})
            # 爬升系列 - 使用"爬升(m)"列
            climb_descent_chart.add_series({
                'name': '爬升',
                'categories': ['每公里数据', 1, 0, km_data_range, 0],  # 公里数列
                'values': ['每公里数据', 1, 5, km_data_range, 5],      # 爬升(m)列
                'fill':   {'color': 'red'}
            })
            # 下降系列 - 使用"下降(m)"列
            climb_descent_chart.add_series({
                'name': '下降',
                'categories': ['每公里数据', 1, 0, km_data_range, 0],  # 公里数列
                'values': ['每公里数据', 1, 6, km_data_range, 6],      # 下降(m)列
                'fill':   {'color': 'green'}
            })
            climb_descent_chart.set_title({'name': '每公里爬升与下降'})
            climb_descent_chart.set_x_axis({'name': '公里数'})
            climb_descent_chart.set_y_axis({'name': '高度变化 (m)'})
            climb_descent_chart.set_size({'width': 600, 'height': 400})
            km_worksheet.insert_chart('K20', climb_descent_chart)

            # 9. 在"坡度分布"工作表中创建图表
            slope_worksheet = writer.sheets['坡度分布']
            slope_data_range = len(slope_df)
            
            # 创建坡度分布柱状图
            slope_chart = workbook.add_chart({'type': 'column'})
            slope_chart.add_series({
                'name': '坡度分布',
                'categories': ['坡度分布', 1, 0, slope_data_range, 0],  # 坡度区间列
                'values': ['坡度分布', 1, 2, slope_data_range, 2],      # 距离(km)列
                'fill': {'color': '#4472C4'}
            })
            slope_chart.set_title({'name': '坡度分布统计'})
            slope_chart.set_x_axis({'name': '坡度区间'})
            slope_chart.set_y_axis({'name': '距离 (km)'})
            slope_chart.set_size({'width': 600, 'height': 400})
            slope_worksheet.insert_chart('E2', slope_chart)
            
            # 创建坡度分布饼图
            slope_pie_chart = workbook.add_chart({'type': 'pie'})
            slope_pie_chart.add_series({
                'name': '坡度分布比例',
                'categories': ['坡度分布', 1, 0, slope_data_range, 0],  # 坡度区间列
                'values': ['坡度分布', 1, 2, slope_data_range, 2],      # 距离(km)列
                'data_labels': {'percentage': True, 'category': True}
            })
            slope_pie_chart.set_title({'name': '坡度分布比例'})
            slope_pie_chart.set_size({'width': 600, 'height': 400})
            slope_worksheet.insert_chart('E20', slope_pie_chart)
        
        print(f"Excel文件已保存为: {excel_filename}")
        return excel_filename
    
    def plot_elevation_profile(self):
        """绘制海拔剖面图 - 使用路面距离作为X轴，修复图例标签检查"""
        if self.df is None or len(self.df) == 0:
            self.parse_gpx()
            
        if self.results == {}:
            self.calculate_difficulty()
            
        # 获取每公里数据
        if 'kilometer_data' not in self.results:
            self.get_kilometer_data()
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 确保数据点按累计距离排序
        sorted_df = self.df.sort_values('cumulative_distance').reset_index(drop=True)
        
        # 绘制原始海拔数据（细线）
        ax.plot(sorted_df['cumulative_distance']/1000, sorted_df['elevation'], 
                'gray', linewidth=1, alpha=0.5, label='原始海拔')
        
        # 绘制平滑后的海拔数据（粗线）
        ax.plot(sorted_df['cumulative_distance']/1000, sorted_df['elevation_smoothed'], 
                'b-', linewidth=2, label='平滑海拔')
        
        # 使用集合来跟踪已添加的图例标签
        added_labels = set()
        
        # 标记挑战性路段 - 区分连续大坡度和段落大坡度
        for section in self.results.get('challenging_sections', []):
            start_km = section['start_distance']/1000
            end_km = section['end_distance']/1000
            
            # 根据路段类型选择不同颜色
            if section.get('type') == '连续大坡度':
                color = 'red'
                label = '连续大坡度路段'
            elif section.get('type') == '段落大坡度':
                color = 'orange'
                label = '段落大坡度路段'
            else:
                #color = 'black'
                label = '挑战性路段'
                
            ax.axvspan(start_km, end_km, color=color, alpha=0.3, 
                    label=label if label not in added_labels else "")
            if label not in added_labels:
                added_labels.add(label)
        
        # 标注最高海拔点
        if 'max_elevation_point' in self.results:
            max_elev = self.results['max_elevation_point']
            x_pos = max_elev['distance_km'] #/ 1000
            y_pos = max_elev['elevation']
            label = '最高海拔点'
            ax.plot(x_pos, y_pos, 'm*', markersize=15, 
                    label=label if label not in added_labels else "")
            if label not in added_labels:
                added_labels.add(label)
            ax.annotate(f"最高点: {x_pos:.2f}km\n{y_pos:.0f}m", 
                    xy=(x_pos, y_pos),
                    xytext=(15, 15),
                    textcoords='offset points',
                    fontsize=10,
                    weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="magenta", alpha=0.7),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3"))
        
        # 标注CP点（检查点）
        if self.waypoints:
            for i, wp in enumerate(self.waypoints):
                if 'cumulative_distance' in wp:  # 确保路点已关联到轨迹
                    x_pos = wp['cumulative_distance'] / 1000
                    y_pos = wp['elevation']
                    label = '检查点'
                    ax.plot(x_pos, y_pos, 'ro', markersize=8, 
                        label=label if i == 0 and label not in added_labels else "")
                    if i == 0 and label not in added_labels:
                        added_labels.add(label)
                    ax.annotate(wp['name'], 
                            xy=(x_pos, y_pos),
                            xytext=(5, 5),
                            textcoords='offset points',
                            fontsize=10,
                            color='red',
                            weight='bold')
                    ax.axvline(x=x_pos, color='green', linestyle='--', alpha=0.7)  # 垂直虚线
        
        # 标注每公里数据点
        for km_data in self.results.get('kilometer_data', []):
            x_pos = km_data['距离(km)']
            y_pos = km_data['海拔(m)'] 
            
            # 绘制每公里标记点
            label = '每公里点'
            ax.plot(x_pos, y_pos, 'go', markersize=6, alpha=0.7, 
                label=label if km_data['公里'] == 1 and label not in added_labels else "")
            if km_data['公里'] == 1 and label not in added_labels:
                added_labels.add(label)
            
            # 添加每公里数据标注
            if km_data['公里'] % 5 == 0:  # 每5公里标注一次，避免过于拥挤
                ax.annotate(f"{km_data['公里']}km\n{km_data['海拔(m)']:.0f}m", 
                        xy=(x_pos, y_pos),
                        xytext=(10, 10),
                        textcoords='offset points',
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"))
        
        ax.set_xlabel('距离起点的路面距离 (km)')
        ax.set_ylabel('海拔 (m)')
        ax.set_title(f'{self.base_filename} - 海拔剖面图 (含每公里数据)')
        ax.grid(True, alpha=0.3)
        
        # 添加图例
        ax.legend(loc='best')
        
        # 确保x轴从0开始
        ax.set_xlim(left=0)
        
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
            self.find_segment_challenging_sections()
            
        # 获取每公里数据
        if 'kilometer_data' not in self.results:
            self.get_kilometer_data()
        
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
        
        # 添加每公里数据
        report += "\n每公里数据:\n"
        report += "公里 | 距离(km) | 海拔(m) | 爬升(m) | 下降(m)\n"
        report += "----|---------|---------|---------|--------\n"
        
        for km_data in self.results.get('kilometer_data', []):
            report += (f"{km_data['公里']:3} | {km_data['距离(km)']:7.2f} | {km_data['海拔(m)']:7.0f} | "
                      f"{km_data['爬升(m)']:7.0f} | {km_data['下降(m)']:7.0f}\n")
        
        report += "\n坡度分布:\n"
        for slope_range, dist in self.results['slope_distribution'].items():
            report += f"- {slope_range}: {dist/1000:.2f} km\n"
        
        if self.results.get('challenging_sections'):
            report += "\n挑战性路段:\n"
            
            # 按类型分组
            continuous_sections = [s for s in self.results['challenging_sections'] if s.get('type') == '连续大坡度']
            segment_sections = [s for s in self.results['challenging_sections'] if s.get('type') == '段落大坡度']
            
            if continuous_sections:
                report += "\n连续大坡度路段:\n"
                for i, section in enumerate(continuous_sections):
                    report += (f"{i+1}. 从 {section['start_distance']/1000:.2f} km 到 "
                            f"{section['end_distance']/1000:.2f} km, "
                            f"长度: {section['length']:.0f} m, "
                            f"最大坡度: {section['max_slope']:.1f}%, "
                            f"平均坡度: {section['avg_slope']:.1f}%, "
                            f"路段坡度: {section.get('section_slope', 0):.1f}%, "
                            f"起点海拔: {section['start_elevation']:.0f} m, "
                            f"终点海拔: {section['end_elevation']:.0f} m, "
                            f"海拔变化: {section['elevation_change']:+.0f} m\n")
            
            if segment_sections:
                report += "\n段落大坡度路段:\n"
                for i, section in enumerate(segment_sections):
                    report += (f"{i+1}. 从 {section['start_distance']/1000:.2f} km 到 "
                            f"{section['end_distance']/1000:.2f} km, "
                            f"长度: {section['length']:.0f} m, "
                            f"最大坡度: {section['max_slope']:.1f}%, "
                            f"最小坡度: {section['min_slope']:.1f}%, "
                            f"平均坡度: {section['avg_slope']:.1f}%, "
                            f"陡坡比例: {section.get('steep_ratio', 0):.1f}%, "
                            f"起点海拔: {section['start_elevation']:.0f} m, "
                            f"终点海拔: {section['end_elevation']:.0f} m, "
                            f"海拔变化: {section['elevation_change']:+.0f} m\n")
        
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
            
            print("识别连续大坡度路段...")
            self.find_challenging_sections()  
        
            print("识别段落大坡度路段...")
            self.find_segment_challenging_sections()
            
            print("计算每公里数据...")
            self.get_kilometer_data()
            
            print("生成报告...")
            report = self.generate_report()
            
            # 保存报告到文件
            report_filename = self.save_report_to_file(report)
            
            print("生成Excel文件（含多个图表）...")
            excel_filename = self.generate_excel_report()
            
            print("绘制图表（仅供显示）...")
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
            
            return report, elevation_fig, excel_filename
            
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
    report, elevation_fig, excel_filename = analyzer.analyze()