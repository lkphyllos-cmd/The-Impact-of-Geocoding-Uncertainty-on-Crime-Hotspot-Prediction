# File: Experiment_for_one_Group.py
# Description: 为所有实验情景和所有误差占比，进行100次蒙特卡洛模拟，生成密度图和热点图，并计算预测精度。

import os
import arcpy
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import random
import shutil
import argparse
import re
from tqdm import tqdm
from arcpy.sa import *
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
import warnings

# --- 警告抑制与环境设置 ---
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
arcpy.CheckOutExtension("Spatial")

# ============================================================================
# ==                       辅助函数定义区域 (ESSENTIAL)                       ==
# ============================================================================

# --- 文本标准化函数 ---
STREET_TYPE_MAPPING = {
    'AVENUE': 'AVE', 'AV': 'AVE', 'BOULEVARD': 'BLVD', 'BOUL': 'BLVD',
    'CIRCLE': 'CIR', 'COURT': 'CT', 'DRIVE': 'DR', 'EXPRESSWAY': 'EXPY',
    'FREEWAY': 'FWY', 'HIGHWAY': 'HWY', 'LANE': 'LN', 'PARKWAY': 'PKWY',
    'PLACE': 'PL', 'ROAD': 'RD', 'SQUARE': 'SQ', 'STREET': 'ST', 'TERRACE': 'TER',
}
def ultimate_normalize_street_name(name):
    if not isinstance(name, str): return ""
    name_upper = name.upper().strip()
    if "BLOCK OF" in name_upper:
        street_part = name_upper.split("BLOCK OF", 1)[1].strip()
    else:
        street_part = name_upper
        parts_check = name_upper.split()
        if "BLOCK" in parts_check:
            try:
                block_index = parts_check.index("BLOCK")
                if block_index < len(parts_check) - 1: street_part = ' '.join(parts_check[block_index + 1:])
            except ValueError: pass
    street_part = re.sub(r'[^\w\s]', '', street_part)
    street_part = re.sub(r'\s+', ' ', street_part).strip()
    parts = street_part.split()
    if parts:
        for i, part in enumerate(parts):
            if part in STREET_TYPE_MAPPING: parts[i] = STREET_TYPE_MAPPING[part]
    return " ".join(parts)

# --- 统一预匹配函数 ---
def pre_match_points_to_streets_and_areals(points_gdf, streets_gdf, areal_layers, point_street_col, street_layer_col):
    """为每个点一次性匹配所有需要的街道和面状单元几何。"""
    print("\n--- 正在为点进行全面的预匹配 (沿线 + 区域) ---")
    
    print("  正在统一所有图层的坐标系...")
    # 使用点的坐标系作为我们的目标标准
    target_crs = points_gdf.crs
    
    if streets_gdf.crs != target_crs:
        streets_gdf = streets_gdf.to_crs(target_crs)
    
    for scale_name in areal_layers:
        if areal_layers[scale_name].crs != target_crs:
            areal_layers[scale_name] = areal_layers[scale_name].to_crs(target_crs)
    
    gdf_final = points_gdf.copy()

    # --- 第一部分: 匹配街道几何 (沿线误差) ---
    print("\n  (1/2) 正在匹配街道 (沿线误差)...")
    
    if gdf_final.crs != streets_gdf.crs:
        streets_gdf = streets_gdf.to_crs(gdf_final.crs)
        
    print("    - 第1轮: 正在通过标准化名称进行匹配...")
    gdf_final['norm_street_name_point'] = gdf_final[point_street_col].apply(ultimate_normalize_street_name)
    streets_gdf['norm_street_name_street'] = streets_gdf[street_layer_col].apply(ultimate_normalize_street_name)
    streets_gdf_dissolved = streets_gdf.dissolve(by='norm_street_name_street', aggfunc='first').reset_index()
    street_geom_map = streets_gdf_dissolved.set_index('norm_street_name_street')['geometry']
    gdf_final['matched_street_geom'] = gdf_final['norm_street_name_point'].map(street_geom_map)
    
    unmatched_mask = gdf_final['matched_street_geom'].isna()
    num_unmatched = unmatched_mask.sum()
    
    # 1. 在 f-string 外部先计算好所有数值
    total_points = len(gdf_final)
    matched_points = total_points - num_unmatched
    match_rate = (matched_points / total_points) * 100 if total_points > 0 else 0

    print(f"    - 第1轮完成: {matched_points} / {total_points} ({match_rate:.2f}%) 的点通过名称成功匹配。")
    
    # 第二轮: 空间最近邻匹配
    if num_unmatched > 0:
        print(f"    - 第2轮: 正在为剩余的 {num_unmatched} 个点手动查找空间上最近的街道段...")
        unmatched_gdf = gdf_final[unmatched_mask]
        
        streets_sindex = streets_gdf.sindex
        def find_nearest_segment(point):
            possible_matches_index = list(streets_sindex.nearest(point, return_all=False))[1]
            possible_matches = streets_gdf.iloc[possible_matches_index]
            nearest_geom = min(possible_matches.geometry, key=lambda geom: point.distance(geom))
            return nearest_geom
        
        nearest_geoms = unmatched_gdf.geometry.apply(find_nearest_segment)
        
        gdf_final.loc[unmatched_mask, 'matched_street_geom'] = nearest_geoms

    print("    - 街道匹配完成。")

    # --- 第二部分: 匹配面状单元 ---
    print("\n  (2/2) 正在匹配面状单元 (区域误差)...")

    for scale_name, scale_gdf in areal_layers.items():
        print(f"    - 正在匹配尺度: {scale_name}...")
        new_geom_col = f"{scale_name.lower()}_geom"
        
        spatial_index = scale_gdf.sindex

        def find_containing_polygon(point):
            possible_matches_index = list(spatial_index.intersection(point.bounds))
            if not possible_matches_index: return None
            possible_matches = scale_gdf.iloc[possible_matches_index]

            precise_matches = possible_matches[possible_matches.contains(point)]
            if not precise_matches.empty:
                return precise_matches.iloc[0].geometry
            return None
        
        tqdm.pandas(desc=f"      匹配 {scale_name}")

        gdf_final[new_geom_col] = gdf_final.geometry.progress_apply(find_containing_polygon)

    columns_to_drop = ['norm_street_name_point']
    gdf_final.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    
    print("\n--- 全面预匹配完成！ ---")
    return gdf_final

# --- 误差模拟函数 ---
def generate_random_point_in_polygon(polygon):
    min_x, min_y, max_x, max_y = polygon.bounds
    while True:
        random_point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if polygon.contains(random_point):
            return random_point


def generate_random_point_in_multipolygon(multi_polygon):
    polygons = list(multi_polygon.geoms)
    areas = [p.area for p in polygons]
    if sum(areas) > 0:
        weights = [area / sum(areas) for area in areas]
        chosen_polygon = random.choices(polygons, weights=weights, k=1)[0]
    else:
        chosen_polygon = random.choice(polygons)
    return generate_random_point_in_polygon(chosen_polygon)


def simulate_spatial_error(source_gdf, indices_to_pollute, pattern, context, context_geom_col=None):
    
    # Deletion 是一个简单的操作，单独处理
    if pattern == 'Deletion':
        return source_gdf.drop(indices_to_pollute)
        
    # --- 对于空间偏移，我们采用一个更稳健的更新模式 ---
    
    # 1. 创建一个 'new_geometry' 列，初始时与原始 'geometry' 完全相同
    gdf = source_gdf.copy()
    gdf['new_geometry'] = gdf['geometry']

    # 2. 只遍历需要被处理的行的索引
    indices_in_gdf = [idx for idx in indices_to_pollute if idx in gdf.index]

    if context == 'Street':
        # --- 沿线误差 (Linear) 策略 ---
        for idx in indices_in_gdf:
            original_point = gdf.loc[idx, 'geometry']
            street_geom = gdf.loc[idx, 'matched_street_geom']
            if not (street_geom and not street_geom.is_empty): continue
            target_geom = street_geom
            if street_geom.geom_type == 'MultiLineString':
                min_dist = float('inf'); nearest_segment = None
                for segment in street_geom.geoms:
                    dist = original_point.distance(segment)
                    if dist < min_dist: min_dist = dist; nearest_segment = segment
                target_geom = nearest_segment
            if target_geom and not target_geom.is_empty:
                if pattern == 'Center':
                    new_point = target_geom.interpolate(0.5, normalized=True)
                elif pattern == 'Random':
                    new_point = target_geom.interpolate(random.uniform(0, 1), normalized=True)
                gdf.loc[idx, 'new_geometry'] = new_point
    else:
        # --- 区域误差 (Areal) 策略 ---
        if not context_geom_col:
            raise ValueError("区域误差模拟需要 context_geom_col 参数。")
        for idx in indices_in_gdf:
            area_geom = gdf.loc[idx, context_geom_col]
            if not isinstance(area_geom, BaseGeometry) or area_geom.is_empty: continue
            new_point = None
            if pattern == 'Center':
                new_point = area_geom.centroid
            elif pattern == 'Random':
                if area_geom.geom_type == 'Polygon':
                    new_point = generate_random_point_in_polygon(area_geom)
                elif area_geom.geom_type == 'MultiPolygon':
                    new_point = generate_random_point_in_multipolygon(area_geom)
            if new_point:
                gdf.loc[idx, 'new_geometry'] = new_point
    
    gdf = gdf.drop(columns=['geometry'])
    gdf = gdf.rename(columns={'new_geometry': 'geometry'})
    gdf = gdf.set_geometry('geometry')

    original_count = len(gdf)
    gdf.dropna(subset=['geometry'], inplace=True)
    gdf = gdf[~gdf.geometry.is_empty]
    if len(gdf) < original_count:
        print(f"  警告: 模拟后移除了 {original_count - len(gdf)} 个无效/空几何。")
        
    return gdf

# --- 核密度估计函数 ---
def run_kde(input_shp, output_tif, study_area, cell_size, bandwidth):
    """执行统一参数的核密度估计，增加详细的日志和路径检查。"""
    try:
        arcpy.AddMessage(f"  -> 开始执行 KDE for: {os.path.basename(input_shp)}")
        arcpy.AddMessage(f"     输出至: {output_tif}")

        # 确保输出目录存在
        output_dir = os.path.dirname(output_tif)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with arcpy.EnvManager(
            extent=study_area, 
            outputCoordinateSystem=arcpy.Describe(input_shp).spatialReference, 
            overwriteOutput=True,
            scratchWorkspace=arcpy.env.scratchWorkspace
        ):
            if not arcpy.Exists(input_shp) or int(arcpy.GetCount_management(input_shp).getOutput(0)) == 0:
                arcpy.AddWarning(f"  KDE输入文件不存在或为空: {input_shp}")
                return None
            
            density_raster = KernelDensity(
                in_features=input_shp,
                population_field="NONE",
                cell_size=cell_size,
                search_radius=bandwidth,
                area_unit_scale_factor="SQUARE_KILOMETERS",
                out_cell_values="DENSITIES",
                method="PLANAR"
            )
            
            max_val_result = arcpy.GetRasterProperties_management(density_raster, "MAXIMUM")
            max_val = float(max_val_result.getOutput(0)) if max_val_result and max_val_result.getOutput(0) is not None else 0
            
            if max_val > 0:
                normalized_raster = (density_raster / max_val) * 100
            else: 
                normalized_raster = density_raster
            
            clipped_raster = ExtractByMask(normalized_raster, study_area)
            clipped_raster.save(output_tif)
            
            if arcpy.Exists(output_tif):
                arcpy.AddMessage(f"  [SUCCESS] KDE 结果已成功保存。")
                return output_tif
            else:
                arcpy.AddError("  [FAILURE] KDE .save() 命令执行后，文件仍不存在！")
                return None
            
    except Exception as e:
        arcpy.AddError(f"  KDE 过程发生严重错误 for {os.path.basename(input_shp)}.")

        if arcpy.GetMessages(2):
            arcpy_messages = arcpy.GetMessages(2).replace("\n", " ").replace("\r", " ")
            arcpy.AddError(f"    ArcPy Error Messages: {arcpy_messages}")
        return None

# --- 生成热点图函数 ---
def create_binary_hotspot(density_tif, output_hotspot_tif, study_area):
    """根据阈值创建二值热点图，增加详细的日志和路径检查。"""
    try:
        arcpy.AddMessage(f"  -> 开始创建二值热点图 for: {os.path.basename(density_tif)}")
        arcpy.AddMessage(f"     输出至: {output_hotspot_tif}")

        output_dir = os.path.dirname(output_hotspot_tif)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with arcpy.EnvManager(
            extent=study_area,
            snapRaster=density_tif,
            overwriteOutput=True,
            scratchWorkspace=arcpy.env.scratchWorkspace
        ):
            if not arcpy.Exists(density_tif):
                arcpy.AddWarning(f"  创建二值图失败: 输入密度图 '{density_tif}' 不存在。")
                return None
            
            density_raster = Raster(density_tif)
            
            try:
                mean_val_result = arcpy.GetRasterProperties_management(density_raster, "MEAN")
                std_val_result = arcpy.GetRasterProperties_management(density_raster, "STD")
                
                if not (mean_val_result and std_val_result and mean_val_result.getOutput(0) is not None and std_val_result.getOutput(0) is not None):
                    arcpy.AddWarning(f"  无法计算栅格统计数据 for {os.path.basename(density_tif)} (可能为空)，跳过。")
                    return None
                
                mean_val = float(mean_val_result.getOutput(0))
                std_val = float(std_val_result.getOutput(0))
            except Exception as prop_e:
                arcpy.AddWarning(f"  计算栅格统计数据时出错: {prop_e}")
                return None
                
            threshold = mean_val + 2 * std_val
            
            hotspot_raster_full = Con(density_raster >= threshold, 1, 0)
            hotspot_raster_clipped = ExtractByMask(hotspot_raster_full, study_area)
            hotspot_raster_clipped.save(output_hotspot_tif)

            if arcpy.Exists(output_hotspot_tif):
                arcpy.AddMessage(f"  [SUCCESS] 二值热点图已成功保存。")
                return output_hotspot_tif
            else:
                arcpy.AddError("  [FAILURE] Hotspot .save() 命令执行后，文件仍不存在！")
                return None

    except Exception as e:
        arcpy.AddError(f"  创建二值热点图时发生未知错误 for {os.path.basename(density_tif)}.")
        if arcpy.GetMessages(2):
            arcpy_messages = arcpy.GetMessages(2).replace("\n", " ").replace("\r", " ")
            arcpy.AddError(f"    ArcPy Error Messages: {arcpy_messages}")
        return None

# --- 计算PAI函数 ---
def calculate_pai_metrics(hotspot_tif, observed_points_gdf, study_area_sq_km):
    """计算PAI及相关指标。"""
    try:
        with rasterio.open(hotspot_tif) as src:
            raster_crs = src.crs; cell_area_sq_meters = src.res[0] * src.res[1]
            points_reprojected = observed_points_gdf.to_crs(raster_crs) if observed_points_gdf.crs != raster_crs else observed_points_gdf
            coords = [(p.x, p.y) for p in points_reprojected.geometry]
            values = [val[0] for val in src.sample(coords)]; hit_count = sum(1 for val in values if val == 1)
            hotspot_pixels = np.sum(src.read(1) == 1)
            predicted_area = (hotspot_pixels * cell_area_sq_meters) / 1_000_000
            observed_count = len(observed_points_gdf)
            hit_rate = (hit_count / observed_count) * 100 if observed_count > 0 else 0
            area_percentage = (predicted_area / study_area_sq_km) * 100 if study_area_sq_km > 0 else 0
            pai = hit_rate / area_percentage if area_percentage > 0 else 0
            return {'Hit_Count': hit_count, 'Total_Points': observed_count, 'Predicted_Area_km2': predicted_area, 'Hit_Rate_Percent': hit_rate, 'Area_Percent': area_percentage, 'PAI': pai}
    except Exception as e: 
        return {}

# --- 计算F1函数 ---
def calculate_f1_metrics(error_hotspot_tif, reference_hotspot_tif):
    """计算空间一致性指标。"""
    try:
        with arcpy.EnvManager(extent=arcpy.Describe(reference_hotspot_tif).extent, overwriteOutput=True):
            error_arr = arcpy.RasterToNumPyArray(error_hotspot_tif, nodata_to_value=0);
            ref_arr = arcpy.RasterToNumPyArray(reference_hotspot_tif, nodata_to_value=0)
            min_rows = min(error_arr.shape[0], ref_arr.shape[0]);
            min_cols = min(error_arr.shape[1], ref_arr.shape[1])
            error_arr = (error_arr[:min_rows, :min_cols] > 0).astype(np.int8);
            ref_arr = (ref_arr[:min_rows, :min_cols] > 0).astype(np.int8)
            TP = np.sum((error_arr == 1) & (ref_arr == 1));
            FP = np.sum((error_arr == 1) & (ref_arr == 0))
            TN = np.sum((error_arr == 0) & (ref_arr == 0));
            FN = np.sum((error_arr == 0) & (ref_arr == 1))
            accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0;
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            return {'TP': int(TP), 'FP': int(FP), 'TN': int(TN), 'FN': int(FN), 'Accuracy': accuracy,
                    'Precision': precision, 'Recall': recall, 'F1_Score': f1_score}
    except Exception as e:
        return {}


def main():
    parser = argparse.ArgumentParser(description='为所有实验情景生成热点图样本，并同时计算PAI和F1指标。')
    # --- 输入参数 (与主分析脚本几乎完全一致) ---
    parser.add_argument('--group_dir', required=True)
    parser.add_argument('--study_area', required=True)
    parser.add_argument('--streets_shp', required=True)
    parser.add_argument('--tract_shp', required=True)
    parser.add_argument('--neighborhood_shp', required=True)
    parser.add_argument('--grid_100_shp', required=True)
    parser.add_argument('--grid_300_shp', required=True)
    parser.add_argument('--grid_500_shp', required=True)
    parser.add_argument('--point_street_col', required=True)
    parser.add_argument('--street_layer_col', required=True)
    parser.add_argument('--cell_size', type=float, default=50)
    parser.add_argument('--bandwidth', type=float, default=300)
    args = parser.parse_args()

    # --- 1. 初始化和加载数据 ---
    print(f"--- 开始为实验组 {os.path.basename(args.group_dir)} 生成所有误差占比的热点图样本 ---")
    training_shp = os.path.join(args.group_dir, 'training_data.shp')
    validation_shp = os.path.join(args.group_dir, 'validation_data.shp')
    indices_dir = os.path.join(args.group_dir, 'Sampling_Indices')
    
    # 定义总的输出根目录
    output_root_dir = os.path.join(args.group_dir, f"Visualization_and_Metrics_Results_{args.bandwidth}")
    temp_dir = os.path.join(output_root_dir, 'temp')
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    arcpy.env.scratchWorkspace = temp_dir; arcpy.env.overwriteOutput = True
    
    print("正在加载所有数据...");
    training_gdf = gpd.read_file(training_shp)
    validation_gdf = gpd.read_file(validation_shp)
    streets_gdf = gpd.read_file(args.streets_shp)
    areal_layers = { 
        'Tract': gpd.read_file(args.tract_shp), 'Neighborhood': gpd.read_file(args.neighborhood_shp),
        'Grid_100': gpd.read_file(args.grid_100_shp), 'Grid_300': gpd.read_file(args.grid_300_shp), 'Grid_500': gpd.read_file(args.grid_500_shp)
    }
    study_area_sq_km = sum([f[0] for f in arcpy.da.SearchCursor(args.study_area, ["SHAPE@AREA"])]) / 1_000_000

    # --- 2. 统一预处理 ---
    training_gdf_matched = pre_match_points_to_streets_and_areals(
        training_gdf, streets_gdf, areal_layers, args.point_street_col, args.street_layer_col
    )
    
    print("\n--- 预匹配结果统计 ---")
    print(f"  总点数: {len(training_gdf_matched)}")
    
    # 检查街道匹配
    street_match_count = training_gdf_matched['matched_street_geom'].notna().sum()
    print(f"  - 街道 (Street) 匹配数: {street_match_count} / {len(training_gdf_matched)}")
    
    # 检查所有区域尺度的匹配
    areal_scales_to_check = ['Tract', 'Neighborhood', 'Grid_100', 'Grid_300', 'Grid_500']
    for scale in areal_scales_to_check:
        geom_col = f"{scale.lower()}_geom"
        if geom_col in training_gdf_matched.columns:
            scale_match_count = training_gdf_matched[geom_col].notna().sum()
            print(f"  - {scale} 尺度匹配数: {scale_match_count} / {len(training_gdf_matched)}")
        else:
            print(f"  - 警告: 未在预匹配结果中找到列 '{geom_col}'")

    # --- 3. 生成基准热点图 (用于对比) ---
    print("\n正在生成基准(无误差)的密度图和热点图...")
    base_hotspot_dir = os.path.join(output_root_dir, "Baseline")
    os.makedirs(base_hotspot_dir, exist_ok=True)
    base_density_tif = os.path.join(base_hotspot_dir, 'baseline_density.tif')
    base_hotspot_tif = os.path.join(base_hotspot_dir, 'baseline_hotspot.tif')
    if run_kde(training_shp, base_density_tif, args.study_area, args.cell_size, args.bandwidth):
        create_binary_hotspot(base_density_tif, base_hotspot_tif, args.study_area)
    reference_hotspot_tif = base_hotspot_tif

    # --- 4. 核心循环 (MODIFIED) ---
    all_results = [] 
    percentages = range(5, 51, 5)
    num_trials = 100
    spatial_contexts = ['Street', 'Tract', 'Neighborhood', 'Grid_100', 'Grid_300', 'Grid_500']
    offset_patterns = ['Center', 'Random']
    total_iterations = len(percentages) * (1 + len(spatial_contexts) * len(offset_patterns))
    pbar = tqdm(total=total_iterations, desc="生成样本并计算指标")
    aux_geom_cols = ['matched_street_geom'] + [f"{s.lower()}_geom" for s in areal_layers.keys()] 

    for p in percentages:

        for trial in range(1, num_trials + 1):
        
        # 读取当前误差占比和当前 trial 对应的抽样索引
            indices_filename = f"p{str(p).zfill(2)}_t{str(trial).zfill(3)}_indices.txt"
            indices_path = os.path.join(indices_dir, indices_filename)
            try:
                with open(indices_path, 'r') as f:
                    indices_to_pollute = [int(line.strip()) for line in f]
            except FileNotFoundError:
                pbar.write(f"警告: 未找到 {p}% 的索引文件，跳过。")
                pbar.update(1 + len(spatial_contexts) * len(offset_patterns))
                continue

            # --- A. Deletion ---
            pbar.set_description("Processing: Deletion @ {p}%")
            sim_gdf_del = training_gdf_matched.drop(indices_to_pollute)
            output_subdir = os.path.join(output_root_dir, "Deletion")
            os.makedirs(output_subdir, exist_ok=True)
            temp_sim_shp = os.path.join(temp_dir, "temp_sim.shp")
            sim_gdf_del[[col for col in sim_gdf_del.columns if col not in aux_geom_cols]].to_file(temp_sim_shp, encoding='utf-8')
            density_tif = os.path.join(output_subdir, f"Deletion_{p}pct_t{trial}_density.tif")
            hotspot_tif = os.path.join(output_subdir, f"Deletion_{p}pct_t{trial}_hotspot.tif")
            if run_kde(temp_sim_shp, density_tif, args.study_area, args.cell_size, args.bandwidth):
                 hotspot_result = create_binary_hotspot(density_tif, hotspot_tif, args.study_area)
                 if hotspot_result:
                     pai_results = calculate_pai_metrics(hotspot_result, validation_gdf, study_area_sq_km)
                     f1_results = calculate_f1_metrics(hotspot_result, reference_hotspot_tif)
                     all_results.append({'Experiment_Type': 'N/A', 'Scale': 'N/A', 'Type': 'Deletion', 'Error_Percentage': p, 'Trial': trial, **pai_results, **f1_results})
            pbar.update(1)
    
            # --- B. 空间偏移 ---
            for context in spatial_contexts:
                for pattern in offset_patterns:
                    pbar.set_description(f"Processing: {context}/{pattern} @ {p}%")
                
                    geom_col = 'matched_street_geom' if context == 'Street' else f"{context.lower()}_geom"
                    current_data = training_gdf_matched.dropna(subset=[geom_col]).copy()
                    valid_indices = list(set(indices_to_pollute) & set(current_data.index))

                    sim_gdf = simulate_spatial_error(current_data, valid_indices, pattern, context, context_geom_col=geom_col)
             
                    output_subdir = os.path.join(output_root_dir, f"{p}pct_Error", "Trial_{:03d}".format(trial), context, pattern)
                    os.makedirs(output_subdir, exist_ok=True)

                    temp_sim_shp = os.path.join(temp_dir, "temp_sim.shp")
                    sim_gdf[[col for col in sim_gdf.columns if col not in aux_geom_cols]].to_file(temp_sim_shp, encoding='utf-8')
             
                    filename_base = f"{context}_{pattern}_{p}pct"
                    density_tif = os.path.join(output_subdir, f"{filename_base}_density.tif")
                    hotspot_tif = os.path.join(output_subdir, f"{filename_base}_hotspot.tif")
            
                    if run_kde(temp_sim_shp, density_tif, args.study_area, args.cell_size, args.bandwidth):
                        hotspot_result = create_binary_hotspot(density_tif, hotspot_tif, args.study_area)
                        if hotspot_result:
                            pai_results = calculate_pai_metrics(hotspot_result, validation_gdf, study_area_sq_km)
                            f1_results = calculate_f1_metrics(hotspot_result, reference_hotspot_tif)
                            all_results.append({'Experiment_Type': 'Linear' if context == 'Street' else 'Areal','Scale': context, 'Type': pattern, 'Error_Percentage': p, 'Trial': trial, **pai_results, **f1_results})
                    pbar.update(1)
               
    pbar.close()
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    if all_results:
        print("\n所有指标计算完毕，正在生成汇总CSV...")
        final_df = pd.DataFrame(all_results)
        
        output_csv_path = os.path.join(output_root_dir, f"all_scenarios_{os.path.basename(args.group_dir)}_metrics.csv")
        final_df.to_csv(output_csv_path, index=False, float_format='%.4f')
        print(f"\n指标汇总表已保存至: {output_csv_path}")

    print(f"\n--- 所有热点图样本和指标已生成完毕，请查看根目录: {output_root_dir} ---")

if __name__ == "__main__":
    main()