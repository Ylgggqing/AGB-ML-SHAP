# -*- coding: utf-8 -*-

# 核心库
import sys, os, glob, numpy as np, pandas as pd, time
# 多进程与队列，用于隔离计算任务
import multiprocessing
from queue import Empty

# PyQt5
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QGridLayout, QGroupBox, QPushButton,
                             QLineEdit, QLabel, QFileDialog, QProgressBar, QTextEdit,
                             QMessageBox, QDoubleSpinBox, QSpinBox)
from PyQt5.QtCore import QThread, pyqtSignal

# 依赖库
try:
    import laspy
    import rasterio
    from rasterio.transform import from_origin
    import scipy.spatial as spatial
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    import geopandas as gpd
    from shapely.geometry import Polygon, Point
    from rasterio.features import shapes

    GEOSPATIAL_LIBS_AVAILABLE = True
except ImportError as e:
    print(f"Lidar/Raster工具依赖库未完全安装: {e}")
    GEOSPATIAL_LIBS_AVAILABLE = False


# ==============================================================================
#  在独立进程中运行的目标函数
# ==============================================================================

def _idw_interpolation_proc(points, values, grid_x, grid_y, weight, k):
    """这个函数在子进程中运行，可以安全地使用多核并行"""
    if len(points) == 0 or len(values) == 0: return np.full(grid_x.shape, np.nan)
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    tree = spatial.KDTree(points)
    distances, indices = tree.query(grid_points, k=k, workers=-1)
    if distances.ndim == 1:
        distances = np.expand_dims(distances, axis=1)
        indices = np.expand_dims(indices, axis=1)
    distances = np.maximum(distances, 1e-12)
    weights = 1.0 / distances ** weight
    weights_sum = np.sum(weights, axis=1)
    valid_weights = weights_sum > 0
    interpolated_values = np.full(grid_points.shape[0], np.nan)
    neighbor_values = values[indices[valid_weights]]
    interpolated_values[valid_weights] = np.sum(weights[valid_weights] * neighbor_values, axis=1) / weights_sum[
        valid_weights]
    return interpolated_values.reshape(grid_x.shape)


def dsm_dem_chm_process_func(queue, params):
    """
    这个函数是真正执行DEM/DSM/CHM计算的地方，它将运行在一个完全独立的进程中,
    通过队列(queue)与主GUI线程通信。
    """
    try:
        p = params
        queue.put(('progress', 5, f"正在读取LAS文件: {os.path.basename(p['las_path'])}..."))
        las = laspy.read(p['las_path'])
        x, y, z = las.x, las.y, las.z

        try:
            las_crs = las.header.parse_crs()
            if not las_crs: queue.put(('progress', 8, "警告: 无法从LAS文件读取CRS。"))
        except Exception:
            las_crs = None

        queue.put(('progress', 10, "创建插值网格..."))
        min_x, max_x, min_y, max_y = np.min(x), np.max(x), np.min(y), np.max(y)
        extent = (min_x - p['buffer'], max_x + p['buffer'], min_y - p['buffer'], max_y + p['buffer'])
        grid_x, grid_y = np.arange(extent[0], extent[1], p['resolution']), np.arange(extent[2], extent[3],
                                                                                     p['resolution'])
        grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

        queue.put(('progress', 15, "正在插值生成DEM (此步骤可能需要较长时间)..."))
        ground_indices = np.where(las.classification == 2)[0]
        if len(ground_indices) < p['k_neighbors']:
            queue.put(('error', f"LAS文件中地面点({len(ground_indices)})过少，少于K值({p['k_neighbors']})。无法生成DEM。"))
            return
        dem_values = _idw_interpolation_proc(np.vstack([x[ground_indices], y[ground_indices]]).T, z[ground_indices],
                                             grid_xx, grid_yy, p['idw_weight'], p['k_neighbors'])
        queue.put(('progress', 40, "DEM插值完成。"))

        queue.put(('progress', 45, "正在插值生成DSM (此步骤可能需要较长时间)..."))
        dsm_values = _idw_interpolation_proc(np.vstack([x, y]).T, z, grid_xx, grid_yy, p['idw_weight'],
                                             p['k_neighbors'])
        queue.put(('progress', 70, "DSM插值完成。"))

        if p['max_dist'] > 0:
            queue.put(('progress', 75, "正在创建距离掩膜..."))
            tree = spatial.KDTree(np.vstack([x, y]).T)
            distance, _ = tree.query(np.vstack([grid_xx.ravel(), grid_yy.ravel()]).T, k=1, workers=-1)
            mask = (distance > p['max_dist']).reshape(grid_xx.shape)
            dem_values[mask] = np.nan;
            dsm_values[mask] = np.nan
            queue.put(('progress', 80, "距离掩膜应用完成。"))
        else:
            queue.put(('progress', 80, "跳过距离掩膜步骤。"))

        queue.put(('progress', 85, "正在保存DSM, DEM, 和 CHM 文件..."))
        transform = from_origin(extent[0], extent[3], p['resolution'], p['resolution']);
        nodata_value = -9999.0
        common_profile = {'driver': 'GTiff', 'height': dsm_values.shape[0], 'width': dsm_values.shape[1], 'count': 1,
                          'crs': las_crs, 'transform': transform, 'nodata': nodata_value}

        dsm_to_save = np.flipud(dsm_values);
        dsm_to_save[np.isnan(dsm_to_save)] = nodata_value
        with rasterio.open(p['dsm_path'], 'w', dtype=dsm_to_save.dtype, **common_profile) as dst:
            dst.write(dsm_to_save, 1)
        queue.put(('progress', 90, "DSM已保存。"))

        dem_to_save = np.flipud(dem_values);
        dem_to_save[np.isnan(dem_to_save)] = nodata_value
        with rasterio.open(p['dem_path'], 'w', dtype=dem_to_save.dtype, **common_profile) as dst:
            dst.write(dem_to_save, 1)
        queue.put(('progress', 95, "DEM已保存。"))

        chm_values = dsm_values - dem_values
        chm_to_save = np.flipud(chm_values);
        chm_to_save[np.isnan(chm_to_save)] = nodata_value
        with rasterio.open(p['chm_path'], 'w', dtype=chm_to_save.dtype, **common_profile) as dst:
            dst.write(chm_to_save, 1)

        queue.put(('progress', 100, "CHM已保存，所有文件生成完毕。"))
        queue.put(('finished', "DEM, DSM, 和 CHM 生成成功！"))
    except Exception as e:
        queue.put(('error', f"处理失败: {e}"))


# ==============================================================================
#  工作线程
# ==============================================================================
class ProcessWorker(QThread):
    """一个特殊的QThread，用于管理一个独立的multiprocessing.Process和消息队列"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, target_func, params):
        super().__init__()
        self.target_func = target_func
        self.params = params
        self.process = None
        self.queue = None

    def run(self):
        try:
            self.queue = multiprocessing.Queue()
            self.process = multiprocessing.Process(target=self.target_func, args=(self.queue, self.params))
            self.process.start()

            while self.process.is_alive():
                try:
                    message = self.queue.get(timeout=0.1)
                    msg_type, content, *args = message

                    if msg_type == 'progress':
                        self.progress.emit(content, args[0])
                    elif msg_type == 'finished':
                        self.finished.emit(content); break
                    elif msg_type == 'error':
                        self.error.emit(content); break
                except Empty:
                    continue

            self.process.join()
        except Exception as e:
            self.error.emit(f"启动进程时发生错误: {e}")

    def stop(self):
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join()


class GenericWorker(QThread):
    """用于不涉及内部并行的简单任务的通用QThread Worker"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, target_func, *args):
        super().__init__()
        self.target_func = target_func
        self.args = args

    def run(self):
        try:
            # 将信号作为参数传递给目标函数
            self.target_func(self.progress, self.finished, self.error, *self.args)
        except Exception as e:
            self.error.emit(f"任务执行失败: {e}")


# ==============================================================================
#  简单任务的目标函数 (用于 GenericWorker)
# ==============================================================================

def segmentation_func(progress, finished, error, params):
    """分水岭分割的目标函数，兼容旧版 scikit-image"""
    try:
        p = params
        progress.emit(10, f"正在读取CHM文件: {os.path.basename(p['chm_path'])}...")
        with rasterio.open(p['chm_path']) as src:
            chm, transform, crs = src.read(1), src.transform, src.crs

        progress.emit(20, "高斯滤波平滑CHM...");
        chm_smooth = ndi.gaussian_filter(chm, sigma=p['sigma'])

        progress.emit(40, "寻找树冠顶点 (局部最大值)...")
        # 1. 获取顶点坐标列表 (所有版本都支持)
        local_maxi = peak_local_max(chm_smooth, min_distance=p['min_dist'], threshold_abs=p['min_height'])

        # 2. 【核心修复】根据坐标列表手动创建布尔掩码，以替代 'indices=False'
        markers_mask = np.zeros_like(chm, dtype=bool)
        markers_mask[tuple(local_maxi.T)] = True

        # 3. 对布尔掩码进行标记
        markers = ndi.label(markers_mask)[0]

        progress.emit(60, "执行分水岭分割...")
        labels = watershed(-chm_smooth, markers, mask=chm >= p['min_height'])

        progress.emit(75, "正在提取树冠边界并转换为矢量...")
        shapes_gen = shapes(labels.astype(np.int16), mask=(labels > 0), transform=transform)
        gdf_crowns = gpd.GeoDataFrame({'geometry': [Polygon(s['coordinates'][0]) for s, v in shapes_gen if v != 0]},
                                      crs=crs)
        if not gdf_crowns.empty:
            gdf_crowns['crown_id'] = range(1, len(gdf_crowns) + 1)
            gdf_crowns.to_file(p['shp_path'], driver='ESRI Shapefile', encoding='utf-8')

        progress.emit(90, "正在提取树顶点坐标...")
        tree_tops_data = []
        if len(local_maxi) > 0:
            for i, row in enumerate(local_maxi):
                x_coord, y_coord = rasterio.transform.xy(transform, row[0], row[1])
                tree_tops_data.append({'tree_id': i + 1, 'X': x_coord, 'Y': y_coord, 'Z_height': chm[row[0], row[1]],
                                       'geometry': Point(x_coord, y_coord)})
            gdf_tops = gpd.GeoDataFrame(tree_tops_data, crs=crs)
            gdf_tops.to_file(p['tops_path'], driver='ESRI Shapefile', encoding='utf-8')

        progress.emit(100, "分割完成。")
        finished.emit("分水岭分割成功完成！")
    except Exception as e:
        error.emit(f"分割失败: {e}")


def las_to_csv_func(progress, finished, error, in_dir, out_dir):
    try:
        las_files = glob.glob(os.path.join(in_dir, '*.las')) + glob.glob(os.path.join(in_dir, '*.laz'))
        if not las_files: error.emit("输入目录中未找到.las或.laz文件。"); return
        total = len(las_files)
        for i, f in enumerate(las_files):
            progress.emit(int((i + 1) / total * 100), f"正在转换 ({i + 1}/{total}): {os.path.basename(f)}")
            las = laspy.read(f)
            df = pd.DataFrame(np.vstack((las.x, las.y, las.z)).transpose(), columns=['X', 'Y', 'Z'])
            df.to_csv(os.path.join(out_dir, os.path.splitext(os.path.basename(f))[0] + '.csv'), index=False)
        finished.emit("批量LAS到CSV转换完成！")
    except Exception as e:
        error.emit(f"转换失败: {e}")


def csv_sampling_func(progress, finished, error, in_dir, out_dir, sample_ratio):
    try:
        csv_files = glob.glob(os.path.join(in_dir, '*.csv'))
        if not csv_files: error.emit("输入目录中未找到.csv文件。"); return
        total = len(csv_files)
        for i, f in enumerate(csv_files):
            progress.emit(int((i + 1) / total * 100), f"正在采样 ({i + 1}/{total}): {os.path.basename(f)}")
            pd.read_csv(f).sample(frac=sample_ratio).to_csv(
                os.path.join(out_dir, os.path.splitext(os.path.basename(f))[0] + '_sampled.csv'), index=False)
        finished.emit("批量CSV采样完成！")
    except Exception as e:
        error.emit(f"采样失败: {e}")


# ==============================================================================
#  主界面类
# ==============================================================================
class LidarToolsTab(QWidget):
    def __init__(self):
        super().__init__();
        self.thread = None
        if not GEOSPATIAL_LIBS_AVAILABLE:
            main_layout = QVBoxLayout(self);
            main_layout.addWidget(QLabel(
                "错误：缺少必要的地理空间处理库。\n请安装 laspy, rasterio, gdal, geopandas, shapely, scikit-image。"))
            return
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        # UI 定义... (与上一版本完全相同)
        dem_group = QGroupBox("1. 从LAS生成DEM, DSM, CHM (IDW插值法)");
        dem_layout = QGridLayout(dem_group)
        self.dem_las_path = QLineEdit(readOnly=True, placeholderText="选择.las或.laz文件");
        self.dem_output_dir = QLineEdit(readOnly=True, placeholderText="选择输出文件夹")
        self.dem_resolution = QDoubleSpinBox(value=0.5, minimum=0.1, maximum=10.0, singleStep=0.1, decimals=2);
        self.dem_buffer = QDoubleSpinBox(value=2.0, minimum=0.0, maximum=50.0, singleStep=1.0)
        self.dem_idw_weight = QDoubleSpinBox(value=2.0, minimum=0.5, maximum=5.0, singleStep=0.5);
        self.dem_k_neighbors = QSpinBox(value=8, minimum=1, maximum=20)
        self.dem_max_dist = QDoubleSpinBox(value=1.5, minimum=0.0, maximum=50.0, singleStep=0.5,
                                           toolTip="最大插值距离(米), 0表示不限制")
        dem_layout.addWidget(QLabel("输入LAS文件:"), 0, 0);
        dem_layout.addWidget(self.dem_las_path, 0, 1);
        dem_layout.addWidget(QPushButton("...", clicked=self.select_dem_las), 0, 2)
        dem_layout.addWidget(QLabel("输出文件夹:"), 1, 0);
        dem_layout.addWidget(self.dem_output_dir, 1, 1);
        dem_layout.addWidget(QPushButton("...", clicked=self.select_dem_output_dir), 1, 2)
        dem_layout.addWidget(QLabel("栅格分辨率 (米):"), 2, 0);
        dem_layout.addWidget(self.dem_resolution, 2, 1);
        dem_layout.addWidget(QLabel("边界缓冲区 (米):"), 3, 0);
        dem_layout.addWidget(self.dem_buffer, 3, 1)
        dem_layout.addWidget(QLabel("IDW 权重 (p):"), 4, 0);
        dem_layout.addWidget(self.dem_idw_weight, 4, 1);
        dem_layout.addWidget(QLabel("最近邻点数 (k):"), 5, 0);
        dem_layout.addWidget(self.dem_k_neighbors, 5, 1)
        dem_layout.addWidget(QLabel("最大插值距离 (米):"), 6, 0);
        dem_layout.addWidget(self.dem_max_dist, 6, 1)
        self.dem_run_btn = QPushButton("开始生成");
        dem_layout.addWidget(self.dem_run_btn, 7, 0, 1, 3);
        main_layout.addWidget(dem_group)
        seg_group = QGroupBox("2. CHM分水岭分割 (生成树冠边界和顶点)");
        seg_layout = QGridLayout(seg_group)
        self.seg_chm_path = QLineEdit(readOnly=True, placeholderText="选择CHM .tif文件");
        self.seg_output_prefix = QLineEdit("segmented")
        self.seg_min_height = QDoubleSpinBox(value=5.0, minimum=0, maximum=100.0, singleStep=1.0);
        self.seg_min_dist = QSpinBox(value=3, minimum=1, maximum=20);
        self.seg_sigma = QDoubleSpinBox(value=0.5, minimum=0.1, maximum=5.0, singleStep=0.1)
        seg_layout.addWidget(QLabel("输入CHM文件:"), 0, 0);
        seg_layout.addWidget(self.seg_chm_path, 0, 1);
        seg_layout.addWidget(QPushButton("...", clicked=self.select_seg_chm), 0, 2)
        seg_layout.addWidget(QLabel("输出文件名前缀:"), 1, 0);
        seg_layout.addWidget(self.seg_output_prefix, 1, 1, 1, 2);
        seg_layout.addWidget(QLabel("最小树高 (米):"), 2, 0);
        seg_layout.addWidget(self.seg_min_height, 2, 1)
        seg_layout.addWidget(QLabel("顶点最小距离(像素):"), 3, 0);
        seg_layout.addWidget(self.seg_min_dist, 3, 1);
        seg_layout.addWidget(QLabel("高斯滤波 Sigma:"), 4, 0);
        seg_layout.addWidget(self.seg_sigma, 4, 1)
        self.seg_run_btn = QPushButton("开始分割");
        seg_layout.addWidget(self.seg_run_btn, 5, 0, 1, 3);
        main_layout.addWidget(seg_group)
        las_csv_group = QGroupBox("3. LAS -> CSV 批量转换");
        las_csv_layout = QGridLayout(las_csv_group)
        self.las_in_dir = QLineEdit(readOnly=True, placeholderText="选择包含.las/.laz文件的输入文件夹");
        self.csv_out_dir = QLineEdit(readOnly=True, placeholderText="选择CSV文件输出文件夹")
        las_csv_layout.addWidget(QLabel("输入文件夹:"), 0, 0);
        las_csv_layout.addWidget(self.las_in_dir, 0, 1);
        las_csv_layout.addWidget(QPushButton("...", clicked=lambda: self.select_dir(self.las_in_dir)), 0, 2)
        las_csv_layout.addWidget(QLabel("输出文件夹:"), 1, 0);
        las_csv_layout.addWidget(self.csv_out_dir, 1, 1);
        las_csv_layout.addWidget(QPushButton("...", clicked=lambda: self.select_dir(self.csv_out_dir)), 1, 2)
        self.las_csv_run_btn = QPushButton("开始批量转换");
        las_csv_layout.addWidget(self.las_csv_run_btn, 2, 0, 1, 3);
        main_layout.addWidget(las_csv_group)
        csv_sample_group = QGroupBox("4. CSV点云批量随机采样");
        csv_sample_layout = QGridLayout(csv_sample_group)
        self.csv_in_dir = QLineEdit(readOnly=True, placeholderText="选择包含.csv文件的输入文件夹");
        self.sample_out_dir = QLineEdit(readOnly=True, placeholderText="选择采样后文件输出文件夹");
        self.sample_ratio = QDoubleSpinBox(value=0.5, minimum=0.01, maximum=0.99, singleStep=0.1, decimals=2)
        csv_sample_layout.addWidget(QLabel("输入文件夹:"), 0, 0);
        csv_sample_layout.addWidget(self.csv_in_dir, 0, 1);
        csv_sample_layout.addWidget(QPushButton("...", clicked=lambda: self.select_dir(self.csv_in_dir)), 0, 2)
        csv_sample_layout.addWidget(QLabel("输出文件夹:"), 1, 0);
        csv_sample_layout.addWidget(self.sample_out_dir, 1, 1);
        csv_sample_layout.addWidget(QPushButton("...", clicked=lambda: self.select_dir(self.sample_out_dir)), 1, 2)
        csv_sample_layout.addWidget(QLabel("采样比例:"), 2, 0);
        csv_sample_layout.addWidget(self.sample_ratio, 2, 1);
        self.csv_sample_run_btn = QPushButton("开始批量采样");
        csv_sample_layout.addWidget(self.csv_sample_run_btn, 3, 0, 1, 3);
        main_layout.addWidget(csv_sample_group)
        self.progress_bar = QProgressBar(value=0);
        self.log_box = QTextEdit(readOnly=True);
        main_layout.addWidget(self.progress_bar);
        main_layout.addWidget(self.log_box)
        self.dem_run_btn.clicked.connect(self.run_dem_dsm_chm);
        self.seg_run_btn.clicked.connect(self.run_segmentation);
        self.las_csv_run_btn.clicked.connect(self.run_las_to_csv);
        self.csv_sample_run_btn.clicked.connect(self.run_csv_sampling)

    def select_dir(self, line_edit):
        path = QFileDialog.getExistingDirectory(self, "选择文件夹"); path and line_edit.setText(path)

    def select_dem_las(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择LAS文件", "",
                                              "LAS Files (*.las *.laz)"); path and self.dem_las_path.setText(path)

    def select_dem_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出文件夹"); path and self.dem_output_dir.setText(path)

    def select_seg_chm(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择CHM文件", "",
                                              "GeoTIFF Files (*.tif *.tiff)"); path and self.seg_chm_path.setText(path)

    def start_task(self, worker_class, args, task_name):
        if self.thread and self.thread.isRunning(): QMessageBox.warning(self, "警告",
                                                                        "另一个任务正在运行，请稍候。"); return
        self.log_box.clear();
        self.progress_bar.setValue(0);
        self.log_box.append(f"开始 {task_name} 任务...")
        self.thread = worker_class(*args)
        self.thread.progress.connect(lambda p, m: (self.progress_bar.setValue(p), self.log_box.append(m)))
        self.thread.finished.connect(self.task_finished)
        self.thread.error.connect(self.task_error)
        self.thread.start()

    def run_dem_dsm_chm(self):
        in_path, out_dir = self.dem_las_path.text(), self.dem_output_dir.text()
        if not all([in_path, out_dir]): QMessageBox.warning(self, "输入错误", "请选择输入LAS文件和输出文件夹。"); return
        basename = os.path.splitext(os.path.basename(in_path))[0]
        params = {'las_path': in_path, 'resolution': self.dem_resolution.value(), 'buffer': self.dem_buffer.value(),
                  'idw_weight': self.dem_idw_weight.value(), 'k_neighbors': self.dem_k_neighbors.value(),
                  'max_dist': self.dem_max_dist.value(), 'dsm_path': os.path.join(out_dir, f"{basename}_dsm.tif"),
                  'dem_path': os.path.join(out_dir, f"{basename}_dem.tif"),
                  'chm_path': os.path.join(out_dir, f"{basename}_chm.tif")}
        self.start_task(ProcessWorker, (dsm_dem_chm_process_func, params), "DEM/DSM/CHM生成")

    def run_segmentation(self):
        in_path, prefix = self.seg_chm_path.text(), self.seg_output_prefix.text()
        if not all([in_path, prefix]): QMessageBox.warning(self, "输入错误", "请选择输入CHM文件并指定输出前缀。"); return
        out_dir = os.path.dirname(in_path)
        params = {'chm_path': in_path, 'min_height': self.seg_min_height.value(), 'min_dist': self.seg_min_dist.value(),
                  'sigma': self.seg_sigma.value(), 'shp_path': os.path.join(out_dir, f"{prefix}_crowns.shp"),
                  'tops_path': os.path.join(out_dir, f"{prefix}_tops.shp")}
        self.start_task(GenericWorker, (segmentation_func, params), "分水岭分割")

    def run_las_to_csv(self):
        in_dir, out_dir = self.las_in_dir.text(), self.csv_out_dir.text()
        if not all([in_dir, out_dir]): QMessageBox.warning(self, "输入错误", "请选择输入和输出文件夹。"); return
        self.start_task(GenericWorker, (las_to_csv_func, in_dir, out_dir), "LAS转CSV")

    def run_csv_sampling(self):
        in_dir, out_dir = self.csv_in_dir.text(), self.sample_out_dir.text()
        if not all([in_dir, out_dir]): QMessageBox.warning(self, "输入错误", "请选择输入和输出文件夹。"); return
        self.start_task(GenericWorker, (csv_sampling_func, in_dir, out_dir, self.sample_ratio.value()), "CSV采样")

    def task_finished(self, message):
        self.progress_bar.setValue(100); self.log_box.append(f"\n{message}"); QMessageBox.information(self, "成功",
                                                                                                      message); self.cleanup_thread()

    def task_error(self, message):
        self.log_box.append(f"\n错误: {message}"); QMessageBox.critical(self, "错误", message); self.cleanup_thread()

    def cleanup_thread(self):
        if self.thread:
            if isinstance(self.thread, ProcessWorker): self.thread.stop()
            self.thread.quit();
            self.thread.wait();
            self.thread = None