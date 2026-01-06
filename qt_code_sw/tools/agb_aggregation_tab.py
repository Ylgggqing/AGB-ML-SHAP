# -*- coding: utf-8 -*-

# 核心库
import sys
import os
import glob
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from shapely.geometry import Point, Polygon
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import matplotlib.pyplot as plt

# PyQt5 相关库
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QProgressBar, QTextEdit,
                             QLabel, QLineEdit, QFileDialog, QMessageBox, QWidget, QVBoxLayout,
                             QHBoxLayout, QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox,
                             QCheckBox, QFormLayout)
from PyQt5.QtCore import QThread, QObject, pyqtSignal


# ==============================================================================
#  后端逻辑的辅助函数 - 保持在模块顶层以便并行处理
# ==============================================================================

def _calculate_stratified_volume(points_df, slice_height=0.5):
    """(辅助函数) 计算点云体积"""
    if points_df.shape[0] < 4: return 0.0
    min_z, max_z = points_df['Z'].min(), points_df['Z'].max()
    total_volume = 0.0
    for z in np.arange(min_z, max_z, slice_height):
        slice_points = points_df[(points_df['Z'] >= z) & (points_df['Z'] < z + slice_height)]
        if slice_points.shape[0] >= 3:
            try:
                hull = ConvexHull(slice_points[['X', 'Y']].values)
                total_volume += hull.volume * slice_height
            except (QhullError, RuntimeError):
                pass
    return total_volume


def _process_single_plot(plot_data, tree_groups, agb_map_df, tree_hulls_dict, slice_height):
    """(辅助函数) 处理单个样地"""
    plot_geom = plot_data.geometry
    plot_agb, tree_count = 0.0, 0
    for tree_id, tree_hull in tree_hulls_dict.items():
        if tree_hull is None or not plot_geom.intersects(tree_hull): continue

        try:
            total_tree_agb = agb_map_df.loc[tree_id, 'AGB']
        except KeyError:
            continue

        tree_points = tree_groups[tree_id]
        if plot_geom.contains(tree_hull):
            tree_count += 1
            plot_agb += total_tree_agb
        else:
            is_inside = [plot_geom.contains(Point(xy)) for xy in zip(tree_points['X'], tree_points['Y'])]
            if sum(is_inside) == 0: continue
            tree_count += 1
            total_volume = _calculate_stratified_volume(tree_points, slice_height)
            if total_volume > 0:
                inside_volume = _calculate_stratified_volume(tree_points[np.array(is_inside)], slice_height)
                plot_agb += (inside_volume / total_volume) * total_tree_agb

    plot_result = plot_data.to_dict()
    plot_result['AGB'], plot_result['TreeCount'] = plot_agb, tree_count
    return plot_result


# ==============================================================================
#  后端工作线程 (AggregationWorker) - 无需修改
# ==============================================================================
class AggregationWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str, str)
    error = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.is_running = True

    def stop(self):
        self.is_running = False

    def run(self):
        try:
            self.progress.emit(5, "正在加载样地格网...");
            grid_gdf = gpd.read_file(self.params['grid_shp'])
            self.progress.emit(10, "正在加载AGB映射表...");
            agb_map_df = pd.read_csv(self.params['agb_csv']).set_index('TreeID')
            self.progress.emit(15, "正在加载所有单木点云 (可能需要一些时间)...")
            csv_files = glob.glob(os.path.join(self.params['points_folder'], '*.csv'))
            if not csv_files: self.error.emit("错误：点云文件夹中未找到CSV文件。"); return

            all_points_df = pd.concat(
                [pd.read_csv(f, usecols=[0, 1, 2, 3], names=['X', 'Y', 'Z', 'TreeID'], header=0) for f in csv_files])
            all_points_df['TreeID'] = all_points_df['TreeID'].astype(int)
            tree_groups = dict(tuple(all_points_df.groupby('TreeID')))
            if not self.is_running: self.error.emit("任务中止。"); return

            self.progress.emit(40, "正在为所有树预计算2D凸包...")
            tree_hulls_dict = {}
            for tree_id, points_df in tree_groups.items():
                if points_df.shape[0] >= 3:
                    try:
                        tree_hulls_dict[tree_id] = Polygon(
                            points_df[['X', 'Y']].values[ConvexHull(points_df[['X', 'Y']].values).vertices])
                    except (QhullError, RuntimeError):
                        tree_hulls_dict[tree_id] = None
                else:
                    tree_hulls_dict[tree_id] = None
            if not self.is_running: self.error.emit("任务中止。"); return

            if self.params['generate_plot']:
                self.progress.emit(50, "正在生成数据概览图...")
                tree_hulls_gdf = gpd.GeoDataFrame(pd.DataFrame({'TreeID': list(tree_hulls_dict.keys())}),
                                                  geometry=list(tree_hulls_dict.values()), crs=grid_gdf.crs).dropna(
                    subset=['geometry'])
                fig, ax = plt.subplots(1, 1, figsize=(20, 20))
                grid_gdf.plot(ax=ax, facecolor='none', edgecolor='black');
                tree_hulls_gdf.plot(ax=ax, facecolor='blue', edgecolor='none', alpha=0.4)
                ax.set_title('Overview of Tree Hulls and Sample Plots');
                ax.set_aspect('equal');
                plt.savefig(self.params['plot_path'], dpi=300);
                plt.close(fig)

            self.progress.emit(60, "正在进行并行聚合计算...")
            plots_data = [plot for _, plot in grid_gdf.iterrows()]
            with ProcessPoolExecutor(max_workers=self.params['num_workers']) as executor:
                results = list(executor.map(_process_single_plot, plots_data, [tree_groups] * len(plots_data),
                                            [agb_map_df] * len(plots_data), [tree_hulls_dict] * len(plots_data),
                                            [self.params['slice_height']] * len(plots_data)))
            if not self.is_running: self.error.emit("任务中止。"); return

            self.progress.emit(95, "正在整理结果并输出文件...")
            results_gdf = gpd.GeoDataFrame(results, geometry='geometry', crs=grid_gdf.crs)
            final_gdf = results_gdf[results_gdf['TreeCount'] >= self.params['min_trees']].copy()
            final_gdf.drop(columns=['TreeCount']).to_file(self.params['output_shp'], driver='ESRI Shapefile',
                                                          encoding='utf-8')
            pd.DataFrame(final_gdf.drop(columns='geometry')).to_csv(self.params['output_csv'], index=False,
                                                                    encoding='utf-8-sig')
            self.finished.emit(self.params['output_shp'], self.params['output_csv'])
        except Exception as e:
            self.error.emit(f"发生严重错误: {e}")


# ==============================================================================
#  【修改点】: 主界面类重构为 QWidget
# ==============================================================================
class AggregationTab(QWidget):  # 继承自 QWidget
    def __init__(self):
        super().__init__()
        # 【修改点】: 移除 setWindowTitle, setGeometry 和 setCentralWidget
        self.thread, self.worker = None, None

        # 【修改点】: 布局直接应用到 self (QWidget)
        main_layout = QVBoxLayout(self)

        # --- 输入设置 ---
        inputs_group = QGroupBox("输入文件设置")
        inputs_layout = QGridLayout(inputs_group)
        self.points_folder_edit = QLineEdit();
        self.grid_shp_edit = QLineEdit();
        self.agb_csv_edit = QLineEdit()
        inputs_layout.addWidget(QLabel("单木点云文件夹:"), 0, 0);
        inputs_layout.addWidget(self.points_folder_edit, 0, 1);
        inputs_layout.addWidget(QPushButton("浏览...", clicked=self.select_points_folder), 0, 2)
        inputs_layout.addWidget(QLabel("样地格网SHP:"), 1, 0);
        inputs_layout.addWidget(self.grid_shp_edit, 1, 1);
        inputs_layout.addWidget(QPushButton("浏览...", clicked=self.select_grid_shp), 1, 2)
        inputs_layout.addWidget(QLabel("单木AGB映射CSV:"), 2, 0);
        inputs_layout.addWidget(self.agb_csv_edit, 2, 1);
        inputs_layout.addWidget(QPushButton("浏览...", clicked=self.select_agb_csv), 2, 2)
        main_layout.addWidget(inputs_group)

        # --- 输出设置 ---
        outputs_group = QGroupBox("输出文件设置")
        outputs_layout = QGridLayout(outputs_group)
        self.output_shp_edit = QLineEdit();
        self.output_csv_edit = QLineEdit()
        outputs_layout.addWidget(QLabel("输出样地SHP:"), 0, 0);
        outputs_layout.addWidget(self.output_shp_edit, 0, 1);
        outputs_layout.addWidget(QPushButton("另存为...", clicked=self.select_output_shp), 0, 2)
        outputs_layout.addWidget(QLabel("输出样地CSV:"), 1, 0);
        outputs_layout.addWidget(self.output_csv_edit, 1, 1);
        outputs_layout.addWidget(QPushButton("另存为...", clicked=self.select_output_csv), 1, 2)
        main_layout.addWidget(outputs_group)

        # --- 参数设置 ---
        params_group = QGroupBox("算法参数设置")
        params_layout = QFormLayout(params_group)
        self.min_trees_spin = QSpinBox(value=3, minimum=1)
        self.slice_height_spin = QDoubleSpinBox(value=0.5, minimum=0.01, singleStep=0.1, decimals=2)
        self.num_workers_spin = QSpinBox(value=max(1, multiprocessing.cpu_count() - 1), minimum=1,
                                         maximum=multiprocessing.cpu_count())
        params_layout.addRow("样地内最小树木数:", self.min_trees_spin)
        params_layout.addRow("体积计算切片高度:", self.slice_height_spin)
        params_layout.addRow("并行计算核心数:", self.num_workers_spin)
        main_layout.addWidget(params_group)

        # --- 可视化设置 ---
        plot_group = QGroupBox("可选的可视化")
        plot_layout = QHBoxLayout(plot_group)
        self.plot_checkbox = QCheckBox("生成概览图");
        self.plot_path_edit = QLineEdit();
        self.plot_path_button = QPushButton("另存为...", clicked=self.select_plot_path)
        plot_layout.addWidget(self.plot_checkbox);
        plot_layout.addWidget(self.plot_path_edit);
        plot_layout.addWidget(self.plot_path_button)
        self.plot_checkbox.toggled.connect(self.toggle_plot_widgets);
        self.toggle_plot_widgets(False)
        main_layout.addWidget(plot_group)

        # --- 控制与反馈 ---
        self.start_button = QPushButton("开始聚合");
        self.start_button.clicked.connect(self.start_aggregation)
        self.progress_bar = QProgressBar();
        self.log_box = QTextEdit();
        self.log_box.setReadOnly(True)
        main_layout.addWidget(self.start_button);
        main_layout.addWidget(self.progress_bar);
        main_layout.addWidget(self.log_box)
        self._check_inputs()

    # --- 所有方法保持不变 ---
    def select_points_folder(self):
        path = QFileDialog.getExistingDirectory(self, "选择单木点云文件夹"); path and self.points_folder_edit.setText(
            path); self._check_inputs()

    def select_grid_shp(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择样地格网Shapefile", "",
                                              "Shapefile (*.shp)"); path and self.grid_shp_edit.setText(
            path); self._check_inputs()

    def select_agb_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择单木AGB映射CSV", "",
                                              "CSV 文件 (*.csv)"); path and self.agb_csv_edit.setText(
            path); self._check_inputs()

    def select_output_shp(self):
        path, _ = QFileDialog.getSaveFileName(self, "选择输出Shapefile路径", "",
                                              "Shapefile (*.shp)"); path and self.output_shp_edit.setText(
            path); self._check_inputs()

    def select_output_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "选择输出CSV路径", "",
                                              "CSV 文件 (*.csv)"); path and self.output_csv_edit.setText(
            path); self._check_inputs()

    def select_plot_path(self):
        path, _ = QFileDialog.getSaveFileName(self, "选择概览图保存路径", "",
                                              "PNG 文件 (*.png)"); path and self.plot_path_edit.setText(path)

    def toggle_plot_widgets(self, checked):
        self.plot_path_edit.setEnabled(checked); self.plot_path_button.setEnabled(checked)

    def _check_inputs(self):
        all_ok = all([os.path.isdir(self.points_folder_edit.text()), os.path.exists(self.grid_shp_edit.text()),
                      os.path.exists(self.agb_csv_edit.text()), bool(self.output_shp_edit.text()),
                      bool(self.output_csv_edit.text())])
        self.start_button.setEnabled(all_ok)

    def start_aggregation(self):
        params = {
            'points_folder': self.points_folder_edit.text(), 'grid_shp': self.grid_shp_edit.text(),
            'agb_csv': self.agb_csv_edit.text(),
            'output_shp': self.output_shp_edit.text(), 'output_csv': self.output_csv_edit.text(),
            'min_trees': self.min_trees_spin.value(),
            'slice_height': self.slice_height_spin.value(), 'num_workers': self.num_workers_spin.value(),
            'generate_plot': self.plot_checkbox.isChecked(),
            'plot_path': self.plot_path_edit.text()
        }
        if params['generate_plot'] and not params['plot_path']: QMessageBox.warning(self, "警告",
                                                                                    "请为概览图指定一个保存路径。"); return

        self.set_ui_enabled(False);
        self.log_box.clear();
        self.progress_bar.setValue(0);
        self.log_box.append("任务开始...")
        self.thread = QThread();
        self.worker = AggregationWorker(params);
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run);
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.task_finished);
        self.worker.error.connect(self.task_error)
        self.thread.start()

    def set_ui_enabled(self, enabled):
        self.start_button.setEnabled(enabled); [child.setEnabled(enabled) for child in self.findChildren(QGroupBox)]

    def update_progress(self, percent, message):
        self.progress_bar.setValue(percent); self.log_box.append(message)

    def task_finished(self, out_shp, out_csv):
        self.progress_bar.setValue(100)
        self.set_ui_enabled(True)
        if self.thread: self.thread.quit(); self.thread.wait()
        QMessageBox.information(self, "成功", f"AGB聚合完成！\n结果已保存到:\n- {out_shp}\n- {out_csv}")

    def task_error(self, message):
        self.set_ui_enabled(True)
        if self.thread and self.thread.isRunning(): self.thread.quit(); self.thread.wait()
        QMessageBox.critical(self, "错误", message)

# 【修改点】: 移除整个 if __name__ == '__main__': 块
# multiprocessing.freeze_support() 已经移到 main.py