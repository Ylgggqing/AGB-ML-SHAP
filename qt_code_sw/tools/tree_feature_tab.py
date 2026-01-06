# -*- coding: utf-8 -*-

# 核心库
import sys
import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import alphashape
from shapely.geometry import Polygon
import warnings

# PyQt5 相关库
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QProgressBar, QTextEdit,
                             QLabel, QLineEdit, QFileDialog, QMessageBox, QWidget, QVBoxLayout,
                             QGroupBox, QCheckBox, QGridLayout)
from PyQt5.QtCore import QThread, QObject, pyqtSignal

# 忽略alphashape可能产生的奇异矩阵警告
warnings.filterwarnings("ignore", category=UserWarning, message="Singular matrix")


# ==============================================================================
#  后端工作线程 (TreeFeatureWorker) - 无需修改
# ==============================================================================

class TreeFeatureWorker(QObject):
    # 信号定义
    progress = pyqtSignal(int, str)  # 百分比, 消息
    finished = pyqtSignal(str)  # 输出文件路径
    error = pyqtSignal(str)  # 错误消息

    def __init__(self, input_folder, output_path, features_to_calc):
        super().__init__()
        self.input_folder = input_folder
        self.output_path = output_path
        self.features_to_calc = features_to_calc
        self.is_running = True

    def stop(self):
        self.is_running = False

    def _calculate_2d_concave_hull_area(self, points, alpha=0.5):
        if len(points) < 3: return 0
        try:
            hull = alphashape.alphashape(points, alpha)
            if hull.is_empty or not hull.is_valid: return 0
            if isinstance(hull, Polygon): return hull.area
            return sum(p.area for p in hull.geoms)
        except Exception:
            return 0  # 返回0而不是报错

    def _calculate_3d_volume(self, points, alpha=0.5, z_step=1.0):
        if len(points) < 3: return 0.0
        min_z, max_z = points[:, 2].min(), points[:, 2].max()
        total_volume = 0.0
        for z in np.arange(min_z, max_z, z_step):
            layer_points = points[(points[:, 2] >= z) & (points[:, 2] < z + z_step)]
            if len(layer_points) >= 3:
                area = self._calculate_2d_concave_hull_area(layer_points[:, :2], alpha)
                total_volume += area * z_step
        return total_volume

    def run(self):
        """主处理逻辑，在QThread中执行"""
        try:
            self.progress.emit(0, f"正在扫描文件夹: {self.input_folder}...")
            csv_files = [f for f in os.listdir(self.input_folder) if f.endswith(".csv")]
            if not csv_files:
                self.error.emit("错误：在指定文件夹中未找到任何CSV文件。")
                return

            total_files = len(csv_files)
            all_results = []
            percentile_levels = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]

            for i, file_name in enumerate(csv_files):
                if not self.is_running:
                    self.error.emit("任务被用户中止。")
                    return

                percent = int((i + 1) / total_files * 100)
                self.progress.emit(percent, f"正在处理: {file_name} ({i + 1}/{total_files})")

                file_path = os.path.join(self.input_folder, file_name)
                try:
                    df = pd.read_csv(file_path)
                    df['X'] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
                    df['Y'] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
                    df['Z'] = pd.to_numeric(df.iloc[:, 2], errors='coerce')
                    df['Intensity'] = pd.to_numeric(df.iloc[:, 4], errors='coerce')
                    df.dropna(subset=['X', 'Y', 'Z'], inplace=True)
                    if df.empty: continue

                    tree_id = os.path.splitext(file_name)[0]
                    result = {"Tree_ID": tree_id}
                    max_z_point = df.loc[df['Z'].idxmax()]
                    result['X'], result['Y'] = max_z_point['X'], max_z_point['Y']

                    points_xyz = df[['X', 'Y', 'Z']].values
                    heights = df['Z'].values
                    intensities = df['Intensity'].dropna().values

                    if self.features_to_calc.get("canopy_structure") and len(points_xyz) >= 3:
                        points_unique = np.unique(points_xyz, axis=0)
                        result['S'] = self._calculate_2d_concave_hull_area(points_unique[:, :2])
                        result['V'] = self._calculate_3d_volume(points_unique)
                        result['CD'] = ((points_unique[:, 0].max() - points_unique[:, 0].min()) + (
                                    points_unique[:, 1].max() - points_unique[:, 1].min())) / 2

                    if len(heights) > 0:
                        if self.features_to_calc.get("h_percentiles"): result.update({f"H{p}": val for p, val in
                                                                                      zip(percentile_levels,
                                                                                          np.percentile(heights,
                                                                                                        percentile_levels))})
                        if self.features_to_calc.get("aih_percentiles"):
                            sorted_heights = np.sort(heights)
                            cumulative_heights = np.cumsum(sorted_heights)
                            total_height = cumulative_heights[-1] if len(cumulative_heights) > 0 else 0
                            if total_height > 0:
                                cumulative_percentage = (cumulative_heights / total_height) * 100
                                aih_vals = [sorted_heights[min(np.searchsorted(cumulative_percentage, p, side='right'),
                                                               len(sorted_heights) - 1)] for p in percentile_levels]
                                result.update({f"AIH{p}": val for p, val in zip(percentile_levels, aih_vals)})
                                result["AIHiq"] = np.percentile(aih_vals, 75) - np.percentile(aih_vals, 25)
                        if self.features_to_calc.get("h_stats"):
                            h_mean, h_median, h_std, h_max, h_min = np.mean(heights), np.median(heights), np.std(
                                heights), np.max(heights), np.min(heights)
                            result.update(
                                {"Hiq": np.percentile(heights, 75) - np.percentile(heights, 25), "Hmin": h_min,
                                 "Hmax": h_max, "Hmean": h_mean, "Hmedian": h_median, "Hstd": h_std,
                                 "Hvar": np.var(heights), "Hcv": h_std / h_mean if h_mean != 0 else 0,
                                 "Hsq": np.sqrt(np.mean(heights ** 2)), "Hcm": np.cbrt(np.mean(heights ** 3)),
                                 "Hskew": skew(heights), "Hkurt": kurtosis(heights),
                                 "Hmadme": np.median(np.abs(heights - h_median)),
                                 "Hmad": np.mean(np.abs(heights - h_mean)),
                                 "Hcanopy": (h_mean - h_min) / (h_max - h_min) if (h_max - h_min) != 0 else 0})

                    if self.features_to_calc.get("density") and len(heights) > 1:
                        h_min, h_max = np.min(heights), np.max(heights)
                        if h_max > h_min:
                            height_bins = np.linspace(h_min, h_max, 11)
                            for k in range(len(height_bins) - 1): result[f"D{k}"] = np.sum(
                                (heights >= height_bins[k]) & (heights < height_bins[k + 1])) / len(heights)
                        else:
                            for k in range(10): result[f"D{k}"] = 0.0

                    if len(intensities) > 0:
                        if self.features_to_calc.get("i_percentiles"): result.update({f"I{p}": val for p, val in
                                                                                      zip(percentile_levels,
                                                                                          np.percentile(intensities,
                                                                                                        percentile_levels))})
                        if self.features_to_calc.get("aii_percentiles"):
                            sorted_intensities = np.sort(intensities)
                            cumulative_intensities = np.cumsum(sorted_intensities)
                            total_intensity = cumulative_intensities[-1] if len(cumulative_intensities) > 0 else 0
                            if total_intensity > 0:
                                cumulative_percentage = (cumulative_intensities / total_intensity) * 100
                                aii_vals = [sorted_intensities[
                                                min(np.searchsorted(cumulative_percentage, p, side='right'),
                                                    len(sorted_intensities) - 1)] for p in percentile_levels]
                                result.update({f"AII{p}": val for p, val in zip(percentile_levels, aii_vals)})
                                result["AIIiq"] = np.percentile(aii_vals, 75) - np.percentile(aii_vals, 25)
                        if self.features_to_calc.get("i_stats"):
                            i_mean, i_median, i_std, i_max, i_min = np.mean(intensities), np.median(
                                intensities), np.std(intensities), np.max(intensities), np.min(intensities)
                            result.update(
                                {"Iiq": np.percentile(intensities, 75) - np.percentile(intensities, 25), "Imin": i_min,
                                 "Imax": i_max, "Imean": i_mean, "Imedian": i_median, "Istd": i_std,
                                 "Ivar": np.var(intensities), "Icv": i_std / i_mean if i_mean != 0 else 0,
                                 "Iskew": skew(intensities), "Ikurt": kurtosis(intensities)})

                    all_results.append(result)
                except Exception as file_error:
                    print(f"警告：处理文件 {file_name} 时出错: {file_error}")
                    self.progress.emit(percent, f"警告: 处理 {file_name} 失败 - {file_error}")
                    continue

            self.progress.emit(100, "所有文件处理完毕，正在生成汇总文件...")
            final_df = pd.DataFrame(all_results)
            if self.output_path.endswith('.xlsx'):
                final_df.to_excel(self.output_path, index=False)
            else:
                final_df.to_csv(self.output_path, index=False, encoding='utf-8-sig')
            self.finished.emit(self.output_path)
        except Exception as e:
            self.error.emit(f"发生严重错误: {str(e)}")


# ==============================================================================
#  【修改点】: 主界面类重构为 QWidget
# ==============================================================================
class TreeFeatureExtractorTab(QWidget):  # 继承自 QWidget
    def __init__(self):
        super().__init__()
        # 【修改点】: 移除 setWindowTitle, setGeometry 和 setCentralWidget
        self.thread, self.worker = None, None

        # 【修改点】: 布局直接应用到 self (QWidget)
        main_layout = QVBoxLayout(self)

        # --- 输入/输出设置 ---
        io_group = QGroupBox("文件路径设置")
        io_layout = QGridLayout(io_group)
        self.folder_path_edit = QLineEdit()
        self.output_path_edit = QLineEdit()
        io_layout.addWidget(QLabel("单木点云文件夹:"), 0, 0)
        io_layout.addWidget(self.folder_path_edit, 0, 1)
        io_layout.addWidget(QPushButton("浏览...", clicked=self.select_folder), 0, 2)
        io_layout.addWidget(QLabel("输出文件 (CSV/XLSX):"), 1, 0)
        io_layout.addWidget(self.output_path_edit, 1, 1)
        io_layout.addWidget(QPushButton("另存为...", clicked=self.select_output), 1, 2)
        main_layout.addWidget(io_group)

        # --- 参数选择 ---
        params_group = QGroupBox("选择要计算的特征类别")
        params_layout = QGridLayout(params_group)
        self.feature_checkboxes = {
            "canopy_structure": QCheckBox("冠层结构 (S, V, CD)"),
            "h_stats": QCheckBox("高度基本统计 (Hmax, Hmean, Hstd 等)"),
            "h_percentiles": QCheckBox("高度百分位数 (H1-H99)"),
            "aih_percentiles": QCheckBox("累计高度百分位数 (AIH1-AIH99)"),
            "density": QCheckBox("密度特征 (D0-D9)"),
            "i_stats": QCheckBox("强度基本统计 (Imean, Istd 等)"),
            "i_percentiles": QCheckBox("强度百分位数 (I1-I99)"),
            "aii_percentiles": QCheckBox("累计强度百分位数 (AII1-AII99)"),
        }
        row, col = 0, 0
        for checkbox in self.feature_checkboxes.values():
            checkbox.setChecked(True)
            params_layout.addWidget(checkbox, row, col)
            col += 1
            if col > 1:
                col = 0;
                row += 1
        main_layout.addWidget(params_group)

        # --- 控制与反馈 ---
        self.start_button = QPushButton("开始提取")
        self.start_button.clicked.connect(self.start_extraction)
        self.progress_bar = QProgressBar()
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        main_layout.addWidget(self.start_button)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.log_box)

        self._check_inputs()

    # --- 所有方法保持不变 ---
    def select_folder(self):
        path = QFileDialog.getExistingDirectory(self, "选择包含单木点云CSV的文件夹")
        if path:
            self.folder_path_edit.setText(path)
        self._check_inputs()

    def select_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "选择输出文件路径", "", "Excel 文件 (*.xlsx);;CSV 文件 (*.csv)")
        if path:
            self.output_path_edit.setText(path)
        self._check_inputs()

    def _check_inputs(self):
        folder_ok = os.path.isdir(self.folder_path_edit.text())
        output_ok = bool(self.output_path_edit.text())
        self.start_button.setEnabled(folder_ok and output_ok)

    def start_extraction(self):
        features_to_calculate = {name: checkbox.isChecked() for name, checkbox in self.feature_checkboxes.items()}
        self.set_ui_enabled(False)
        self.log_box.clear()
        self.progress_bar.setValue(0)
        self.log_box.append("任务开始...")

        self.thread = QThread()
        self.worker = TreeFeatureWorker(
            input_folder=self.folder_path_edit.text(),
            output_path=self.output_path_edit.text(),
            features_to_calc=features_to_calculate
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.task_finished)
        self.worker.error.connect(self.task_error)
        self.thread.start()

    def set_ui_enabled(self, enabled):
        self.start_button.setEnabled(enabled)
        for child in self.findChildren(QGroupBox):
            child.setEnabled(enabled)

    def update_progress(self, percent, message):
        self.progress_bar.setValue(percent)
        self.log_box.append(message)

    def task_finished(self, output_path):
        self.set_ui_enabled(True)
        if self.thread:
            self.thread.quit()
            self.thread.wait()
            self.thread = None
        self.log_box.append(f"\n任务成功完成！结果已保存到: {output_path}")
        QMessageBox.information(self, "成功", f"特征提取完成！\n结果已保存到: {output_path}")

    def task_error(self, message):
        self.set_ui_enabled(True)
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
            self.thread = None
        self.log_box.append(f"\n错误: {message}")
        QMessageBox.critical(self, "错误", message)
