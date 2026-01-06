# -*- coding: utf-8 -*-

# 核心库
import sys
import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# PyQt5 相关库
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QProgressBar, QTextEdit,
                             QLabel, QLineEdit, QFileDialog, QMessageBox, QWidget, QVBoxLayout,
                             QGroupBox, QCheckBox, QGridLayout, QDoubleSpinBox, QFormLayout)
from PyQt5.QtCore import QThread, QObject, pyqtSignal


# ==============================================================================
#  后端工作线程 (PlotFeatureWorker) - 无需修改
# ==============================================================================
class PlotFeatureWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, input_folder, output_path, features_to_calc, volumetric_params):
        super().__init__()
        self.input_folder = input_folder
        self.output_path = output_path
        self.features_to_calc = features_to_calc
        self.volumetric_params = volumetric_params
        self.is_running = True

    def stop(self):
        self.is_running = False

    def _calculate_volumetric_indices(self, point_cloud_df):
        params = self.volumetric_params
        x = point_cloud_df['X'].values
        y = point_cloud_df['Y'].values
        z = point_cloud_df['Z'].values

        canopy_points_mask = z > params['z_threshold']
        canopy_z = z[canopy_points_mask]
        if len(canopy_z) == 0: return {'3DVI': 0, '3DPI': 0}

        canopy_x = x[canopy_points_mask]
        canopy_y = y[canopy_points_mask]

        # 3DVI
        voxel_size = params['voxel_size']
        min_x, max_x = np.min(canopy_x), np.max(canopy_x)
        min_y, max_y = np.min(canopy_y), np.max(canopy_y)
        min_z_canopy = np.min(canopy_z)
        x_coords = ((canopy_x - min_x) // voxel_size).astype(int)
        y_coords = ((canopy_y - min_y) // voxel_size).astype(int)
        z_coords = ((canopy_z - min_z_canopy) // voxel_size).astype(int)
        unique_voxels_count = len(np.unique(np.column_stack((x_coords, y_coords, z_coords)), axis=0))
        Wn = int(np.ceil((max_x - min_x) / voxel_size))
        Ln = int(np.ceil((max_y - min_y) / voxel_size))
        D3VI = unique_voxels_count / (Wn * Ln) if Wn * Ln > 0 else 0

        # 3DPI
        max_z_canopy = np.max(canopy_z)
        height_range = max_z_canopy - min_z_canopy
        if height_range <= 0:
            D3PI = 0
        else:
            num_segments = int(np.ceil(height_range / params['vertical_segment_size']))
            bins = np.linspace(min_z_canopy, max_z_canopy, num_segments + 1)
            pi, _ = np.histogram(canopy_z, bins=bins)
            pt = len(canopy_z)
            pcs = np.cumsum(pi[::-1])[::-1]
            if pt > 0:
                term = (pi / pt) * (pcs / pt) ** params['k_factor']
                D3PI = np.sum(term)
            else:
                D3PI = 0

        return {'3DVI': D3VI, '3DPI': D3PI}

    def run(self):
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
                    column_names = ['X', 'Y', 'Z', 'intensity', 'classfication', 'GPSTime', 'returnNumber']
                    df = pd.read_csv(file_path, header=0, names=column_names, on_bad_lines='skip')
                    df['X'] = pd.to_numeric(df['X'], errors='coerce')
                    df['Y'] = pd.to_numeric(df['Y'], errors='coerce')
                    df['Z'] = pd.to_numeric(df['Z'], errors='coerce')
                    df['intensity'] = pd.to_numeric(df['intensity'], errors='coerce')
                    df.dropna(subset=['X', 'Y', 'Z', 'intensity'], inplace=True)
                    if df.empty: continue

                    plot_id = os.path.splitext(file_name)[0]
                    result = {"Plot_ID": plot_id}

                    max_z_point = df.loc[df['Z'].idxmax()]
                    result['X'], result['Y'] = max_z_point['X'], max_z_point['Y']

                    heights = df['Z'].values
                    intensities = df['intensity'].values

                    # --- 高度相关特征 ---
                    if len(heights) > 0:
                        if self.features_to_calc.get("h_percentiles"):
                            h_percentiles_vals = np.percentile(heights, percentile_levels)
                            result.update({f"H{p}": val for p, val in zip(percentile_levels, h_percentiles_vals)})
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
                            h_mean, h_median, h_max, h_min = np.mean(heights), np.median(heights), np.max(
                                heights), np.min(heights)
                            result.update(
                                {"Hiq": np.percentile(heights, 75) - np.percentile(heights, 25), "Hmin": h_min,
                                 "Hmax": h_max, "Hmean": h_mean, "Hmedian": h_median, "Hstd": np.std(heights),
                                 "Hvar": np.var(heights), "Hcv": np.std(heights) / h_mean if h_mean != 0 else 0,
                                 "Hsq": np.sqrt(np.mean(heights ** 2)), "Hcm": np.cbrt(np.mean(heights ** 3)),
                                 "Hskew": skew(heights), "Hkurt": kurtosis(heights),
                                 "Hmadme": np.median(np.abs(heights - h_median)),
                                 "Hmad": np.mean(np.abs(heights - h_mean)),
                                 "Hcanopy": (h_mean - h_min) / (h_max - h_min) if (h_max - h_min) != 0 else 0})

                    # --- 密度特征 ---
                    if self.features_to_calc.get("density") and len(heights) > 1:
                        h_min, h_max = np.min(heights), np.max(heights)
                        height_range = h_max - h_min
                        if height_range > 0:
                            height_bins = np.linspace(h_min, h_max, 11)
                            for k in range(len(height_bins) - 1): result[f"D{k}"] = np.sum(
                                (heights >= height_bins[k]) & (heights < height_bins[k + 1])) / len(heights)
                        else:
                            for k in range(10): result[f"D{k}"] = 0.0

                    # --- 强度相关特征 ---
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
                            i_mean, i_median = np.mean(intensities), np.median(intensities)
                            result.update({"Iiq": np.percentile(intensities, 75) - np.percentile(intensities, 25),
                                           "Imax": np.max(intensities), "Imin": np.min(intensities), "Imean": i_mean,
                                           "Imedian": i_median, "Imadme": np.median(np.abs(intensities - i_median)),
                                           "Istd": np.std(intensities), "Ivar": np.var(intensities),
                                           "Icv": np.std(intensities) / i_mean if i_mean != 0 else 0,
                                           "Isq": np.sqrt(np.mean(intensities ** 2)),
                                           "Icm": np.cbrt(np.mean(intensities ** 3)), "Iskew": skew(intensities),
                                           "Ikurt": kurtosis(intensities),
                                           "Imad": np.mean(np.abs(intensities - i_mean))})

                    # --- 体积指数 ---
                    if self.features_to_calc.get("volumetric_indices"): result.update(
                        self._calculate_volumetric_indices(df))

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
class PlotFeatureExtractorTab(QWidget):  # 继承自 QWidget
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
        io_layout.addWidget(QLabel("样地点云文件夹:"), 0, 0)
        io_layout.addWidget(self.folder_path_edit, 0, 1)
        io_layout.addWidget(QPushButton("浏览...", clicked=self.select_folder), 0, 2)
        io_layout.addWidget(QLabel("输出文件 (CSV/XLSX):"), 1, 0)
        io_layout.addWidget(self.output_path_edit, 1, 1)
        io_layout.addWidget(QPushButton("另存为...", clicked=self.select_output), 1, 2)
        main_layout.addWidget(io_group)

        # --- 特征选择 ---
        features_group = QGroupBox("选择要计算的特征类别")
        features_layout = QGridLayout(features_group)
        self.feature_checkboxes = {
            "h_stats": QCheckBox("高度基本统计 (Hmean, Hstd 等)"), "h_percentiles": QCheckBox("高度百分位数 (H1-H99)"),
            "aih_percentiles": QCheckBox("累计高度百分位数 (AIH1-AIH99)"), "density": QCheckBox("密度特征 (D0-D9)"),
            "i_stats": QCheckBox("强度基本统计 (Imean, Istd 等)"), "i_percentiles": QCheckBox("强度百分位数 (I1-I99)"),
            "aii_percentiles": QCheckBox("累计强度百分位数 (AII1-AII99)"),
            "volumetric_indices": QCheckBox("3D体积/剖面指数 (3DVI, 3DPI)")
        }
        row, col = 0, 0
        for checkbox in self.feature_checkboxes.values():
            checkbox.setChecked(True)
            features_layout.addWidget(checkbox, row, col)
            col += 1
            if col > 1: col = 0; row += 1
        main_layout.addWidget(features_group)

        # --- 3DVI / 3DPI 参数设置 ---
        vol_params_group = QGroupBox("3DVI / 3DPI 参数")
        vol_params_layout = QFormLayout(vol_params_group)
        self.z_thresh_spin = QDoubleSpinBox(self, value=0.12, singleStep=0.01, decimals=3)
        self.voxel_size_spin = QDoubleSpinBox(self, value=0.07, singleStep=0.01, decimals=3)
        self.v_seg_size_spin = QDoubleSpinBox(self, value=0.02, singleStep=0.01, decimals=3)
        self.k_factor_spin = QDoubleSpinBox(self, value=-3.5, minimum=-10.0, maximum=10.0, singleStep=0.1, decimals=2)
        vol_params_layout.addRow("Z阈值 (z_threshold):", self.z_thresh_spin)
        vol_params_layout.addRow("体素大小 (voxel_size):", self.voxel_size_spin)
        vol_params_layout.addRow("垂直分层大小 (vertical_segment_size):", self.v_seg_size_spin)
        vol_params_layout.addRow("修正因子 (k_factor):", self.k_factor_spin)
        main_layout.addWidget(vol_params_group)

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
        path = QFileDialog.getExistingDirectory(self, "选择包含样地点云CSV的文件夹")
        if path: self.folder_path_edit.setText(path)
        self._check_inputs()

    def select_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "选择输出文件路径", "", "Excel 文件 (*.xlsx);;CSV 文件 (*.csv)")
        if path: self.output_path_edit.setText(path)
        self._check_inputs()

    def _check_inputs(self):
        folder_ok = os.path.isdir(self.folder_path_edit.text())
        output_ok = bool(self.output_path_edit.text())
        self.start_button.setEnabled(folder_ok and output_ok)

    def start_extraction(self):
        features_to_calculate = {name: cb.isChecked() for name, cb in self.feature_checkboxes.items()}
        volumetric_params = {
            "z_threshold": self.z_thresh_spin.value(), "voxel_size": self.voxel_size_spin.value(),
            "vertical_segment_size": self.v_seg_size_spin.value(), "k_factor": self.k_factor_spin.value()
        }

        self.set_ui_enabled(False)
        self.log_box.clear()
        self.progress_bar.setValue(0)
        self.log_box.append("任务开始...")

        self.thread = QThread()
        self.worker = PlotFeatureWorker(
            input_folder=self.folder_path_edit.text(), output_path=self.output_path_edit.text(),
            features_to_calc=features_to_calculate, volumetric_params=volumetric_params
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.task_finished)
        self.worker.error.connect(self.task_error)
        self.thread.start()

    def set_ui_enabled(self, enabled):
        self.start_button.setEnabled(enabled)
        for child in self.findChildren(QGroupBox): child.setEnabled(enabled)

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

# 【修改点】: 移除整个 if __name__ == '__main__': 块
# 因为此文件现在是模块, 由 main.py 导入和运行