import sys
import os
import pandas as pd
import numpy as np
import rasterio
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm  # Used internally for progress calculation

from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QProgressBar, QTextEdit,
                             QLabel, QLineEdit, QFileDialog, QMessageBox, QWidget, QVBoxLayout,
                             QHBoxLayout, QGroupBox, QCheckBox, QGridLayout)
from PyQt5.QtCore import QThread, QObject, pyqtSignal, Qt


# ==============================================================================
#  Backend Logic - S2FeatureWorker
#  (Refactored from your original script)
# ==============================================================================
class S2FeatureWorker(QObject):
    # Signals to communicate with the GUI
    progress = pyqtSignal(int, str)  # Percentage (0-100), Message
    finished = pyqtSignal(str)  # Output file path
    error = pyqtSignal(str)  # Error message

    def __init__(self, raster_path, csv_path, output_path, texture_bands, window_sizes, x_col, y_col, id_col):
        super().__init__()
        self.raster_path = raster_path
        self.csv_path = csv_path
        self.output_path = output_path
        self.texture_bands = texture_bands
        self.window_sizes = window_sizes
        self.x_col, self.y_col, self.id_col = x_col, y_col, id_col
        self.is_running = True
        self.scaling_factor = 10000.0

    def stop(self):
        self.is_running = False

    # --- Internal helper methods for GLCM (moved from original script) ---
    def _calculate_entropy(self, glcm):
        # ... (Your GLCM helper functions remain the same)
        entropies = []
        for d in range(glcm.shape[2]):
            for a in range(glcm.shape[3]):
                glcm_slice = glcm[:, :, d, a]
                if np.sum(glcm_slice) == 0: entropies.append(0); continue
                non_zero = glcm_slice[glcm_slice > 0]
                entropies.append(-np.sum(non_zero * np.log2(non_zero)))
        return np.mean(entropies)

    def _calculate_variance_glcm(self, glcm):
        variances = []
        i, j = np.mgrid[0:glcm.shape[0], 0:glcm.shape[1]]
        for d in range(glcm.shape[2]):
            for a in range(glcm.shape[3]):
                glcm_slice = glcm[:, :, d, a]
                if np.sum(glcm_slice) == 0: variances.append(0); continue
                mean_i = np.sum(i * glcm_slice)
                variances.append(np.sum(((i - mean_i) ** 2) * glcm_slice))
        return np.mean(variances)

    def _calculate_cluster_shade(self, glcm):
        shades = []
        i, j = np.mgrid[0:glcm.shape[0], 0:glcm.shape[1]]
        for d in range(glcm.shape[2]):
            for a in range(glcm.shape[3]):
                glcm_slice = glcm[:, :, d, a]
                if np.sum(glcm_slice) == 0: shades.append(0); continue
                mu_i = np.sum(i * glcm_slice)
                mu_j = np.sum(j * glcm_slice)
                shades.append(np.sum((((i - mu_i) + (j - mu_j)) ** 3) * glcm_slice))
        return np.mean(shades)

    def run(self):
        """Main processing logic, executed in the QThread."""
        try:
            # --- 1. Load data ---
            self.progress.emit(0, f"正在加载坐标文件: {os.path.basename(self.csv_path)}...")
            points_df = pd.read_csv(self.csv_path)

            with rasterio.open(self.raster_path) as src:
                self.progress.emit(5, f"栅格文件加载成功: {src.width}x{src.height}, {src.count} bands.")

                # --- 2. Extract VI and Reflectance ---
                self.progress.emit(10, "步骤 1/3: 开始提取植被指数和反射率...")
                vi_df = self._extract_vi_and_reflectance(points_df, src)
                if not self.is_running: self.error.emit("任务被中止。"); return

                # --- 3. Extract Texture Features ---
                texture_df = pd.DataFrame({self.id_col: points_df[self.id_col]})
                if self.texture_bands and self.window_sizes:
                    self.progress.emit(40, "步骤 2/3: 开始提取纹理特征...")
                    texture_df = self._extract_texture(points_df, src)
                    if not self.is_running: self.error.emit("任务被中止。"); return
                else:
                    self.progress.emit(40, "步骤 2/3: 跳过纹理特征提取（未选择波段/窗口）。")

                # --- 4. Merge and Save ---
                self.progress.emit(95, "步骤 3/3: 合并所有特征...")
                final_df = pd.merge(vi_df, texture_df, on=self.id_col, how='left')

                self.progress.emit(98, f"正在保存结果到: {self.output_path}...")
                final_df.to_csv(self.output_path, index=False, encoding='utf-8-sig')

            self.progress.emit(100, "所有任务成功完成！")
            self.finished.emit(self.output_path)

        except Exception as e:
            self.error.emit(f"发生错误: {str(e)}")

    def _extract_vi_and_reflectance(self, points_df, src):
        # ... (This logic is mostly from your original script) ...
        results = []
        # Simplified for brevity, you can add all your VIs here
        band_map = {"B2": 2, "B3": 3, "B4": 4, "B5": 5, "B6": 6, "B7": 7, "B8": 8, "B11": 11, "B12": 12}

        band_arrays_reflectance = {}
        for name, idx in band_map.items():
            band_data = src.read(idx).astype(np.float32)
            nodata = src.nodatavals[idx - 1]
            if nodata is not None: band_data[band_data == nodata] = np.nan
            band_arrays_reflectance[name] = band_data / self.scaling_factor

        B, G, R, NIR, SWIR1 = (band_arrays_reflectance.get(b, 0) for b in ["B2", "B3", "B4", "B8", "B11"])
        with np.errstate(divide='ignore', invalid='ignore'):
            indices = {'NDVI': (NIR - R) / (NIR + R), 'NDWI': (G - NIR) / (G + NIR)}

        total_points = len(points_df)
        for i, point in points_df.iterrows():
            if not self.is_running: break
            row, col = src.index(point[self.x_col], point[self.y_col])
            point_result = {self.id_col: point[self.id_col], self.x_col: point[self.x_col],
                            self.y_col: point[self.y_col]}

            for name, arr in indices.items():
                point_result[name] = arr[row, col]

            for name, arr in band_arrays_reflectance.items():
                point_result[f'{name}_refl'] = arr[row, col]

            results.append(point_result)

            # Update progress (from 10% to 40%)
            percent = 10 + int((i + 1) / total_points * 30)
            self.progress.emit(percent, f"提取指数: {i + 1}/{total_points}")

        return pd.DataFrame(results)

    def _extract_texture(self, points_df, src):
        # ... (This logic is from your original script) ...
        results = []
        feature_map = {'contrast': 'CON', 'dissimilarity': 'DIS', 'homogeneity': 'HOM', 'ASM': 'ASM',
                       'energy': 'ENERGY', 'correlation': 'COR'}

        total_points = len(points_df)
        for i, point in points_df.iterrows():
            if not self.is_running: break
            row, col = src.index(point[self.x_col], point[self.y_col])
            point_result = {self.id_col: point[self.id_col]}

            for band_idx in self.texture_bands:
                band_data = src.read(band_idx)
                for w in self.window_sizes:
                    half_w = w // 2
                    if not (half_w <= row < src.height - half_w and half_w <= col < src.width - half_w): continue

                    window = band_data[row - half_w: row + half_w + 1, col - half_w: col + half_w + 1]
                    # Normalize window to 8-bit for GLCM
                    min_val, max_val = window.min(), window.max()
                    if min_val >= max_val: continue
                    window_8bit = ((window - min_val) / (max_val - min_val) * 255).astype(np.uint8)

                    glcm = graycomatrix(window_8bit, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                        levels=256, symmetric=True, normed=True)

                    for prop, abbr in feature_map.items():
                        point_result[f'B{band_idx}_W{w}_{abbr}'] = np.mean(graycoprops(glcm, prop))

                    point_result[f'B{band_idx}_W{w}_SHA'] = self._calculate_cluster_shade(glcm)

            results.append(point_result)

            # Update progress (from 40% to 95%)
            percent = 40 + int((i + 1) / total_points * 55)
            self.progress.emit(percent, f"提取纹理: {i + 1}/{total_points}")

        return pd.DataFrame(results)


# ==============================================================================
#  Frontend UI - S2FeatureExtractorWindow
# ==============================================================================
class S2FeatureExtractorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("哨兵二号特征提取工具")
        self.setGeometry(100, 100, 600, 500)
        self.thread, self.worker = None, None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Input Files Group ---
        inputs_group = QGroupBox("输入文件")
        inputs_layout = QGridLayout(inputs_group)
        self.raster_path_edit = QLineEdit()
        self.csv_path_edit = QLineEdit()
        inputs_layout.addWidget(QLabel("栅格影像:"), 0, 0)
        inputs_layout.addWidget(self.raster_path_edit, 0, 1)
        inputs_layout.addWidget(QPushButton("浏览...", clicked=self.select_raster), 0, 2)
        inputs_layout.addWidget(QLabel("坐标CSV:"), 1, 0)
        inputs_layout.addWidget(self.csv_path_edit, 1, 1)
        inputs_layout.addWidget(QPushButton("浏览...", clicked=self.select_csv), 1, 2)
        main_layout.addWidget(inputs_group)

        # --- Parameters Group ---
        params_group = QGroupBox("纹理参数")
        params_layout = QVBoxLayout(params_group)
        # Bands
        bands_layout = QHBoxLayout()
        bands_layout.addWidget(QLabel("波段:"))
        self.band_checkboxes = {
            f'B{b}': QCheckBox(f'B{b}') for b in [2, 3, 4, 8, 11, 12]
        }
        for checkbox in self.band_checkboxes.values(): bands_layout.addWidget(checkbox)
        params_layout.addLayout(bands_layout)
        # Window Sizes
        windows_layout = QHBoxLayout()
        windows_layout.addWidget(QLabel("窗口大小:"))
        self.window_checkboxes = {
            size: QCheckBox(f'{size}x{size}') for size in [3, 5, 7, 9]
        }
        for checkbox in self.window_checkboxes.values(): windows_layout.addWidget(checkbox)
        params_layout.addLayout(windows_layout)
        main_layout.addWidget(params_group)

        # --- Output File Group ---
        output_group = QGroupBox("输出文件")
        output_layout = QGridLayout(output_group)
        self.output_path_edit = QLineEdit()
        output_layout.addWidget(QLabel("输出CSV:"), 0, 0)
        output_layout.addWidget(self.output_path_edit, 0, 1)
        output_layout.addWidget(QPushButton("另存为...", clicked=self.select_output), 0, 2)
        main_layout.addWidget(output_group)

        # --- Controls and Log ---
        self.start_button = QPushButton("开始提取")
        self.start_button.clicked.connect(self.start_extraction)
        self.progress_bar = QProgressBar()
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        main_layout.addWidget(self.start_button)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.log_box)

        self._check_inputs()

    def select_raster(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择栅格影像", "", "TIF 文件 (*.tif *.tiff);;所有文件 (*)")
        if path: self.raster_path_edit.setText(path)
        self._check_inputs()

    def select_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择坐标CSV文件", "", "CSV 文件 (*.csv);;所有文件 (*)")
        if path: self.csv_path_edit.setText(path)
        self._check_inputs()

    def select_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "选择输出CSV文件路径", "", "CSV 文件 (*.csv);;所有文件 (*)")
        if path: self.output_path_edit.setText(path)
        self._check_inputs()

    def _check_inputs(self):
        """Enable start button only if all inputs are valid."""
        raster_ok = os.path.exists(self.raster_path_edit.text())
        csv_ok = os.path.exists(self.csv_path_edit.text())
        output_ok = bool(self.output_path_edit.text())
        self.start_button.setEnabled(raster_ok and csv_ok and output_ok)

    def start_extraction(self):
        # Gather parameters from UI
        selected_bands_map = {'B2': 2, 'B3': 3, 'B4': 4, 'B8': 8, 'B11': 11, 'B12': 12}
        selected_bands = [selected_bands_map[name] for name, cb in self.band_checkboxes.items() if cb.isChecked()]
        selected_windows = [size for size, cb in self.window_checkboxes.items() if cb.isChecked()]

        if (selected_bands and not selected_windows) or (not selected_bands and selected_windows):
            QMessageBox.warning(self, "参数错误", "如果要计算纹理特征，必须同时选择至少一个波段和一个窗口大小。")
            return

        self.set_ui_enabled(False)
        self.log_box.clear()
        self.progress_bar.setValue(0)
        self.log_box.append("任务开始...")

        self.thread = QThread()
        self.worker = S2FeatureWorker(
            raster_path=self.raster_path_edit.text(),
            csv_path=self.csv_path_edit.text(),
            output_path=self.output_path_edit.text(),
            texture_bands=selected_bands,
            window_sizes=selected_windows,
            x_col='X', y_col='Y', id_col='ID'  # Assuming these column names
        )
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.task_finished)
        self.worker.error.connect(self.task_error)

        self.thread.start()

    def set_ui_enabled(self, enabled):
        self.start_button.setEnabled(enabled)
        # Disable all input groups
        for child in self.findChildren(QGroupBox):
            child.setEnabled(enabled)

    def update_progress(self, percent, message):
        self.progress_bar.setValue(percent)
        self.log_box.append(message)

    def task_finished(self, output_path):
        self.set_ui_enabled(True)
        self.thread.quit()
        self.thread.wait()
        QMessageBox.information(self, "成功", f"特征提取完成！\n结果已保存到: {output_path}")

    def task_error(self, message):
        self.set_ui_enabled(True)
        self.thread.quit()
        self.thread.wait()
        QMessageBox.critical(self, "错误", message)


# ==============================================================================
#  Application Entry Point
# ==============================================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = S2FeatureExtractorWindow()
    window.show()
    sys.exit(app.exec_())