# -*- coding: utf-8 -*-

import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# PyQt5 相关库
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
                             QProgressBar, QTextEdit, QLabel, QLineEdit, QFileDialog,
                             QMessageBox, QCheckBox, QFormLayout, QGridLayout)
from PyQt5.QtCore import QThread, QObject, pyqtSignal

# 动态导入 Boruta 和 SHAP，处理库不存在的情况
try:
    from boruta import BorutaPy

    BORUTA_AVAILABLE = True
except ImportError:
    BORUTA_AVAILABLE = False

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ==============================================================================
#  后端工作线程 (BorutaWorker)
# ==============================================================================
class BorutaWorker(QObject):
    progress = pyqtSignal(str, int, int)
    finished = pyqtSignal(str, str)  # output_csv_path, shap_plot_path (or empty string)
    error = pyqtSignal(str)

    def __init__(self, input_path, output_path, must_keep_feature, generate_shap):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.must_keep_feature = must_keep_feature
        self.generate_shap = generate_shap
        self.is_running = True

    def stop(self):
        self.is_running = False

    def run(self):
        try:
            total_steps = 3
            if self.generate_shap:
                total_steps = 6

            # 1. 读取和预处理数据
            self.progress.emit(f"步骤 1/{total_steps}: 正在读取数据...", 1, total_steps)
            df = pd.read_csv(self.input_path, index_col=0)

            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]

            # 分离必须保留的特征
            must_keep_series = None
            if self.must_keep_feature:
                if self.must_keep_feature in X.columns:
                    self.progress.emit(f"  -> 指定保留特征: '{self.must_keep_feature}'", 1, total_steps)
                    must_keep_series = X[[self.must_keep_feature]]
                    X = X.drop(columns=[self.must_keep_feature])
                else:
                    self.progress.emit(f"  -> 警告: 指定保留特征 '{self.must_keep_feature}' 不存在。", 1, total_steps)
                    self.must_keep_feature = None

            # 处理空值
            if y.isnull().values.any():
                rows_to_keep = y.notna()
                X = X.loc[rows_to_keep]
                y = y.loc[rows_to_keep]
                if must_keep_series is not None:
                    must_keep_series = must_keep_series.loc[rows_to_keep]
                self.progress.emit(f"  -> 已删除目标变量中的空值行。", 1, total_steps)

            if X.isnull().values.any():
                X.dropna(axis=1, inplace=True)
                self.progress.emit(f"  -> 已删除特征变量中的空值列。", 1, total_steps)

            if y.empty or X.empty:
                self.error.emit("错误：处理空值后，数据为空，无法继续。")
                return

            # 2. 运行 Boruta
            self.progress.emit(f"步骤 2/{total_steps}: 初始化 Boruta...", 2, total_steps)
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=5)
            boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42, max_iter=100)

            self.progress.emit(f"  -> 开始 Boruta 特征选择 (这可能需要一些时间)...", 2, total_steps)
            boruta_selector.fit(X.values, y.values)
            self.progress.emit(f"  -> Boruta 特征选择完成。", 2, total_steps)

            # 3. 保存筛选结果
            self.progress.emit(f"步骤 3/{total_steps}: 保存筛选后的文件...", 3, total_steps)
            boruta_selected_features = X.columns[boruta_selector.support_].tolist()
            final_X_boruta = X[boruta_selected_features]

            if must_keep_series is not None:
                selected_df = pd.concat([must_keep_series, final_X_boruta, y], axis=1)
            else:
                selected_df = pd.concat([final_X_boruta, y], axis=1)

            selected_df.to_csv(self.output_path, index=True)
            final_features = selected_df.columns[:-1].tolist()
            self.progress.emit(f"  -> 筛选出的特征: {final_features}", 3, total_steps)
            self.progress.emit(f"  -> 筛选后的数据已保存至: {self.output_path}", 3, total_steps)

            # 4. (可选) 生成 SHAP 图
            shap_plot_path = ""
            if self.generate_shap and SHAP_AVAILABLE:
                if selected_df.shape[1] <= 1:
                    self.progress.emit("  -> 警告: 没有足够的特征用于SHAP分析，跳过此步骤。", 3, total_steps)
                    self.finished.emit(self.output_path, shap_plot_path)
                    return

                # 4.1 训练新模型
                self.progress.emit(f"步骤 4/{total_steps}: 训练模型用于 SHAP 分析...", 4, total_steps)
                X_selected = selected_df.iloc[:, :-1]
                y_selected = selected_df.iloc[:, -1]
                model_for_shap = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model_for_shap.fit(X_selected, y_selected)

                # 4.2 计算 SHAP 值
                self.progress.emit(f"步骤 5/{total_steps}: 正在计算 SHAP 值...", 5, total_steps)
                explainer = shap.TreeExplainer(model_for_shap)
                shap_values = explainer.shap_values(X_selected)

                # 4.3 生成并保存 SHAP 图
                self.progress.emit(f"步骤 6/{total_steps}: 正在生成并保存 SHAP 图...", 6, total_steps)
                output_dir = os.path.dirname(self.output_path)
                base_name = os.path.splitext(os.path.basename(self.output_path))[0]
                shap_plot_path = os.path.join(output_dir, f"{base_name}_SHAP.png")

                plt.figure()
                shap.summary_plot(shap_values, X_selected, plot_type="bar", show=False)
                plt.tight_layout()
                plt.savefig(shap_plot_path, dpi=300)
                plt.close()
                self.progress.emit(f"  -> SHAP 特征重要性图已保存至: {shap_plot_path}", 6, total_steps)

            elif self.generate_shap and not SHAP_AVAILABLE:
                self.error.emit("错误: 'shap' 库未安装。无法生成 SHAP 图。")
                return

            self.finished.emit(self.output_path, shap_plot_path)

        except Exception as e:
            self.error.emit(f"处理失败: {str(e)}")


# ==============================================================================
#  主界面类 (BorutaFeatureSelectionTab)
# ==============================================================================
class BorutaFeatureSelectionTab(QWidget):
    def __init__(self):
        super().__init__()
        if not BORUTA_AVAILABLE:
            self.setup_unavailable_ui("BorutaPy")
            return

        self.worker, self.thread, self.input_file_path = None, None, None
        main_layout = QVBoxLayout(self)

        # 1. 文件选择
        file_group = QGroupBox("输入与输出")
        file_layout = QGridLayout(file_group)
        file_layout.addWidget(QLabel("数据文件:"), 0, 0)
        self.file_path_edit = QLineEdit(placeholderText="请选择一个包含特征和目标变量的CSV文件...", readOnly=True)
        self.select_file_button = QPushButton("选择文件...")
        file_layout.addWidget(self.file_path_edit, 0, 1)
        file_layout.addWidget(self.select_file_button, 0, 2)
        main_layout.addWidget(file_group)

        # 2. 参数设置
        params_group = QGroupBox("参数设置")
        params_layout = QFormLayout(params_group)
        self.must_keep_edit = QLineEdit(placeholderText="例如: Hmax")
        self.generate_shap_checkbox = QCheckBox("生成 SHAP 特征重要性图 (需安装shap库)")
        if SHAP_AVAILABLE:
            self.generate_shap_checkbox.setChecked(True)
        else:
            self.generate_shap_checkbox.setChecked(False)
            self.generate_shap_checkbox.setEnabled(False)
            self.generate_shap_checkbox.setText("生成 SHAP 特征重要性图 (shap库未安装)")

        params_layout.addRow("必须保留的特征 (可选):", self.must_keep_edit)
        params_layout.addRow(self.generate_shap_checkbox)
        main_layout.addWidget(params_group)

        # 3. 运行与反馈
        self.run_button = QPushButton("开始筛选")
        self.progress_bar = QProgressBar(value=0)
        self.log_box = QTextEdit(readOnly=True)
        main_layout.addWidget(self.run_button)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.log_box)

        # 信号连接
        self.select_file_button.clicked.connect(self.select_file)
        self.run_button.clicked.connect(self.start_selection)
        self._update_run_button_state()

    def setup_unavailable_ui(self, lib_name):
        layout = QVBoxLayout(self)
        label = QLabel(f"错误: '{lib_name}' 库未安装。\n请运行 'pip install {lib_name}' 来启用此功能。")
        layout.addWidget(label)
        self.setEnabled(False)

    def _update_run_button_state(self):
        self.run_button.setEnabled(bool(self.input_file_path))

    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择数据文件", "", "CSV 文件 (*.csv);;所有文件 (*)")
        if path:
            self.input_file_path = path
            self.file_path_edit.setText(path)
            self.log_box.append(f"已选择文件: {os.path.basename(path)}")
        self._update_run_button_state()

    def start_selection(self):
        if not self.input_file_path: return
        if self.thread and self.thread.isRunning():
            QMessageBox.warning(self, "警告", "一个任务正在运行中...")
            return

        # 获取输出路径
        default_name = os.path.splitext(os.path.basename(self.input_file_path))[0] + "_boruta.csv"
        output_path, _ = QFileDialog.getSaveFileName(self, "保存筛选后的文件", default_name, "CSV 文件 (*.csv)")
        if not output_path:
            self.log_box.append("已取消保存，任务中止。")
            return

        must_keep = self.must_keep_edit.text().strip() or None
        generate_shap = self.generate_shap_checkbox.isChecked()

        # 清理并禁用UI
        self.log_box.clear()
        self.progress_bar.setValue(0)
        self.set_ui_enabled(False)
        self.log_box.append("开始 Boruta 特征选择任务...")

        # 启动线程
        self.thread = QThread()
        self.worker = BorutaWorker(self.input_file_path, output_path, must_keep, generate_shap)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.task_complete)
        self.worker.error.connect(self.task_error)

        self.thread.start()

    def set_ui_enabled(self, enabled):
        self.run_button.setEnabled(enabled)
        self.select_file_button.setEnabled(enabled)
        self.findChild(QGroupBox, "参数设置").setEnabled(enabled)
        if enabled: self._update_run_button_state()

    def cleanup_after_task(self):
        self.set_ui_enabled(True)
        self.progress_bar.setValue(0)
        if self.thread:
            self.thread.quit()
            self.thread.wait()
            self.thread, self.worker = None, None

    def update_progress(self, message, current, total):
        self.log_box.append(message)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def task_complete(self, output_csv, shap_plot):
        self.log_box.append("\n所有任务完成！")
        self.cleanup_after_task()

        msg = f"特征选择成功！\n\n筛选后的文件保存在:\n{output_csv}"
        if shap_plot:
            msg += f"\n\nSHAP重要性图保存在:\n{shap_plot}"

        QMessageBox.information(self, '任务完成', msg)

    def task_error(self, error_message):
        self.log_box.append(f"\n错误: {error_message}")
        self.cleanup_after_task()
        QMessageBox.critical(self, "发生错误", error_message)