# --- START OF FILE tools/prediction_tab.py ---

# -*- coding: utf-8 -*-
import os
import pandas as pd
import joblib
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QPushButton, QProgressBar, QTextEdit, QLabel,
                             QLineEdit, QFileDialog, QMessageBox)
from PyQt5.QtCore import QThread, QObject, pyqtSignal


# ==============================================================================
#  后端预测工作线程 (PredictionWorker)
# ==============================================================================
class PredictionWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, model_path, data_path, output_path):
        super().__init__()
        self.model_path = model_path
        self.data_path = data_path
        self.output_path = output_path
        self.is_running = True

    def stop(self):
        self.is_running = False

    def run(self):
        try:
            self.progress.emit(f"正在加载模型: {os.path.basename(self.model_path)}")
            model = joblib.load(self.model_path)
            self.progress.emit("模型加载成功。")

            if not self.is_running: return

            self.progress.emit(f"正在加载待预测数据: {os.path.basename(self.data_path)}")
            new_data = pd.read_csv(self.data_path)
            X_new = new_data.copy()  # 复制一份用于预测
            self.progress.emit("数据加载成功。")

            # 关键一步：检查并加载与模型配对的缩放器
            model_dir = os.path.dirname(self.model_path)
            scaler_path = os.path.join(model_dir, "standard_scaler.joblib")

            model_type = type(model).__name__
            needs_scaling = model_type in ['SVR', 'KNeighborsRegressor', 'LinearRegression']

            if needs_scaling:
                if not os.path.exists(scaler_path):
                    self.error.emit(f"错误: 模型 '{model_type}' 需要数据缩放器, "
                                    f"但在 '{model_dir}' 目录下未找到 'standard_scaler.joblib'。")
                    return
                self.progress.emit("找到并加载数据缩放器...")
                scaler = joblib.load(scaler_path)
                X_new_transformed = scaler.transform(X_new)
                self.progress.emit("数据缩放完成。")
            else:
                X_new_transformed = X_new  # 如果不需要，直接使用原数据

            if not self.is_running: return

            self.progress.emit("开始进行预测...")
            predictions = model.predict(X_new_transformed)
            self.progress.emit("预测完成。")

            # 将预测结果添加到原始数据中并保存
            new_data['Predictions'] = predictions
            new_data.to_csv(self.output_path, index=False, encoding='utf-8-sig')
            self.progress.emit(f"预测结果已保存至: {self.output_path}")

            self.finished.emit(self.output_path)

        except Exception as e:
            self.error.emit(f"预测过程中发生错误: {str(e)}")


# ==============================================================================
#  预测功能主界面 (PredictionTab)
# ==============================================================================
class PredictionTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.thread = None
        self.model_file_path = None
        self.prediction_data_path = None

        main_layout = QVBoxLayout(self)

        prediction_group = QGroupBox("模型预测与应用")
        prediction_layout = QVBoxLayout(prediction_group)

        # 模型文件选择
        model_file_layout = QHBoxLayout()
        model_file_layout.addWidget(QLabel("模型文件:"))
        self.model_file_edit = QLineEdit(placeholderText="选择已训练的模型 (.joblib)", readOnly=True)
        self.select_model_file_button = QPushButton("选择模型...")
        model_file_layout.addWidget(self.model_file_edit)
        model_file_layout.addWidget(self.select_model_file_button)
        prediction_layout.addLayout(model_file_layout)

        # 预测数据选择
        predict_data_layout = QHBoxLayout()
        predict_data_layout.addWidget(QLabel("数据文件:"))
        self.predict_data_edit = QLineEdit(placeholderText="选择用于预测的CSV数据", readOnly=True)
        self.select_predict_data_button = QPushButton("选择数据...")
        predict_data_layout.addWidget(self.predict_data_edit)
        predict_data_layout.addWidget(self.select_predict_data_button)
        prediction_layout.addLayout(predict_data_layout)

        # 预测运行按钮
        self.predict_button = QPushButton("开始预测")
        prediction_layout.addWidget(self.predict_button)
        main_layout.addWidget(prediction_group)

        # 进度与日志
        main_layout.addWidget(QLabel("进度与日志:"))
        self.progress_bar = QProgressBar(value=0)
        self.log_box = QTextEdit(readOnly=True)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.log_box)

        # 信号连接
        self.select_model_file_button.clicked.connect(self.select_model_file)
        self.select_predict_data_button.clicked.connect(self.select_prediction_file)
        self.predict_button.clicked.connect(self.start_prediction_task)

        self._update_predict_button_state()

    def _update_predict_button_state(self):
        model_selected = bool(self.model_file_path)
        data_selected = bool(self.prediction_data_path)
        self.predict_button.setEnabled(model_selected and data_selected)

    def select_model_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "Joblib 文件 (*.joblib);;所有文件 (*)")
        if path:
            self.model_file_path = path
            self.model_file_edit.setText(path)
            self.log_box.append(f"已选择模型文件: {os.path.basename(path)}")
        self._update_predict_button_state()

    def select_prediction_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择预测数据文件", "", "CSV 文件 (*.csv);;所有文件 (*)")
        if path:
            self.prediction_data_path = path
            self.predict_data_edit.setText(path)
            self.log_box.append(f"已选择待预测数据: {os.path.basename(path)}")
        self._update_predict_button_state()

    def start_prediction_task(self):
        if not self.model_file_path or not self.prediction_data_path: return
        if self.thread and self.thread.isRunning():
            QMessageBox.warning(self, "警告", "一个任务正在运行中，请等待其完成后再开始新任务。")
            return

        output_path, _ = QFileDialog.getSaveFileName(self, "保存预测结果", "", "CSV 文件 (*.csv)")
        if not output_path:
            self.log_box.append("已取消保存预测结果，任务中止。")
            return

        if self.thread: self.thread.quit(); self.thread.wait()

        self.set_ui_enabled(False)
        self.log_box.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(0)  # 不确定进度的模式
        self.log_box.append("开始预测任务...")

        self.thread = QThread()
        self.worker = PredictionWorker(self.model_file_path, self.prediction_data_path, output_path)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(lambda msg: self.log_box.append(msg))
        self.worker.finished.connect(self.task_complete)
        self.worker.error.connect(self.task_error)

        self.thread.start()

    def set_ui_enabled(self, enabled):
        self.predict_button.setEnabled(enabled)
        self.select_model_file_button.setEnabled(enabled)
        self.select_predict_data_button.setEnabled(enabled)
        if enabled:
            self._update_predict_button_state()

    def cleanup_after_task(self):
        self.set_ui_enabled(True)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        if self.thread:
            self.thread.quit()
            self.thread.wait()
            self.thread, self.worker = None, None

    def task_complete(self, output_path):
        self.log_box.append("\n预测任务完成！")
        self.cleanup_after_task()
        QMessageBox.information(self, "预测完成", f"预测结果已成功保存到:\n{output_path}")

    def task_error(self, error_message):
        self.log_box.append(f"\n错误: {error_message}")
        self.cleanup_after_task()
        QMessageBox.critical(self, "发生错误", error_message)