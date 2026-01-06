# -*- coding: utf-8 -*-

# ... (所有 import 语句保持不变) ...
import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QProgressBar, QTextEdit,
                             QLabel, QLineEdit, QFileDialog, QMessageBox, QWidget, QVBoxLayout,
                             QHBoxLayout, QDialog, QTabWidget, QFormLayout, QSpinBox,
                             QDoubleSpinBox, QComboBox, QDialogButtonBox)
from PyQt5.QtCore import QThread, QObject, pyqtSignal

# ... (动态导入和绘图函数部分保持不变) ...
try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
def save_detailed_scatter_plot(y_true, y_pred, model_name, file_basename, output_path):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    plt.figure(figsize=(8, 8)); plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', label=f'Predictions (R²={r2:.3f})'); min_val = min(y_true.min(), y_pred.min()); max_val = max(y_true.max(), y_pred.max()); plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Line'); plt.xlabel("True Values"); plt.ylabel("Predicted Values"); plt.title(f"{model_name} on {file_basename}\nRMSE: {rmse:.3f}"); plt.legend(); plt.grid(True); plt.savefig(output_path, dpi=150, bbox_inches='tight'); plt.close()

# ... (ModelingWorker 类保持不变) ...
class ModelingWorker(QObject):
    progress = pyqtSignal(str, int, int)
    model_finished = pyqtSignal(str, dict)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    def __init__(self, file_path, custom_params): super().__init__(); self.file_path = file_path; self.custom_params = custom_params; self.is_running = True
    def stop(self): self.is_running = False
    def run(self):
        try:
            data = pd.read_csv(self.file_path); X = data.iloc[:, :-1]; y = data.iloc[:, -1]; X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43); scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)
            models = {'SVR': SVR(**self.custom_params.get('SVR', {})), 'XGBoost': xgb.XGBRegressor(**self.custom_params.get('XGBoost', {})), 'RandomForest': RandomForestRegressor(**self.custom_params.get('RandomForest', {}))}
            if TABPFN_AVAILABLE and X_train.shape[0] <= 1024: models['TabPFN'] = TabPFNRegressor(**self.custom_params.get('TabPFN', {}))
            total_models = len(models)
            for i, (model_name, model) in enumerate(models.items()):
                if not self.is_running: self.error.emit("任务被用户中止。"); return
                self.progress.emit(f"正在处理 {model_name}...", i + 1, total_models)
                try:
                    if model_name == 'SVR': model.fit(X_train_scaled, y_train); y_pred_test = model.predict(X_test_scaled)
                    else: model.fit(X_train, y_train); y_pred_test = model.predict(X_test)
                    test_r2 = r2_score(y_test, y_pred_test); test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test)); result_dict = {'test_r2': test_r2, 'test_rmse': test_rmse, 'y_true': y_test.values, 'y_pred': y_pred_test}; self.model_finished.emit(model_name, result_dict)
                except Exception as e: self.error.emit(f"模型 '{model_name}' 训练失败: {str(e)}"); continue
            self.finished.emit()
        except Exception as e: self.error.emit(f"处理失败: {str(e)}")

# ==============================================================================
#  参数设置对话框 (Parameters Dialog) (已修正)
# ==============================================================================
class ParametersDialog(QDialog):
    # ... (__init__, create_svr_tab, _update_svr_fields, create_xgb_tab 保持不变) ...
    def __init__(self, initial_params, parent=None):
        super().__init__(parent); self.setWindowTitle("调整模型参数"); self.setMinimumWidth(450); self.params = initial_params; main_layout = QVBoxLayout(self); self.tabs = QTabWidget(); main_layout.addWidget(self.tabs); self.create_svr_tab(); self.create_rf_tab(); self.create_xgb_tab(); button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel); button_box.accepted.connect(self.accept); button_box.rejected.connect(self.reject); main_layout.addWidget(button_box)
    def create_svr_tab(self):
        tab = QWidget(); layout = QFormLayout(tab); self.svr_kernel = QComboBox(); self.svr_kernel.addItems(['rbf', 'linear', 'poly', 'sigmoid']); self.svr_kernel.setCurrentText(self.params['SVR'].get('kernel', 'rbf')); layout.addRow("Kernel:", self.svr_kernel); self.svr_c = QDoubleSpinBox(); self.svr_c.setRange(0.01, 10000.0); self.svr_c.setSingleStep(0.1); self.svr_c.setValue(self.params['SVR'].get('C', 1.0)); layout.addRow("C (正则化参数):", self.svr_c); self.svr_epsilon = QDoubleSpinBox(); self.svr_epsilon.setRange(0.0, 10.0); self.svr_epsilon.setSingleStep(0.01); self.svr_epsilon.setValue(self.params['SVR'].get('epsilon', 0.1)); layout.addRow("Epsilon (ε):", self.svr_epsilon); self.svr_gamma = QDoubleSpinBox(); self.svr_gamma.setRange(0.001, 100.0); self.svr_gamma.setSingleStep(0.01); self.svr_gamma.setValue(self.params['SVR'].get('gamma', 0.1)); layout.addRow("Gamma:", self.svr_gamma); self.svr_degree = QSpinBox(); self.svr_degree.setRange(1, 10); self.svr_degree.setValue(self.params['SVR'].get('degree', 3)); layout.addRow("Degree (多项式阶数):", self.svr_degree); self.svr_kernel.currentTextChanged.connect(self._update_svr_fields); self._update_svr_fields(self.svr_kernel.currentText()); self.tabs.addTab(tab, "SVR")
    def _update_svr_fields(self, kernel): is_poly = (kernel == 'poly'); uses_gamma = (kernel in ['rbf', 'poly', 'sigmoid']); self.svr_degree.setEnabled(is_poly); self.svr_gamma.setEnabled(uses_gamma)
    def create_xgb_tab(self):
        tab = QWidget(); layout = QFormLayout(tab); self.xgb_n_estimators = QSpinBox(); self.xgb_n_estimators.setRange(10, 5000); self.xgb_n_estimators.setSingleStep(10); self.xgb_n_estimators.setValue(self.params['XGBoost'].get('n_estimators', 100)); layout.addRow("N Estimators (树数量):", self.xgb_n_estimators); self.xgb_max_depth = QSpinBox(); self.xgb_max_depth.setRange(1, 100); self.xgb_max_depth.setValue(self.params['XGBoost'].get('max_depth', 7)); layout.addRow("Max Depth (最大深度):", self.xgb_max_depth); self.xgb_learning_rate = QDoubleSpinBox(); self.xgb_learning_rate.setRange(0.001, 1.0); self.xgb_learning_rate.setSingleStep(0.01); self.xgb_learning_rate.setDecimals(3); self.xgb_learning_rate.setValue(self.params['XGBoost'].get('learning_rate', 0.05)); layout.addRow("Learning Rate (学习率):", self.xgb_learning_rate); self.xgb_subsample = QDoubleSpinBox(); self.xgb_subsample.setRange(0.1, 1.0); self.xgb_subsample.setSingleStep(0.1); self.xgb_subsample.setValue(self.params['XGBoost'].get('subsample', 0.8)); layout.addRow("Subsample (样本采样率):", self.xgb_subsample); self.xgb_colsample_bytree = QDoubleSpinBox(); self.xgb_colsample_bytree.setRange(0.1, 1.0); self.xgb_colsample_bytree.setSingleStep(0.1); self.xgb_colsample_bytree.setValue(self.params['XGBoost'].get('colsample_bytree', 0.8)); layout.addRow("Colsample by Tree (特征采样率):", self.xgb_colsample_bytree); self.xgb_gamma = QDoubleSpinBox(); self.xgb_gamma.setRange(0.0, 10.0); self.xgb_gamma.setSingleStep(0.1); self.xgb_gamma.setValue(self.params['XGBoost'].get('gamma', 0)); layout.addRow("Gamma (最小损失减少):", self.xgb_gamma); self.xgb_reg_alpha = QDoubleSpinBox(); self.xgb_reg_alpha.setRange(0.0, 100.0); self.xgb_reg_alpha.setSingleStep(0.1); self.xgb_reg_alpha.setValue(self.params['XGBoost'].get('reg_alpha', 0)); layout.addRow("Reg Alpha (L1正则化):", self.xgb_reg_alpha); self.xgb_reg_lambda = QDoubleSpinBox(); self.xgb_reg_lambda.setRange(0.0, 100.0); self.xgb_reg_lambda.setSingleStep(0.1); self.xgb_reg_lambda.setValue(self.params['XGBoost'].get('reg_lambda', 1)); layout.addRow("Reg Lambda (L2正则化):", self.xgb_reg_lambda); self.tabs.addTab(tab, "XGBoost")

    def create_rf_tab(self):
        tab = QWidget()
        layout = QFormLayout(tab)

        self.rf_n_estimators = QSpinBox()
        self.rf_n_estimators.setRange(10, 5000); self.rf_n_estimators.setSingleStep(10)
        self.rf_n_estimators.setValue(self.params['RandomForest'].get('n_estimators', 100))
        layout.addRow("N Estimators (树数量):", self.rf_n_estimators)

        self.rf_max_depth = QSpinBox()
        self.rf_max_depth.setRange(1, 100)
        # 允许深度为0或负数，scikit-learn会将其解释为None (无限制)
        self.rf_max_depth.setSpecialValueText("None (无限制)")
        self.rf_max_depth.setMinimum(0)
        self.rf_max_depth.setValue(self.params['RandomForest'].get('max_depth', 10) or 0)
        layout.addRow("Max Depth (最大深度):", self.rf_max_depth)

        self.rf_min_samples_split = QSpinBox()
        self.rf_min_samples_split.setRange(2, 100)
        self.rf_min_samples_split.setValue(self.params['RandomForest'].get('min_samples_split', 2))
        layout.addRow("Min Samples Split:", self.rf_min_samples_split)

        self.rf_min_samples_leaf = QSpinBox()
        self.rf_min_samples_leaf.setRange(1, 100)
        self.rf_min_samples_leaf.setValue(self.params['RandomForest'].get('min_samples_leaf', 1))
        layout.addRow("Min Samples Leaf:", self.rf_min_samples_leaf)

        # --- 修正点 ---
        self.rf_max_features = QComboBox()
        self.rf_max_features.addItems(['sqrt', 'log2', 'None'])  # 移除了 'auto'
        current_max_features = self.params['RandomForest'].get('max_features', 'sqrt')
        # 处理 None 的情况
        self.rf_max_features.setCurrentText(str(current_max_features))
        layout.addRow("Max Features:", self.rf_max_features)

        self.tabs.addTab(tab, "RandomForest")

    def get_parameters(self):
        # --- 修正点 ---
        # 处理 max_features
        max_features_val = self.rf_max_features.currentText()
        if max_features_val == 'None':
            max_features_val = None

        # 处理 max_depth
        max_depth_val = self.rf_max_depth.value()
        if max_depth_val == 0:
            max_depth_val = None

        updated_params = {
            'SVR': {
                'kernel': self.svr_kernel.currentText(), 'C': self.svr_c.value(), 'gamma': self.svr_gamma.value(),
                'epsilon': self.svr_epsilon.value(), 'degree': self.svr_degree.value()
            },
            'RandomForest': {
                'n_estimators': self.rf_n_estimators.value(),
                'max_depth': max_depth_val,
                'min_samples_split': self.rf_min_samples_split.value(),
                'min_samples_leaf': self.rf_min_samples_leaf.value(),
                'max_features': max_features_val,
                'random_state': 43, 'n_jobs': -1
            },
            'XGBoost': {
                'n_estimators': self.xgb_n_estimators.value(), 'max_depth': self.xgb_max_depth.value(),
                'learning_rate': self.xgb_learning_rate.value(), 'subsample': self.xgb_subsample.value(),
                'colsample_bytree': self.xgb_colsample_bytree.value(), 'gamma': self.xgb_gamma.value(),
                'reg_alpha': self.xgb_reg_alpha.value(), 'reg_lambda': self.xgb_reg_lambda.value(),
                'random_state': 43, 'n_jobs': -1
            },
            'TabPFN': self.params.get('TabPFN', {})
        }
        return updated_params


# ==============================================================================
#  前端界面 (Frontend UI) - MainWindow (已修正)
# ==============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AGB 机器学习建模工具")
        self.setGeometry(100, 100, 500, 400)

        self.worker = None
        self.thread = None
        self.training_file_path = None
        self.model_results_cache = {}

        # --- 修正点 ---
        self.custom_params = {
            'SVR': {
                'kernel': 'rbf', 'C': 15.0, 'gamma': 0.1, 'epsilon': 0.1, 'degree': 3
            },
            'XGBoost': {
                'n_estimators': 1000, 'max_depth': 7, 'learning_rate': 0.05,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0,
                'reg_alpha': 0, 'reg_lambda': 1,
                'random_state': 43, 'n_jobs': -1
            },
            'RandomForest': {
                'n_estimators': 500, 'max_depth': 10, 'min_samples_split': 2,
                'min_samples_leaf': 1, 'max_features': 'sqrt',  # 'auto' -> 'sqrt'
                'random_state': 43, 'n_jobs': -1
            },
            'TabPFN': {'random_state': 43}
        }

        # ... (UI布局和信号连接部分保持不变) ...
        central_widget = QWidget(); self.setCentralWidget(central_widget); main_layout = QVBoxLayout(central_widget); file_selection_layout = QHBoxLayout(); self.file_label = QLabel("训练文件:"); self.file_path_edit = QLineEdit(); self.file_path_edit.setPlaceholderText("请选择一个CSV训练文件..."); self.file_path_edit.setReadOnly(True); self.select_file_button = QPushButton("选择文件..."); file_selection_layout.addWidget(self.file_label); file_selection_layout.addWidget(self.file_path_edit); file_selection_layout.addWidget(self.select_file_button); main_layout.addLayout(file_selection_layout); self.tune_params_button = QPushButton("调整模型参数..."); main_layout.addWidget(self.tune_params_button); self.run_button = QPushButton("开始建模"); self.progress_bar = QProgressBar(); self.progress_bar.setValue(0); main_layout.addWidget(self.run_button); main_layout.addWidget(self.progress_bar); self.log_box = QTextEdit(); self.log_box.setReadOnly(True); main_layout.addWidget(self.log_box); self.run_button.setEnabled(False); self.select_file_button.clicked.connect(self.select_training_file); self.tune_params_button.clicked.connect(self.open_parameters_dialog); self.run_button.clicked.connect(self.start_modeling_task)
    # ... (MainWindow 的所有方法保持不变) ...
    def select_training_file(self): file_path, _ = QFileDialog.getOpenFileName(self, "选择训练数据文件", "", "CSV 文件 (*.csv);;所有文件 (*)");_ and (setattr(self, 'training_file_path', file_path), self.file_path_edit.setText(file_path), self.run_button.setEnabled(True), self.log_box.append(f"已选择训练文件: {os.path.basename(file_path)}"))
    def open_parameters_dialog(self): dialog = ParametersDialog(self.custom_params, self); dialog.exec_() == QDialog.Accepted and (setattr(self, 'custom_params', dialog.get_parameters()), self.log_box.append("模型参数已更新。"), QMessageBox.information(self, "成功", "模型参数已成功更新！"))
    def start_modeling_task(self):
        if not self.training_file_path: QMessageBox.warning(self, "警告", "请先选择一个训练文件！"); return
        if self.thread and self.thread.isRunning(): QMessageBox.warning(self, "警告", "一个任务正在运行中，请等待其完成。"); return
        if self.thread: self.thread.quit(); self.thread.wait()
        self.model_results_cache.clear(); self.run_button.setEnabled(False); self.select_file_button.setEnabled(False); self.tune_params_button.setEnabled(False); self.log_box.clear(); self.progress_bar.setValue(0); self.log_box.append("开始建模任务...")
        self.thread = QThread(); self.worker = ModelingWorker(self.training_file_path, self.custom_params); self.worker.moveToThread(self.thread); self.thread.started.connect(self.worker.run); self.worker.progress.connect(self.update_progress); self.worker.model_finished.connect(self.log_model_result_and_cache); self.worker.finished.connect(self.task_complete); self.worker.error.connect(self.task_error); self.thread.start()
    def update_progress(self, message, current, total): self.log_box.append(message); self.progress_bar.setMaximum(total); self.progress_bar.setValue(current)
    def log_model_result_and_cache(self, model_name, result_dict): r2 = result_dict['test_r2']; rmse = result_dict['test_rmse']; self.log_box.append(f"  - {model_name} 完成: R²={r2:.3f}, RMSE={rmse:.3f}"); self.model_results_cache[model_name] = result_dict
    def cleanup_after_task(self): self.run_button.setEnabled(True); self.select_file_button.setEnabled(True); self.tune_params_button.setEnabled(True); self.progress_bar.setValue(0)
    def task_complete(self): self.log_box.append("\n所有任务完成！"); self.cleanup_after_task(); QMessageBox.question(self, '任务完成', "建模已完成。是否要生成精度散点图？", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) == QMessageBox.Yes and self.generate_plots()
    def generate_plots(self):
        if not self.model_results_cache: QMessageBox.information(self, "无结果", "没有可供绘图的模型结果。"); return
        output_dir = QFileDialog.getExistingDirectory(self, "选择保存绘图的文件夹")
        if output_dir:
            self.log_box.append(f"\n开始生成散点图，保存至: {output_dir}"); file_basename = os.path.splitext(os.path.basename(self.training_file_path))[0]
            try:
                for model_name, results in self.model_results_cache.items(): self.log_box.append(f"  - 正在绘制 {model_name} 的散点图..."); plot_output_path = os.path.join(output_dir, f"{file_basename}_{model_name}_scatter.png"); save_detailed_scatter_plot(y_true=results['y_true'], y_pred=results['y_pred'], model_name=model_name, file_basename=file_basename, output_path=plot_output_path)
                self.log_box.append("所有绘图已完成！"); QMessageBox.information(self, "成功", f"所有散点图已成功保存到:\n{output_dir}")
            except Exception as e: QMessageBox.critical(self, "绘图错误", f"生成图像时发生错误: {str(e)}")
    def task_error(self, error_message): self.log_box.append(f"\n错误: {error_message}"); self.cleanup_after_task(); QMessageBox.critical(self, "发生错误", error_message)

# ==============================================================================
#  程序入口
# ==============================================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())