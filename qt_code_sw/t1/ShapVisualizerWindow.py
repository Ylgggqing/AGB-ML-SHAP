# -*- coding: utf-8 -*-

# ... (所有 import 语句保持不变) ...
import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import shap
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QProgressBar, QTextEdit,
                             QLabel, QLineEdit, QFileDialog, QMessageBox, QWidget, QVBoxLayout,
                             QGroupBox, QGridLayout, QComboBox, QCheckBox,
                             QDialog, QFormLayout, QSpinBox, QDoubleSpinBox, QDialogButtonBox)
from PyQt5.QtCore import QThread, QObject, pyqtSignal

matplotlib.use('Agg')


# ==============================================================================
#  辅助模块
# ==============================================================================

# --- 参数对话框 (已修正) ---
class ParametersDialog(QDialog):
    def __init__(self, model_type, initial_params, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"调整 {model_type} 参数")
        self.setMinimumWidth(450)
        self.params = initial_params.get(model_type, {})
        self.model_type = model_type

        main_layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        main_layout.addLayout(form_layout)

        self.widgets = {}
        if model_type == 'RandomForest':
            self.widgets['n_estimators'] = QSpinBox(value=self.params.get('n_estimators', 500), minimum=10,
                                                    maximum=5000, singleStep=10)
            self.widgets['max_depth'] = QSpinBox(value=self.params.get('max_depth', 20) or 0, minimum=0, maximum=100)
            self.widgets['max_depth'].setSpecialValueText("None (无限制)")
            self.widgets['min_samples_split'] = QSpinBox(value=self.params.get('min_samples_split', 2), minimum=2,
                                                         maximum=100)
            self.widgets['min_samples_leaf'] = QSpinBox(value=self.params.get('min_samples_leaf', 1), minimum=1,
                                                        maximum=100)

            # --- 修正点 ---
            self.widgets['max_features'] = QComboBox()
            self.widgets['max_features'].addItems(['sqrt', 'log2', 'None'])  # 移除了 'auto'
            current_mf = self.params.get('max_features', 'sqrt')
            self.widgets['max_features'].setCurrentText(str(current_mf))

            form_layout.addRow("N Estimators:", self.widgets['n_estimators'])
            form_layout.addRow("Max Depth:", self.widgets['max_depth'])
            form_layout.addRow("Min Samples Split:", self.widgets['min_samples_split'])
            form_layout.addRow("Min Samples Leaf:", self.widgets['min_samples_leaf'])
            form_layout.addRow("Max Features:", self.widgets['max_features'])

        elif model_type == 'XGBoost':
            self.widgets['n_estimators'] = QSpinBox(value=self.params.get('n_estimators', 500), minimum=10,
                                                    maximum=5000, singleStep=10)
            self.widgets['max_depth'] = QSpinBox(value=self.params.get('max_depth', 20), minimum=1, maximum=100)
            self.widgets['learning_rate'] = QDoubleSpinBox(value=self.params.get('learning_rate', 0.05), minimum=0.001,
                                                           maximum=1.0, singleStep=0.01, decimals=3)
            self.widgets['subsample'] = QDoubleSpinBox(value=self.params.get('subsample', 0.8), minimum=0.1,
                                                       maximum=1.0, singleStep=0.1)
            self.widgets['colsample_bytree'] = QDoubleSpinBox(value=self.params.get('colsample_bytree', 0.8),
                                                              minimum=0.1, maximum=1.0, singleStep=0.1)
            self.widgets['gamma'] = QDoubleSpinBox(value=self.params.get('gamma', 0), minimum=0.0, maximum=10.0,
                                                   singleStep=0.1)
            self.widgets['reg_alpha'] = QDoubleSpinBox(value=self.params.get('reg_alpha', 0), minimum=0.0,
                                                       maximum=100.0, singleStep=0.1)
            self.widgets['reg_lambda'] = QDoubleSpinBox(value=self.params.get('reg_lambda', 1), minimum=0.0,
                                                        maximum=100.0, singleStep=0.1)
            form_layout.addRow("N Estimators:", self.widgets['n_estimators']);
            form_layout.addRow("Max Depth:", self.widgets['max_depth']);
            form_layout.addRow("Learning Rate:", self.widgets['learning_rate']);
            form_layout.addRow("Subsample:", self.widgets['subsample']);
            form_layout.addRow("Colsample by Tree:", self.widgets['colsample_bytree']);
            form_layout.addRow("Gamma:", self.widgets['gamma']);
            form_layout.addRow("Reg Alpha (L1):", self.widgets['reg_alpha']);
            form_layout.addRow("Reg Lambda (L2):", self.widgets['reg_lambda'])

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel);
        button_box.accepted.connect(self.accept);
        button_box.rejected.connect(self.reject);
        main_layout.addWidget(button_box)

    def get_parameters(self):
        updated = {}
        for name, widget in self.widgets.items():
            if isinstance(widget, QComboBox):
                val = widget.currentText()
                updated[name] = None if val == 'None' else val
            elif isinstance(widget, QSpinBox) and widget.specialValueText() and widget.value() == widget.minimum():
                updated[name] = None
            else:
                updated[name] = widget.value()

        if self.model_type == 'XGBoost':
            updated.update({'objective': 'reg:squarederror', 'random_state': 43, 'n_jobs': -1})
        else:
            updated.update({'random_state': 43, 'n_jobs': -1})
        return updated


# ... (绘图函数和 ShapWorker 类保持不变) ...
def plot_summary(shap_values, X, output_path): plt.figure(); shap.summary_plot(shap_values, X, show=False,
                                                                               plot_size=None); plt.savefig(output_path,
                                                                                                            dpi=300,
                                                                                                            bbox_inches='tight'); plt.close()


def plot_dependence(shap_values, X, feature, output_path): plt.figure(); shap.dependence_plot(feature, shap_values, X,
                                                                                              show=False); plt.savefig(
    output_path, dpi=300, bbox_inches='tight'); plt.close()


def plot_spatial(gdf, column, output_path, title=''): fig, ax = plt.subplots(1, 1, figsize=(10, 10)); gdf.plot(
    column=column, ax=ax, legend=True, markersize=10, cmap='viridis'); ax.set_title(title or column); ax.set_xlabel(
    "X Coordinate"); ax.set_ylabel("Y Coordinate"); plt.savefig(output_path, dpi=300, bbox_inches='tight'); plt.close(
    fig)


class ShapWorker(QObject):
    progress = pyqtSignal(int, str);
    finished = pyqtSignal();
    error = pyqtSignal(str)

    def __init__(self, params):
        super().__init__(); self.params = params; self.is_running = True

    def stop(self):
        self.is_running = False

    def run(self):
        try:
            p = self.params;
            self.progress.emit(5, "正在加载特征数据...");
            data = pd.read_csv(p['feature_csv'], index_col=0);
            X = data.iloc[:, :-1];
            y = data.iloc[:, -1];
            self.progress.emit(15, f"正在训练 {p['model_type']} 模型...");
            if p['model_type'] == 'RandomForest':
                model = RandomForestRegressor(**p['model_params'])
            else:
                model = xgb.XGBRegressor(**p['model_params'])
            model.fit(X, y);
            if not self.is_running: self.error.emit("任务中止。"); return
            self.progress.emit(40, "正在计算SHAP值 (这可能需要一些时间)...");
            explainer = shap.TreeExplainer(model);
            shap_values = explainer.shap_values(X);
            if not self.is_running: self.error.emit("任务中止。"); return
            output_folder = p['output_folder'];
            os.makedirs(output_folder, exist_ok=True)
            if p['plot_summary']: self.progress.emit(60, "正在生成摘要图..."); path = os.path.join(output_folder,
                                                                                                   "summary_plot.png"); plot_summary(
                shap_values, X, path)
            if p['plot_dependence']: self.progress.emit(70,
                                                        f"正在为特征 '{p['dependence_feature']}' 生成依赖图..."); path = os.path.join(
                output_folder, f"dependence_plot_{p['dependence_feature']}.png"); plot_dependence(shap_values, X, p[
                'dependence_feature'], path)
            if p['plot_spatial']: self.progress.emit(80, "正在准备空间数据..."); coords_df = pd.read_csv(
                p['coords_csv'], index_col=0); shap_df = pd.DataFrame(shap_values, index=X.index,
                                                                      columns=[f'SHAP_{col}' for col in
                                                                               X.columns]); combined_df = pd.concat(
                [X, shap_df, coords_df], axis=1).dropna(subset=['X', 'Y']); gdf = gpd.GeoDataFrame(combined_df,
                                                                                                   geometry=gpd.points_from_xy(
                                                                                                       combined_df.X,
                                                                                                       combined_df.Y)); self.progress.emit(
                90, f"正在为特征 '{p['spatial_feature']}' 生成空间分布图..."); path = os.path.join(output_folder,
                                                                                                   f"spatial_map_{p['spatial_feature']}.png"); plot_spatial(
                gdf, p['spatial_feature'], path)
            self.progress.emit(100, "所有任务完成！");
            self.finished.emit()
        except Exception as e:
            self.error.emit(f"发生错误: {str(e)}")


# ==============================================================================
#  前端界面 (Frontend UI) - ShapVisualizerWindow (已修正)
# ==============================================================================
class ShapVisualizerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SHAP 可视化分析工具")
        self.setGeometry(100, 100, 700, 700)
        self.thread, self.worker = None, None

        # --- 修正点 ---
        self.model_params = {
            'RandomForest': {
                'n_estimators': 500, 'max_depth': 20, 'min_samples_split': 2,
                'min_samples_leaf': 1, 'max_features': 'sqrt',  # 'auto' -> 'sqrt'
                'random_state': 43, 'n_jobs': -1
            },
            'XGBoost': {
                'n_estimators': 500, 'max_depth': 20, 'learning_rate': 0.05,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0,
                'reg_alpha': 0, 'reg_lambda': 1,
                'objective': 'reg:squarederror', 'random_state': 43, 'n_jobs': -1
            }
        }

        # ... (UI布局和信号连接部分保持不变) ...
        central_widget = QWidget();
        self.setCentralWidget(central_widget);
        main_layout = QVBoxLayout(central_widget);
        group1 = QGroupBox("1. 模型与数据设置");
        layout1 = QGridLayout(group1);
        self.model_combo = QComboBox();
        self.model_combo.addItems(['RandomForest', 'XGBoost']);
        self.params_button = QPushButton("调整参数...");
        self.feature_csv_edit = QLineEdit();
        self.coords_csv_edit = QLineEdit();
        layout1.addWidget(QLabel("选择模型:"), 0, 0);
        layout1.addWidget(self.model_combo, 0, 1);
        layout1.addWidget(self.params_button, 0, 2);
        layout1.addWidget(QLabel("特征数据CSV:"), 1, 0);
        layout1.addWidget(self.feature_csv_edit, 1, 1);
        layout1.addWidget(
            QPushButton("浏览...", clicked=lambda: self.select_file(self.feature_csv_edit, "CSV (*.csv)")), 1, 2);
        layout1.addWidget(QLabel("坐标数据CSV:"), 2, 0);
        layout1.addWidget(self.coords_csv_edit, 2, 1);
        layout1.addWidget(QPushButton("浏览...", clicked=lambda: self.select_file(self.coords_csv_edit, "CSV (*.csv)")),
                          2, 2);
        main_layout.addWidget(group1);
        self.feature_csv_edit.textChanged.connect(self.populate_feature_combos);
        self.feature_csv_edit.textChanged.connect(self._check_inputs);
        self.coords_csv_edit.textChanged.connect(self._check_inputs);
        group2 = QGroupBox("2. 可视化输出选择");
        layout2 = QGridLayout(group2);
        self.summary_check = QCheckBox("摘要图 (Summary Plot)");
        self.summary_check.setChecked(True);
        self.dependence_check = QCheckBox("依赖图 (Dependence Plot)");
        self.spatial_check = QCheckBox("空间分布图 (Spatial Map)");
        self.dependence_combo = QComboBox();
        self.dependence_combo.setEnabled(False);
        self.spatial_combo = QComboBox();
        self.spatial_combo.setEnabled(False);
        self.output_folder_edit = QLineEdit();
        layout2.addWidget(self.summary_check, 0, 0, 1, 3);
        layout2.addWidget(self.dependence_check, 1, 0);
        layout2.addWidget(self.dependence_combo, 1, 1, 1, 2);
        layout2.addWidget(self.spatial_check, 2, 0);
        layout2.addWidget(self.spatial_combo, 2, 1, 1, 2);
        layout2.addWidget(QLabel("输出文件夹:"), 3, 0);
        layout2.addWidget(self.output_folder_edit, 3, 1);
        layout2.addWidget(QPushButton("浏览...", clicked=self.select_folder), 3, 2);
        main_layout.addWidget(group2);
        self.output_folder_edit.textChanged.connect(self._check_inputs);
        self.start_button = QPushButton("开始分析");
        self.progress_bar = QProgressBar();
        self.log_box = QTextEdit();
        main_layout.addWidget(self.start_button);
        main_layout.addWidget(self.progress_bar);
        main_layout.addWidget(self.log_box);
        self.params_button.clicked.connect(self.open_params_dialog);
        self.dependence_check.toggled.connect(self.dependence_combo.setEnabled);
        self.dependence_check.toggled.connect(self._check_inputs);
        self.spatial_check.toggled.connect(self.spatial_combo.setEnabled);
        self.spatial_check.toggled.connect(self._check_inputs);
        self.start_button.clicked.connect(self.start_analysis);
        self._check_inputs()

    # ... (ShapVisualizerWindow 的所有方法保持不变) ...
    def select_file(self, line_edit, file_filter):
        path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", file_filter); path and line_edit.setText(path)

    def select_folder(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出文件夹"); path and self.output_folder_edit.setText(path)

    def populate_feature_combos(self):
        path = self.feature_csv_edit.text();
        self.dependence_combo.clear();
        self.spatial_combo.clear()
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, index_col=0, nrows=1);
                features = df.columns[:-1].tolist();
                shap_features = [f'SHAP_{f}' for f in features];
                self.dependence_combo.addItems(features);
                self.spatial_combo.addItems(features + shap_features)
            except Exception as e:
                self.log_box.append(f"警告: 无法解析特征文件: {e}")

    def open_params_dialog(self):
        model_type = self.model_combo.currentText(); dialog = ParametersDialog(model_type, self.model_params,
                                                                               self); dialog.exec_() == QDialog.Accepted and (
            self.model_params.update({model_type: dialog.get_parameters()}),
            QMessageBox.information(self, "成功", f"{model_type} 参数已更新。"))

    def _check_inputs(self):
        feature_ok = os.path.isfile(self.feature_csv_edit.text()); output_ok = os.path.isdir(
            self.output_folder_edit.text()); coords_ok = not self.spatial_check.isChecked() or os.path.isfile(
            self.coords_csv_edit.text()); self.start_button.setEnabled(feature_ok and output_ok and coords_ok)

    def start_analysis(self):
        params = {'model_type': self.model_combo.currentText(),
                  'model_params': self.model_params[self.model_combo.currentText()],
                  'feature_csv': self.feature_csv_edit.text(), 'coords_csv': self.coords_csv_edit.text(),
                  'output_folder': self.output_folder_edit.text(), 'plot_summary': self.summary_check.isChecked(),
                  'plot_dependence': self.dependence_check.isChecked(),
                  'dependence_feature': self.dependence_combo.currentText(),
                  'plot_spatial': self.spatial_check.isChecked(), 'spatial_feature': self.spatial_combo.currentText()}
        if not any([params['plot_summary'], params['plot_dependence'], params['plot_spatial']]): QMessageBox.warning(
            self, "无操作", "请至少选择一种可视化输出。"); return
        self.set_ui_enabled(False);
        self.log_box.clear();
        self.progress_bar.setValue(0);
        self.log_box.append("任务开始...")
        self.thread = QThread();
        self.worker = ShapWorker(params);
        self.worker.moveToThread(self.thread);
        self.thread.started.connect(self.worker.run);
        self.worker.progress.connect(self.update_progress);
        self.worker.finished.connect(self.task_finished);
        self.worker.error.connect(self.task_error);
        self.thread.start()

    def set_ui_enabled(self, enabled):
        [child.setEnabled(enabled) for child in self.findChildren(QGroupBox)]; self.start_button.setEnabled(enabled)

    def update_progress(self, percent, message):
        self.progress_bar.setValue(percent); self.log_box.append(message)

    def task_finished(self):
        self.set_ui_enabled(True); self.thread.quit(); self.thread.wait(); QMessageBox.information(self, "成功",
                                                                                                   f"SHAP分析与可视化完成！\n结果已保存到: {self.output_folder_edit.text()}")

    def task_error(self, message):
        self.set_ui_enabled(True); self.thread and self.thread.isRunning() and (self.thread.quit(),
                                                                                self.thread.wait()); QMessageBox.critical(
            self, "错误", message)


# ==============================================================================
#  程序入口
# ==============================================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ShapVisualizerWindow()
    window.show()
    sys.exit(app.exec_())