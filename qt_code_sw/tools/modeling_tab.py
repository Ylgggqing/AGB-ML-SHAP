# -*- coding: utf-8 -*-

# 核心库和机器学习库
import sys
import os
import pandas as pd
import numpy as np
import joblib  # 用于保存和加载模型

# 导入新的模型库
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# 【新增】: 动态导入 TabPFN
try:
    from tabpfn import TabPFNRegressor

    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False

# PyQt5 相关库
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QGroupBox, QPushButton, QProgressBar, QTextEdit, QLabel,
                             QLineEdit, QFileDialog, QMessageBox, QDialog, QTabWidget,
                             QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox,
                             QDialogButtonBox, QCheckBox)
from PyQt5.QtCore import QThread, QObject, pyqtSignal

# 从共享的 plotting_utils 模块导入绘图函数
from .plotting_utils import save_detailed_scatter_plot


# ==============================================================================
#  后端工作线程 (ModelingWorker)
# ==============================================================================
class ModelingWorker(QObject):
    progress = pyqtSignal(str, int, int)
    model_finished = pyqtSignal(str, dict)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, file_path, custom_params, selected_models, save_model_path=None):
        super().__init__()
        self.file_path = file_path
        self.custom_params = custom_params
        self.selected_models = selected_models
        self.save_model_path = save_model_path
        self.is_running = True

    def stop(self):
        self.is_running = False

    def run(self):
        try:
            # 1. 数据准备
            data = pd.read_csv(self.file_path)
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            if self.save_model_path:
                scaler_path = os.path.join(self.save_model_path, "standard_scaler.joblib")
                joblib.dump(scaler, scaler_path)
                self.progress.emit(f"数据缩放器已保存至: {scaler_path}", 0, len(self.selected_models))

            # 2. 动态构建模型字典
            models_dict = {
                'SVR': SVR(**self.custom_params.get('SVR', {})),
                'XGBoost': xgb.XGBRegressor(**self.custom_params.get('XGBoost', {})),
                'RandomForest': RandomForestRegressor(**self.custom_params.get('RandomForest', {})),
                'CatBoost': cb.CatBoostRegressor(**self.custom_params.get('CatBoost', {'verbose': 0})),
                'LightGBM': lgb.LGBMRegressor(**self.custom_params.get('LightGBM', {})),
                'KNN': KNeighborsRegressor(**self.custom_params.get('KNN', {})),
                'LinearRegression': LinearRegression(**self.custom_params.get('LinearRegression', {}))
            }

            # 【新增】: 如果库可用，添加 TabPFN (不传入任何参数，使用默认值)
            if TABPFN_AVAILABLE:
                # TabPFN 不需要传递参数，直接使用空字典 {} 初始化
                models_dict['TabPFN'] = TabPFNRegressor()

            models_to_run = {name: model for name, model in models_dict.items() if name in self.selected_models}

            if not models_to_run:
                self.error.emit("没有选择任何模型来运行。")
                return

            # 3. 模型训练、评估和保存
            total_models = len(models_to_run)
            # 这些模型通常需要标准化数据效果才好
            models_need_scaling = ['SVR', 'KNN', 'LinearRegression']

            for i, (model_name, model) in enumerate(models_to_run.items()):
                if not self.is_running: self.error.emit("任务被用户中止。"); return

                self.progress.emit(f"正在处理 {model_name}...", i + 1, total_models)
                try:
                    # TabPFN 不需要标准化，且在样本量大时会自动处理，但传入原始数据即可
                    if model_name in models_need_scaling:
                        model.fit(X_train_scaled, y_train)
                        y_pred_test = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred_test = model.predict(X_test)

                    test_r2 = r2_score(y_test, y_pred_test)
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                    result_dict = {'test_r2': test_r2, 'test_rmse': test_rmse, 'y_true': y_test.values,
                                   'y_pred': y_pred_test}
                    self.model_finished.emit(model_name, result_dict)

                    if self.save_model_path:
                        model_filename = f"{model_name}_model.joblib"
                        model_save_path = os.path.join(self.save_model_path, model_filename)
                        joblib.dump(model, model_save_path)
                        self.progress.emit(f"  -> 模型 '{model_name}' 已保存至 {model_filename}", i + 1, total_models)

                except Exception as e:
                    self.error.emit(f"模型 '{model_name}' 训练失败: {str(e)}")
                    continue

            self.finished.emit()
        except Exception as e:
            self.error.emit(f"处理失败: {str(e)}")


# ==============================================================================
#  参数设置对话框 (ParametersDialog)
# ==============================================================================
class ParametersDialog(QDialog):
    def __init__(self, initial_params, parent=None):
        super().__init__(parent)
        self.setWindowTitle("调整模型参数")
        self.setMinimumWidth(450)
        self.params = initial_params
        main_layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.create_svr_tab()
        self.create_rf_tab()
        self.create_xgb_tab()
        self.create_lgbm_tab()
        self.create_catboost_tab()
        self.create_knn_tab()

        # 【修改】: TabPFN 不需要参数配置 Tab，因此这里不创建 TabPFN 的标签页

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

    def create_svr_tab(self):
        tab = QWidget()
        layout = QFormLayout(tab)
        self.svr_kernel = QComboBox()
        self.svr_kernel.addItems(['rbf', 'linear', 'poly', 'sigmoid'])
        self.svr_kernel.setCurrentText(self.params['SVR'].get('kernel', 'rbf'))
        layout.addRow("Kernel:", self.svr_kernel)
        self.svr_c = QDoubleSpinBox()
        self.svr_c.setRange(0.01, 10000.0)
        self.svr_c.setSingleStep(0.1)
        self.svr_c.setValue(self.params['SVR'].get('C', 1.0))
        layout.addRow("C (正则化参数):", self.svr_c)
        self.svr_epsilon = QDoubleSpinBox()
        self.svr_epsilon.setRange(0.0, 10.0)
        self.svr_epsilon.setSingleStep(0.01)
        self.svr_epsilon.setValue(self.params['SVR'].get('epsilon', 0.1))
        layout.addRow("Epsilon (ε):", self.svr_epsilon)
        self.svr_gamma = QDoubleSpinBox()
        self.svr_gamma.setRange(0.001, 100.0)
        self.svr_gamma.setSingleStep(0.01)
        self.svr_gamma.setValue(self.params['SVR'].get('gamma', 0.1))
        layout.addRow("Gamma:", self.svr_gamma)
        self.svr_degree = QSpinBox()
        self.svr_degree.setRange(1, 10)
        self.svr_degree.setValue(self.params['SVR'].get('degree', 3))
        layout.addRow("Degree (多项式阶数):", self.svr_degree)
        self.svr_kernel.currentTextChanged.connect(self._update_svr_fields)
        self._update_svr_fields(self.svr_kernel.currentText())
        self.tabs.addTab(tab, "SVR")

    def _update_svr_fields(self, kernel):
        is_poly = (kernel == 'poly')
        uses_gamma = (kernel in ['rbf', 'poly', 'sigmoid'])
        self.svr_degree.setEnabled(is_poly)
        self.svr_gamma.setEnabled(uses_gamma)

    def create_xgb_tab(self):
        tab = QWidget()
        layout = QFormLayout(tab)
        self.xgb_n_estimators = QSpinBox()
        self.xgb_n_estimators.setRange(10, 5000)
        self.xgb_n_estimators.setSingleStep(10)
        self.xgb_n_estimators.setValue(self.params['XGBoost'].get('n_estimators', 100))
        layout.addRow("N Estimators (树数量):", self.xgb_n_estimators)
        self.xgb_max_depth = QSpinBox()
        self.xgb_max_depth.setRange(1, 100)
        self.xgb_max_depth.setValue(self.params['XGBoost'].get('max_depth', 7))
        layout.addRow("Max Depth (最大深度):", self.xgb_max_depth)
        self.xgb_learning_rate = QDoubleSpinBox()
        self.xgb_learning_rate.setRange(0.001, 1.0)
        self.xgb_learning_rate.setSingleStep(0.01)
        self.xgb_learning_rate.setDecimals(3)
        self.xgb_learning_rate.setValue(self.params['XGBoost'].get('learning_rate', 0.05))
        layout.addRow("Learning Rate (学习率):", self.xgb_learning_rate)
        self.xgb_subsample = QDoubleSpinBox()
        self.xgb_subsample.setRange(0.1, 1.0)
        self.xgb_subsample.setSingleStep(0.1)
        self.xgb_subsample.setValue(self.params['XGBoost'].get('subsample', 0.8))
        layout.addRow("Subsample (样本采样率):", self.xgb_subsample)
        self.xgb_colsample_bytree = QDoubleSpinBox()
        self.xgb_colsample_bytree.setRange(0.1, 1.0)
        self.xgb_colsample_bytree.setSingleStep(0.1)
        self.xgb_colsample_bytree.setValue(self.params['XGBoost'].get('colsample_bytree', 0.8))
        layout.addRow("Colsample by Tree (特征采样率):", self.xgb_colsample_bytree)
        self.xgb_gamma = QDoubleSpinBox()
        self.xgb_gamma.setRange(0.0, 10.0)
        self.xgb_gamma.setSingleStep(0.1)
        self.xgb_gamma.setValue(self.params['XGBoost'].get('gamma', 0))
        layout.addRow("Gamma (最小损失减少):", self.xgb_gamma)
        self.xgb_reg_alpha = QDoubleSpinBox()
        self.xgb_reg_alpha.setRange(0.0, 100.0)
        self.xgb_reg_alpha.setSingleStep(0.1)
        self.xgb_reg_alpha.setValue(self.params['XGBoost'].get('reg_alpha', 0))
        layout.addRow("Reg Alpha (L1正则化):", self.xgb_reg_alpha)
        self.xgb_reg_lambda = QDoubleSpinBox()
        self.xgb_reg_lambda.setRange(0.0, 100.0)
        self.xgb_reg_lambda.setSingleStep(0.1)
        self.xgb_reg_lambda.setValue(self.params['XGBoost'].get('reg_lambda', 1))
        layout.addRow("Reg Lambda (L2正则化):", self.xgb_reg_lambda)
        self.tabs.addTab(tab, "XGBoost")

    def create_rf_tab(self):
        tab = QWidget()
        layout = QFormLayout(tab)
        self.rf_n_estimators = QSpinBox()
        self.rf_n_estimators.setRange(10, 5000)
        self.rf_n_estimators.setSingleStep(10)
        self.rf_n_estimators.setValue(self.params['RandomForest'].get('n_estimators', 100))
        layout.addRow("N Estimators (树数量):", self.rf_n_estimators)
        self.rf_max_depth = QSpinBox()
        self.rf_max_depth.setRange(1, 100)
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
        self.rf_max_features = QComboBox()
        self.rf_max_features.addItems(['sqrt', 'log2', 'None'])
        current_max_features = self.params['RandomForest'].get('max_features', 'sqrt')
        self.rf_max_features.setCurrentText(str(current_max_features))
        layout.addRow("Max Features:", self.rf_max_features)
        self.tabs.addTab(tab, "RandomForest")

    def create_lgbm_tab(self):
        tab = QWidget()
        layout = QFormLayout(tab)
        params = self.params.get('LightGBM', {})
        self.lgbm_n_estimators = QSpinBox(value=params.get('n_estimators', 100), minimum=10, maximum=5000,
                                          singleStep=10)
        self.lgbm_learning_rate = QDoubleSpinBox(value=params.get('learning_rate', 0.1), minimum=0.001, maximum=1.0,
                                                 singleStep=0.01, decimals=3)
        self.lgbm_max_depth = QSpinBox(value=params.get('max_depth', -1), minimum=-1, maximum=100)
        self.lgbm_num_leaves = QSpinBox(value=params.get('num_leaves', 31), minimum=2, maximum=1000)
        layout.addRow("N Estimators:", self.lgbm_n_estimators)
        layout.addRow("Learning Rate:", self.lgbm_learning_rate)
        layout.addRow("Max Depth (-1 is unlimited):", self.lgbm_max_depth)
        layout.addRow("Num Leaves:", self.lgbm_num_leaves)
        self.tabs.addTab(tab, "LightGBM")

    def create_catboost_tab(self):
        tab = QWidget()
        layout = QFormLayout(tab)
        params = self.params.get('CatBoost', {})
        self.cat_iterations = QSpinBox(value=params.get('iterations', 1000), minimum=10, maximum=10000, singleStep=50)
        self.cat_learning_rate = QDoubleSpinBox(value=params.get('learning_rate', 0.03), minimum=0.001, maximum=1.0,
                                                singleStep=0.01, decimals=3)
        self.cat_depth = QSpinBox(value=params.get('depth', 6), minimum=1, maximum=16)
        layout.addRow("Iterations:", self.cat_iterations)
        layout.addRow("Learning Rate:", self.cat_learning_rate)
        layout.addRow("Depth:", self.cat_depth)
        self.tabs.addTab(tab, "CatBoost")

    def create_knn_tab(self):
        tab = QWidget()
        layout = QFormLayout(tab)
        params = self.params.get('KNN', {})
        self.knn_n_neighbors = QSpinBox(value=params.get('n_neighbors', 5), minimum=1, maximum=100)
        self.knn_weights = QComboBox()
        self.knn_weights.addItems(['uniform', 'distance'])
        self.knn_weights.setCurrentText(params.get('weights', 'uniform'))
        layout.addRow("N Neighbors (K):", self.knn_n_neighbors)
        layout.addRow("Weights:", self.knn_weights)
        self.tabs.addTab(tab, "KNN")

    def get_parameters(self):
        max_features_val = self.rf_max_features.currentText()
        if max_features_val == 'None': max_features_val = None
        max_depth_val = self.rf_max_depth.value()
        if max_depth_val == 0: max_depth_val = None

        updated_params = {
            'SVR': {'kernel': self.svr_kernel.currentText(), 'C': self.svr_c.value(), 'gamma': self.svr_gamma.value(),
                    'epsilon': self.svr_epsilon.value(), 'degree': self.svr_degree.value()},
            'RandomForest': {'n_estimators': self.rf_n_estimators.value(), 'max_depth': max_depth_val,
                             'min_samples_split': self.rf_min_samples_split.value(),
                             'min_samples_leaf': self.rf_min_samples_leaf.value(), 'max_features': max_features_val,
                             'random_state': 43, 'n_jobs': -1},
            'XGBoost': {'n_estimators': self.xgb_n_estimators.value(), 'max_depth': self.xgb_max_depth.value(),
                        'learning_rate': self.xgb_learning_rate.value(), 'subsample': self.xgb_subsample.value(),
                        'colsample_bytree': self.xgb_colsample_bytree.value(), 'gamma': self.xgb_gamma.value(),
                        'reg_alpha': self.xgb_reg_alpha.value(), 'reg_lambda': self.xgb_reg_lambda.value(),
                        'random_state': 43, 'n_jobs': -1},
            'LightGBM': {'n_estimators': self.lgbm_n_estimators.value(),
                         'learning_rate': self.lgbm_learning_rate.value(), 'max_depth': self.lgbm_max_depth.value(),
                         'num_leaves': self.lgbm_num_leaves.value(), 'random_state': 43, 'n_jobs': -1},
            'CatBoost': {'iterations': self.cat_iterations.value(), 'learning_rate': self.cat_learning_rate.value(),
                         'depth': self.cat_depth.value(), 'random_state': 43, 'verbose': 0},
            'KNN': {'n_neighbors': self.knn_n_neighbors.value(), 'weights': self.knn_weights.currentText(),
                    'n_jobs': -1},
            'LinearRegression': {'n_jobs': -1}
        }

        # 【修改】: TabPFN 没有参数需要设置，返回空字典
        if TABPFN_AVAILABLE:
            updated_params['TabPFN'] = {}

        return updated_params


# ==============================================================================
#  主界面类 (ModelingTab)
# ==============================================================================
class ModelingTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker, self.thread, self.training_file_path = None, None, None
        self.model_results_cache = {}

        self.custom_params = {
            'SVR': {'kernel': 'rbf', 'C': 15.0, 'gamma': 0.1, 'epsilon': 0.1, 'degree': 3},
            'XGBoost': {'n_estimators': 1000, 'max_depth': 7, 'learning_rate': 0.05, 'subsample': 0.8,
                        'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1, 'random_state': 43,
                        'n_jobs': -1},
            'RandomForest': {'n_estimators': 500, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1,
                             'max_features': 'sqrt', 'random_state': 43, 'n_jobs': -1},
            'LightGBM': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': -1, 'num_leaves': 31,
                         'random_state': 43, 'n_jobs': -1},
            'CatBoost': {'iterations': 1000, 'learning_rate': 0.03, 'depth': 6, 'random_state': 43, 'verbose': 0},
            'KNN': {'n_neighbors': 5, 'weights': 'uniform', 'n_jobs': -1},
            'LinearRegression': {'n_jobs': -1},
            # 【修改】: TabPFN 参数为空
            'TabPFN': {}
        }

        main_layout = QVBoxLayout(self)

        # 1. 文件选择
        file_selection_layout = QHBoxLayout();
        file_selection_layout.addWidget(QLabel("训练文件:"))
        self.file_path_edit = QLineEdit(placeholderText="请选择一个CSV训练文件...", readOnly=True)
        self.select_file_button = QPushButton("选择文件...")
        file_selection_layout.addWidget(self.file_path_edit);
        file_selection_layout.addWidget(self.select_file_button)
        main_layout.addLayout(file_selection_layout)

        # 2. 模型选择
        model_selection_group = QGroupBox("选择要训练的模型")
        model_layout = QGridLayout(model_selection_group)
        self.model_checkboxes = {
            'SVR': QCheckBox("SVR"), 'XGBoost': QCheckBox("XGBoost"), 'RandomForest': QCheckBox("RandomForest"),
            'CatBoost': QCheckBox("CatBoost"),
            'LightGBM': QCheckBox("LightGBM"), 'KNN': QCheckBox("KNN"), 'LinearRegression': QCheckBox("线性回归"),
            # 【新增】: TabPFN 复选框
            'TabPFN': QCheckBox("TabPFN")
        }
        for name in ['XGBoost', 'RandomForest', 'SVR']: self.model_checkboxes.get(name, QCheckBox()).setChecked(True)

        # 检查 TabPFN 是否安装，如果没有则禁用
        if not TABPFN_AVAILABLE:
            self.model_checkboxes['TabPFN'].setEnabled(False)
            self.model_checkboxes['TabPFN'].setText("TabPFN (未安装)")
            self.model_checkboxes['TabPFN'].setToolTip("请运行 pip install tabpfn 安装此库")

        row, col = 0, 0
        for checkbox in self.model_checkboxes.values():
            model_layout.addWidget(checkbox, row, col);
            checkbox.toggled.connect(self._update_run_button_state);
            col += 1
            if col >= 4: col = 0; row += 1
        main_layout.addWidget(model_selection_group)

        # 3. 控制与选项
        options_layout = QHBoxLayout()
        self.tune_params_button = QPushButton("调整模型参数...")
        self.save_model_checkbox = QCheckBox("保存训练后的模型")
        options_layout.addWidget(self.tune_params_button);
        options_layout.addWidget(self.save_model_checkbox);
        options_layout.addStretch()
        main_layout.addLayout(options_layout)

        # 4. 运行与反馈
        self.run_button = QPushButton("开始建模")
        self.progress_bar = QProgressBar(value=0)
        self.log_box = QTextEdit(readOnly=True)
        main_layout.addWidget(self.run_button);
        main_layout.addWidget(self.progress_bar);
        main_layout.addWidget(self.log_box)

        # 信号连接
        self.select_file_button.clicked.connect(self.select_training_file)
        self.tune_params_button.clicked.connect(self.open_parameters_dialog)
        self.run_button.clicked.connect(self.start_modeling_task)
        self._update_run_button_state()

    def _update_run_button_state(self):
        file_selected = bool(self.training_file_path)
        any_model_selected = any(cb.isChecked() for cb in self.model_checkboxes.values())
        self.run_button.setEnabled(file_selected and any_model_selected)

    def select_training_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择训练数据文件", "", "CSV 文件 (*.csv);;所有文件 (*)")
        if path: self.training_file_path = path; self.file_path_edit.setText(path); self.log_box.append(
            f"已选择训练文件: {os.path.basename(path)}")
        self._update_run_button_state()

    def open_parameters_dialog(self):
        dialog = ParametersDialog(self.custom_params, self)
        if dialog.exec_() == QDialog.Accepted:
            self.custom_params = dialog.get_parameters();
            self.log_box.append("模型参数已更新。")
            QMessageBox.information(self, "成功", "模型参数已成功更新！")

    def start_modeling_task(self):
        if not self.training_file_path: return
        if self.thread and self.thread.isRunning(): QMessageBox.warning(self, "警告", "一个任务正在运行中..."); return
        selected_models = [name for name, cb in self.model_checkboxes.items() if cb.isChecked()]
        if not selected_models: QMessageBox.warning(self, "警告", "请至少选择一个要训练的模型！"); return

        save_path = None
        if self.save_model_checkbox.isChecked():
            save_path = QFileDialog.getExistingDirectory(self, "选择保存模型的文件夹")
            if not save_path: self.log_box.append("已取消保存模型，任务中止。"); return
            self.log_box.append(f"模型将被保存到: {save_path}")

        if self.thread: self.thread.quit(); self.thread.wait()
        self.model_results_cache.clear();
        self.set_ui_enabled(False)
        self.log_box.clear();
        self.progress_bar.setValue(0);
        self.log_box.append("开始建模任务...")

        self.thread = QThread()
        self.worker = ModelingWorker(self.training_file_path, self.custom_params, selected_models, save_path)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run);
        self.worker.progress.connect(self.update_progress)
        self.worker.model_finished.connect(self.log_model_result_and_cache);
        self.worker.finished.connect(self.task_complete)
        self.worker.error.connect(self.task_error)
        self.thread.start()

    def set_ui_enabled(self, enabled):
        self.run_button.setEnabled(enabled);
        self.select_file_button.setEnabled(enabled)
        self.tune_params_button.setEnabled(enabled)
        self.findChild(QGroupBox).setEnabled(enabled)  # Disables the model selection groupbox
        self.save_model_checkbox.setEnabled(enabled)
        if enabled: self._update_run_button_state()

    def cleanup_after_task(self):
        self.set_ui_enabled(True);
        self.progress_bar.setValue(0)
        if self.thread: self.thread.quit(); self.thread.wait(); self.thread, self.worker = None, None

    def update_progress(self, message, current, total):
        self.log_box.append(message);
        self.progress_bar.setMaximum(total);
        self.progress_bar.setValue(current)

    def log_model_result_and_cache(self, model_name, result_dict):
        r2, rmse = result_dict['test_r2'], result_dict['test_rmse'];
        self.log_box.append(
            f"  - {model_name} 完成: R²={r2:.3f}, RMSE={rmse:.3f}");
        self.model_results_cache[model_name] = result_dict

    def task_complete(self):
        self.log_box.append("\n所有任务完成！");
        self.cleanup_after_task()
        if QMessageBox.question(self, '任务完成', "建模已完成。是否要生成精度散点图？", QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.Yes) == QMessageBox.Yes:
            self.generate_plots()

    def generate_plots(self):
        if not self.model_results_cache: QMessageBox.information(self, "无结果", "没有可供绘图的模型结果。"); return
        output_dir = QFileDialog.getExistingDirectory(self, "选择保存绘图的文件夹")
        if output_dir:
            self.log_box.append(f"\n开始生成散点图，保存至: {output_dir}")
            file_basename = os.path.splitext(os.path.basename(self.training_file_path))[0]
            try:
                for model_name, results in self.model_results_cache.items():
                    self.log_box.append(f"  - 正在绘制 {model_name} 的散点图...")
                    plot_output_path = os.path.join(output_dir, f"{file_basename}_{model_name}_scatter.png")
                    save_detailed_scatter_plot(y_true=results['y_true'], y_pred=results['y_pred'],
                                               model_name=model_name, file_basename=file_basename,
                                               output_path=plot_output_path)
                self.log_box.append("所有绘图已完成！");
                QMessageBox.information(self, "成功", f"所有散点图已成功保存到:\n{output_dir}")
            except Exception as e:
                QMessageBox.critical(self, "绘图错误", f"生成图像时发生错误: {str(e)}")

    def task_error(self, error_message):
        self.log_box.append(f"\n错误: {error_message}");
        self.cleanup_after_task();
        QMessageBox.critical(self,
                             "发生错误",
                             error_message)
