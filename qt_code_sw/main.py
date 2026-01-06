# --- START OF FILE main.py ---

# -*- coding: utf-8 -*-
import sys
import multiprocessing
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget

# 从 tools 包中导入所有重构后的功能模块 (QWidget)
from tools.modeling_tab import ModelingTab
# 【新增】: 导入新的预测功能选项卡
from tools.prediction_tab import PredictionTab
from tools.plot_feature_tab import PlotFeatureExtractorTab
from tools.tree_feature_tab import TreeFeatureExtractorTab
from tools.s2_feature_tab import S2FeatureExtractorTab
from tools.agb_aggregation_tab import AggregationTab
from tools.shap_visualizer_tab import ShapVisualizerTab
from tools.lidar_tools_tab import LidarToolsTab


class MainAppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ForST-XAI v2.1")  # 版本号可以更新一下
        self.setGeometry(75, 75, 700, 700)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.init_tabs()

    def init_tabs(self):
        # LiDAR 与遥感工具
        self.lidar_tools_tab = LidarToolsTab()
        self.tabs.addTab(self.lidar_tools_tab, "点云与遥感工具")

        # 特征提取
        self.tree_feature_tab = TreeFeatureExtractorTab()
        self.tabs.addTab(self.tree_feature_tab, "LiDAR 单木特征提取")
        self.plot_feature_tab = PlotFeatureExtractorTab()
        self.tabs.addTab(self.plot_feature_tab, "LiDAR 样地特征提取")
        self.s2_feature_tab = S2FeatureExtractorTab()
        self.tabs.addTab(self.s2_feature_tab, "S2 影像特征提取")

        # 数据处理与建模
        self.agb_aggregation_tab = AggregationTab()
        self.tabs.addTab(self.agb_aggregation_tab, "AGB 体积聚合法")

        # 机器学习建模与预测（逻辑分离）
        self.modeling_tab = ModelingTab()
        self.tabs.addTab(self.modeling_tab, "机器学习建模与训练")  # 名称更具体

        # 【新增】: 添加独立的预测选项卡
        self.prediction_tab = PredictionTab()
        self.tabs.addTab(self.prediction_tab, "模型预测与应用")

        # 可解释性分析
        self.shap_visualizer_tab = ShapVisualizerTab()
        self.tabs.addTab(self.shap_visualizer_tab, "SHAP 可解释性分析")

    def closeEvent(self, event):
        # 可以在这里添加一些清理逻辑，例如确保所有后台线程都已停止
        event.accept()


if __name__ == '__main__':
    # 对于打包成 exe 很重要，尤其是在 Windows 和 macOS 上
    multiprocessing.freeze_support()

    app = QApplication(sys.argv)
    main_window = MainAppWindow()
    main_window.show()
    sys.exit(app.exec_())