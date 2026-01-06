ForST-XAI v2.1 includes the following core modules (tabs):
1. LiDAR & Remote Sensing Tools
DEM/DSM/CHM Generation: Generate Digital Elevation Models, Surface Models, and Canopy Height Models from LAS/LAZ point clouds using IDW interpolation.
Watershed Segmentation: Individual tree segmentation based on CHM to extract tree crown polygons and tree tops.
Format Conversion: Batch convert .las/.laz files to .csv format.
Point Cloud Sampling: Batch random sampling for large point cloud CSV datasets.
2. Feature Extraction
Extracts metrics across three spatial dimensions:
LiDAR Tree-Level: Height stats, intensity stats, canopy structure (Volume, Surface Area, Convex Hull), and density metrics.
LiDAR Plot-Level: Includes standard metrics plus advanced volumetric indices like 3DVI (3D Vegetation Index) and 3DPI (3D Profile Index).
Sentinel-2 Imagery: Extracts Vegetation Indices (NDVI, NDWI) and GLCM Texture features (Entropy, Contrast, Cluster Shade, etc.).
3. AGB Aggregation (Volume-Weighted)
Multi-Scale Scaling: A specialized algorithm that aggregates individual tree AGB to the plot level based on the geometric relationship between tree convex hulls and plot grids.
Parallel Processing: Supports multi-core processing for handling large datasets efficiently.
4. Machine Learning Modeling
Algorithm Suite: Integrates RandomForest, XGBoost, LightGBM, CatBoost, SVR, KNN, LinearRegression, and the state-of-the-art TabPFN (Transformer for Tabular Data).
Hyperparameter Tuning: GUI-based parameter adjustment for all models.
Evaluation: Automatic calculation of R² and RMSE, with support for generating "Observed vs. Predicted" scatter plots.
Model Persistence: Save trained models (.joblib) and data scalers for future use.
5. Prediction & Application
Deployment: Load pre-trained models to perform batch predictions on new datasets.
Auto-Preprocessing: Automatically detects and applies the necessary StandardScaler if required by the model.
6. SHAP Explainable AI (XAI)
Visualization: Generates SHAP Summary Plots, Dependence Plots, and Spatial Distribution Maps to reveal feature importance and interaction effects.
 Installation
1. Environment Setup
It is highly recommended to use Anaconda to create a virtual environment to avoid dependency conflicts with geospatial libraries (GDAL, Rasterio).
code
Bash
conda create -n forst_xai python=3.9
conda activate forst_xai
2. Install Dependencies
Install geospatial libraries via conda-forge first:
code
Bash
conda install -c conda-forge gdal rasterio geopandas shapely laspy
Then install the remaining Python libraries via pip:
code
Bash
pip install pyqt5 pandas numpy scipy scikit-learn scikit-image matplotlib seaborn joblib
pip install xgboost lightgbm catboost tabpfn shap alphashape
Note: alphashape might require rtree. Please refer to the official documentation if you encounter installation issues.
Quick Start
Clone or Download this repository.
Ensure all dependencies are installed.
Run the main application:
code
Bash
python main.py
Note for Windows Users
This project uses multiprocessing for parallel tasks. When packaging as an .exe or running on Windows, the entry point in main.py includes the necessary guard code:
code
Python
if __name__ == '__main__':
    multiprocessing.freeze_support()
    # ... application start
Project Structure
code
Code
ForST-XAI/
│
├── main.py                     # Entry point, Main Window
├── plotting_utils.py           # Shared plotting utilities (Scatter plots, etc.)
│
├── tools/                      # Feature Modules
│   ├── __init__.py
│   ├── lidar_tools_tab.py      # LiDAR & RS Tools
│   ├── tree_feature_tab.py     # Tree-level Feature Extraction
│   ├── plot_feature_tab.py     # Plot-level Feature Extraction
│   ├── s2_feature_tab.py       # Sentinel-2 Feature Extraction
│   ├── agb_aggregation_tab.py  # AGB Aggregation Logic
│   ├── modeling_tab.py         # ML Modeling & Training
│   ├── prediction_tab.py       # Model Prediction
│   └── shap_visualizer_tab.py  # SHAP Analysis
│
└── requirements.txt            # (Optional) List of dependencies
Usage Tips
Data Formats:
Point Cloud CSV: Must contain columns named X, Y, Z. Some features require an Intensity column.
Modeling CSV: The last column is assumed to be the target variable (Label/Y), and preceding columns are features (X).
TabPFN Model:
TabPFN is a Transformer model that performs exceptionally well on small-to-medium datasets without extensive tuning.
Note: It may require an internet connection on the first run to download pretrained weights.
Memory Usage:
Please monitor memory usage when processing large LAS files or high-resolution Rasters.
The AGB Aggregation module is CPU-intensive when running with multiple workers.
