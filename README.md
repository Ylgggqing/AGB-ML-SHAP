ForST-XAI includes the following core modules :
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
Algorithm Suite: Integrates RandomForest, XGBoost, LightGBM, CatBoost, SVR, KNN, LinearRegression, and the state-of-the-art TabPFN.
Hyperparameter Tuning: GUI-based parameter adjustment for all models.
Evaluation: Automatic calculation of R² and RMSE, with support for generating "Observed vs. Predicted" scatter plots.
Model Persistence: Save trained models (.joblib) and data scalers for future use.
5. Prediction & Application
Deployment: Load pre-trained models to perform batch predictions on new datasets.
Auto-Preprocessing: Automatically detects and applies the necessary StandardScaler if required by the model.
6. SHAP Explainable AI (XAI)
Visualization: Generates SHAP Summary Plots, Dependence Plots, and Spatial Distribution Maps to reveal feature importance and interaction effects.
