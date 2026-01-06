import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib.ticker import MultipleLocator
import matplotlib

# ==============================================================================
#  全局字体和样式设置
# ==============================================================================
try:
    font_family = 'Times New Roman'
    font = {'family': font_family, "size": 12, 'weight': 'bold'}
    matplotlib.rc('font', **font)
    plt.rcParams['font.family'] = 'Times New Roman'
    sns.set(style='ticks', font_scale=1.1, font='Times New Roman')
except Exception as e:
    print(f"Warning: Could not set 'Times New Roman' font. Matplotlib might use a default font. Error: {e}")


# ==============================================================================
#  核心绘图函数
# ==============================================================================
def save_detailed_scatter_plot(y_true, y_pred, model_name, file_basename, output_path):
    """
    为单个模型绘制详细的散点图，并直接保存到指定路径。
    此函数不调用 plt.show()。

    参数:
    y_true (np.array): 真实的Y值。
    y_pred (np.array): 模型预测的Y值。
    model_name (str): 模型的名称，用于标题和文件名。
    file_basename (str): 原始数据文件的基本名称，用于标题。
    output_path (str): 图像文件的完整保存路径 (e.g., 'C:/.../SVR_plot.png')。
    """
    fig, ax = plt.subplots(dpi=300, figsize=(10, 10))

    y = np.array(y_true).reshape(-1, 1)
    pre = np.array(y_pred).reshape(-1, 1)

    R2_val = r2_score(y, pre)
    RMSE = np.sqrt(mean_squared_error(y, pre))

    OLS1 = LinearRegression()
    OLS1.fit(y, pre)
    fit = OLS1.predict(y)
    b = OLS1.intercept_[0]
    k = OLS1.coef_[0][0]
    Eq1 = f'y = {k:.3f}x + {b:.3f}' if b >= 0 else f'y = {k:.3f}x - {abs(b):.3f}'

    ax.scatter(y_true, y_pred, c='#FD9999', s=75, alpha=0.9, marker='o')
    ax.plot(y_true, fit, color='Indigo', alpha=0.95, linewidth=2, label='Regression Line')
    ax.plot((0, 1), (0, 1), transform=ax.transAxes, ls='--', c='k', lw=2, label="1:1 line")

    text_content = f"{Eq1}\n" \
                   f"R² = {R2_val:.3f}\n" \
                   f"RMSE = {RMSE:.3f}"
    ax.text(0.05, 0.95, text_content, transform=ax.transAxes, fontsize=25,
            fontweight='bold', va='top', bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.8))

    all_values = np.concatenate([y, pre]).flatten()
    min_val, max_val = all_values.min(), all_values.max()
    margin = (max_val - min_val) * 0.05
    ax_min = max(0, min_val - margin)
    ax_max = max_val + margin

    ax.set_xlabel('Measured Values', fontweight='bold', fontsize=25)
    ax.set_ylabel('Predicted Values', fontweight='bold', fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlim(ax_min, ax_max)
    ax.set_ylim(ax_min, ax_max)
    ax.set_aspect('equal')

    tick_range = ax_max - ax_min
    if tick_range > 0:
        num_ticks = 5
        tick_interval = tick_range / num_ticks
        power = 10 ** np.floor(np.log10(tick_interval))
        if power > 0:
            tick_interval = np.ceil(tick_interval / power) * power
            if tick_interval > 0:
                ax.xaxis.set_major_locator(MultipleLocator(tick_interval))
                ax.yaxis.set_major_locator(MultipleLocator(tick_interval))

    ax.set_title(f'"{model_name}" Model Performance on "{file_basename}"', fontsize=20, pad=20)

    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error saving plot to {output_path}: {e}")
    finally:
        plt.close(fig)  # 确保关闭图形，释放内存