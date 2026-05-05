import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    计算预测值 y_pred 和真实值 y_true 之间的均方误差。
    返回一个 float 类型的 MSE。
    """
    # 确保输入是 NumPy 数组，以支持向量化减法和平方运算
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    # 1. 计算残差 (Residuals): y_pred - y_true
    # 2. 计算平方: (y_pred - y_true) ** 2
    # 3. 计算均值: np.mean(...)
    mse = np.mean((y_pred - y_true) ** 2)
    
    return float(mse)