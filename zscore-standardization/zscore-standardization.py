import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):
    """
    标准化 X: (X - mean)/std。如果是 2D 且 axis=0，则按列进行。
    返回 np.ndarray (float)。
    """
    # 确保输入为 numpy 数组
    X = np.array(X, dtype=float)
    
    # 计算指定轴的均值和标准差
    mean = np.mean(X, axis=axis, keepdims=True)
    std = np.std(X, axis=axis, keepdims=True)
    
    # 按照公式计算 Z-Score
    # 引入 eps (epsilon) 是为了防止分母为 0（即标准差为 0 的情况）
    Z = (X - mean) / (std + eps)
    
    return Z