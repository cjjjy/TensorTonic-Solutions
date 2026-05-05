import numpy as np

def tanh(x):
    """
    实现 Tanh 激活函数。
    支持标量、列表或 NumPy 数组。
    """
    # 转换为 NumPy 数组以支持高效的向量化运算
    x = np.array(x)
    
    # 方式 A：直接调用 NumPy 内置函数（性能最高且经过数值优化）
    result = np.tanh(x)
    
    # 方式 B：手动实现公式（用于理解原理）
    # e_x = np.exp(x)
    # e_neg_x = np.exp(-x)
    # result = (e_x - e_neg_x) / (e_x + e_neg_x)
    
    return result