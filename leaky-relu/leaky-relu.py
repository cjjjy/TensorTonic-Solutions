import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    # Write code here
    x = np.array(x)
    
    # 使用 np.where 进行条件选择：
    # 如果 x >= 0，保持不变；否则乘以 alpha
    return np.where(x >= 0, x, alpha * x)